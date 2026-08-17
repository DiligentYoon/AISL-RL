from __future__ import annotations

import os
import torch

import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from lib.domain_randomizer.randomizer import _validate_scale_range, _randomize_prop_by_op, get_isaac_sim_version

def reset_robot_and_object_root_state_uniform(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("jig"),
    object_relative_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    object_relative_yaw: float = 0.0,
):
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # default states
    robot_root_states = robot.data.default_root_state[env_ids].clone()
    obj_root_states = obj.data.default_root_state[env_ids].clone()

    # shared pose random sample
    pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    pose_ranges = torch.tensor(
        [pose_range.get(key, (0.0, 0.0)) for key in pose_keys],
        device=robot.device,
        dtype=torch.float32,
    )
    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=robot.device
    )

    pos_delta = pose_samples[:, 0:3]      # shared sampled xyz
    rpy_delta = pose_samples[:, 3:6]      # shared sampled rpy

    # robot pose
    robot_positions = (
        robot_root_states[:, 0:3]
        + env.scene.env_origins[env_ids]
        + pos_delta
    )

    robot_orient_delta = math_utils.quat_from_euler_xyz(
        rpy_delta[:, 0], rpy_delta[:, 1], rpy_delta[:, 2]
    )
    robot_orientations = math_utils.quat_mul(
        robot_root_states[:, 3:7], robot_orient_delta
    )

    # robot velocity
    vel_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    vel_ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in vel_keys],
        device=robot.device,
        dtype=torch.float32,
    )
    vel_samples = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=robot.device
    )
    robot_velocities = robot_root_states[:, 7:13] + vel_samples

    robot.write_root_pose_to_sim(
        torch.cat([robot_positions, robot_orientations], dim=-1),
        env_ids=env_ids,
    )
    robot.write_root_velocity_to_sim(robot_velocities, env_ids=env_ids)

    # object pose: follow only shared x, y, yaw
    rel_pos = torch.tensor(object_relative_pos, device=robot.device, dtype=torch.float32)
    rel_pos = rel_pos.unsqueeze(0).repeat(len(env_ids), 1)

    shared_yaw = rpy_delta[:, 2] + object_relative_yaw
    yaw_quat = math_utils.quat_from_euler_xyz(
        torch.zeros_like(shared_yaw),
        torch.zeros_like(shared_yaw),
        shared_yaw,
    )

    # rotate only the xy part of the relative offset
    rel_xy = rel_pos[:, :2]
    rel_xy_3d = torch.cat(
        [rel_xy, torch.zeros((len(env_ids), 1), device=robot.device)], dim=-1
    )
    rel_xy_world = math_utils.quat_apply(yaw_quat, rel_xy_3d)[:, :2]

    # start from object's own default root position
    obj_positions = obj_root_states[:, 0:3].clone() + env.scene.env_origins[env_ids]

    # shared translation only in x,y
    obj_positions[:, 0] += pos_delta[:, 0]
    obj_positions[:, 1] += pos_delta[:, 1]

    # then add rotated relative xy offset
    obj_positions[:, 0] += rel_xy_world[:, 0]
    obj_positions[:, 1] += rel_xy_world[:, 1]

    # z should NOT follow robot z
    # keep object's default z and only add its own relative z
    obj_positions[:, 2] += rel_pos[:, 2]

    # object orientation: default orientation followed only by shared yaw
    obj_default_orientation = obj_root_states[:, 3:7]
    obj_orientations = math_utils.quat_mul(obj_default_orientation, yaw_quat)

    obj.write_root_pose_to_sim(
        torch.cat([obj_positions, obj_orientations], dim=-1),
        env_ids=env_ids,
    )

    # object is kinematic/static support object
    obj_velocities = torch.zeros((len(env_ids), 6), device=robot.device)
    obj.write_root_velocity_to_sim(obj_velocities, env_ids=env_ids)


class reset_joint_state_from_buffer(ManagerTermBase):
    """Reset robot joint positions by sampling from a precomputed joint-position
    buffer saved as a .pt file.

    Expected .pt format:
        {
            "joint_pos": Tensor[num_samples, num_logged_joints],
            "joint_names": List[str],
            "wheel_joint_names": List[str],  # optional
        }

    This reset term only modifies joint state.
    Root pose and root velocity are left unchanged.
    """

    REQUIRED_KEYS = ("joint_pos", "joint_names")

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params.get(
            "asset_cfg", SceneEntityCfg("robot")
        )
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self._device = self.asset.device

        self._loaded = False

        self._buffer_joint_pos: torch.Tensor | None = None
        self._buffer_joint_names: list[str] | None = None
        self._buffer_cap: int = 0

        # Mapping from logged joint order to IsaacLab asset joint order
        self._src_indices: torch.Tensor | None = None
        self._dst_indices: torch.Tensor | None = None

    def _load_buffer(self, path: str) -> None:
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Joint position buffer file not found: {path}")

        payload = torch.load(path, map_location=self._device)

        for key in self.REQUIRED_KEYS:
            if key not in payload:
                raise KeyError(f"{path} missing required key '{key}'")

        joint_pos = payload["joint_pos"]
        joint_names = payload["joint_names"]

        if not isinstance(joint_pos, torch.Tensor):
            raise TypeError(
                f"'joint_pos' must be torch.Tensor, got {type(joint_pos)}"
            )

        if joint_pos.ndim != 2:
            raise ValueError(
                f"'joint_pos' must have shape [num_samples, num_joints], "
                f"got {tuple(joint_pos.shape)}"
            )

        if len(joint_names) != joint_pos.shape[1]:
            raise ValueError(
                f"len(joint_names)={len(joint_names)} does not match "
                f"joint_pos.shape[1]={joint_pos.shape[1]}"
            )

        self._buffer_joint_pos = joint_pos.to(
            device=self._device,
            dtype=torch.float32,
        )
        self._buffer_joint_names = list(joint_names)
        self._buffer_cap = self._buffer_joint_pos.shape[0]

        if self._buffer_cap <= 0:
            raise RuntimeError("Joint position buffer is empty.")

        self._build_joint_mapping()

        print(
            f"[reset_joint_state_from_pt_buffer] Loaded {path}\n"
            f"  buffer samples: {self._buffer_cap}\n"
            f"  buffer joints : {len(self._buffer_joint_names)}\n"
            f"  robot joints  : {self.asset.num_joints}\n"
            f"  matched joints: {len(self._src_indices)}"
        )

    def _build_joint_mapping(self) -> None:
        """Build index mapping from buffer joint names to IsaacLab asset joint names."""

        assert self._buffer_joint_names is not None

        # IsaacLab articulation joint names
        robot_joint_names = list(self.asset.joint_names)
        robot_name_to_idx = {name: i for i, name in enumerate(robot_joint_names)}

        src_indices = []
        dst_indices = []
        missing_in_robot = []

        for src_idx, joint_name in enumerate(self._buffer_joint_names):
            if joint_name in robot_name_to_idx:
                dst_idx = robot_name_to_idx[joint_name]
                src_indices.append(src_idx)
                dst_indices.append(dst_idx)
            else:
                missing_in_robot.append(joint_name)

        if len(src_indices) == 0:
            raise RuntimeError(
                "No joint names from the .pt buffer match the robot joint names.\n"
                f"Buffer joint names: {self._buffer_joint_names}\n"
                f"Robot joint names : {robot_joint_names}"
            )

        if len(missing_in_robot) > 0:
            print(
                "[reset_joint_state_from_pt_buffer] Warning: "
                "some buffer joints were not found in robot asset:\n"
                f"  {missing_in_robot}"
            )

        self._src_indices = torch.tensor(
            src_indices, dtype=torch.long, device=self._device
        )
        self._dst_indices = torch.tensor(
            dst_indices, dtype=torch.long, device=self._device
        )

    def __call__(self,
                 env,
                 env_ids: torch.Tensor,
                 dataset_path: str,
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                 use_default_for_unmatched_joints: bool = True,):
        """Apply sampled joint positions to selected envs."""

        if not self._loaded:
            self._load_buffer(dataset_path)
            self._loaded = True

        assert self._buffer_joint_pos is not None
        assert self._src_indices is not None
        assert self._dst_indices is not None

        device = self._device
        n = int(env_ids.numel())
        J = self.asset.num_joints

        # Randomly sample rows from the joint position buffer
        row_ids = torch.randint(low=0, high=self._buffer_cap, size=(n,), device=device,)

        sampled_logged_joint_pos = self._buffer_joint_pos[row_ids]

        # Construct full robot joint position tensor
        if use_default_for_unmatched_joints:
            joint_pos = self.asset.data.default_joint_pos[env_ids].clone()
        else:
            joint_pos = self.asset.data.joint_pos[env_ids].clone()

        # Fill only matched joints
        joint_pos[:, self._dst_indices] = sampled_logged_joint_pos[:, self._src_indices]
        joint_pos[:, :2] += math_utils.sample_uniform(-0.05, 0.05, (n, 2), device=device)

        # Joint velocity is always zero because the buffer only contains joint positions
        joint_vel = torch.zeros((n, J), dtype=torch.float32, device=device)

        self.asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


class randomize_joint_parameters(ManagerTermBase):
    """Randomize the simulated joint parameters of an articulation by adding, scaling, or setting random values."""

    def __init__(self, cfg: EventTermCfg, env: Env):
        """Initialize the term."""
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "friction_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["friction_distribution_params"], "friction_distribution_params")
            if "armature_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["armature_distribution_params"], "armature_distribution_params")
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_fixed_tendon_parameters' does not support operation:"
                f" '{cfg.params['operation']}'.")

    def __call__(
        self,
        env: Env,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        friction_distribution_params: tuple[float, float] | None = None,
        coulomb_distribution_params: tuple[float, float] | None = None,
        viscous_distribution_params: tuple[float, float] | None = None,
        armature_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        # resolve joint indices
        if self.asset_cfg.joint_ids == slice(None):
            joint_ids = slice(None)  # for optimization purposes
        else:
            joint_ids = torch.tensor(self.asset_cfg.joint_ids, dtype=torch.int, device=self.asset.device)

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids_for_slice = env_ids[:, None]
        else:
            env_ids_for_slice = env_ids

        # sample joint properties from the given ranges and set into the physics simulation
        # joint friction coefficient
        if friction_distribution_params is not None:
            friction_coeff = _randomize_prop_by_op(
                self.asset.data.default_joint_friction_coeff.clone(),
                friction_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )

            # ensure the friction coefficient is non-negative
            friction_coeff = torch.clamp(friction_coeff, min=0.0)

            # Always set static friction (indexed once)
            static_friction_coeff = friction_coeff[env_ids_for_slice, joint_ids]

            # if isaacsim version is lower than 5.0.0 we can set only the static friction coefficient
            if get_isaac_sim_version().major >= 5:
                # Randomize raw tensors
                dynamic_friction_coeff = _randomize_prop_by_op(
                    self.asset.data.default_joint_dynamic_friction_coeff.clone(),
                    coulomb_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
                viscous_friction_coeff = _randomize_prop_by_op(
                    self.asset.data.default_joint_viscous_friction_coeff.clone(),
                    viscous_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )

                # Clamp to non-negative
                dynamic_friction_coeff = torch.clamp(dynamic_friction_coeff, min=0.0)
                viscous_friction_coeff = torch.clamp(viscous_friction_coeff, min=0.0)

                # Ensure dynamic ≤ static (same shape before indexing)
                dynamic_friction_coeff = torch.minimum(dynamic_friction_coeff, friction_coeff)

                # Index once at the end
                dynamic_friction_coeff = dynamic_friction_coeff[env_ids_for_slice, joint_ids]
                viscous_friction_coeff = viscous_friction_coeff[env_ids_for_slice, joint_ids]
            else:
                # For versions < 5.0.0, we do not set these values
                dynamic_friction_coeff = None
                viscous_friction_coeff = None

            # Single write call for all versions
            self.asset.write_joint_friction_coefficient_to_sim(
                joint_friction_coeff=static_friction_coeff,
                joint_dynamic_friction_coeff=dynamic_friction_coeff,
                joint_viscous_friction_coeff=viscous_friction_coeff,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )

        # joint armature
        if armature_distribution_params is not None:
            armature = _randomize_prop_by_op(
                self.asset.data.default_joint_armature.clone(),
                armature_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.write_joint_armature_to_sim(
                armature[env_ids_for_slice, joint_ids], joint_ids=joint_ids, env_ids=env_ids
            )