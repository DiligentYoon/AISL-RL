from __future__ import annotations

import os
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg



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
        """Apply sampled joint positions to selected envs.

        Args:
            env:
                IsaacLab environment.
            env_ids:
                Environment indices to reset.
            dataset_path:
                Path to .pt file containing joint_pos and joint_names.
            asset_cfg:
                Target robot asset config.
            use_default_for_unmatched_joints:
                If True, unmatched robot joints are set to default_joint_pos.
                If False, unmatched robot joints keep their current position.
        """

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

        # Joint velocity is always zero because the buffer only contains joint positions
        joint_vel = torch.zeros((n, J), dtype=torch.float32, device=device)

        self.asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)