from __future__ import annotations

import os
import torch

import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg


def reset_joints_by_offset_and_bias(
    env,
    env_ids: torch.Tensor,
    bias: tuple[float, float, float],
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    eps = 1e-10

    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # get default joint state
    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()
    
    # add constant bias
    # signed_bias = torch.sign(joint_pos + eps) * torch.tensor(bias, dtype=torch.float32, device=joint_pos.device)
    joint_pos += torch.tensor(bias, dtype=torch.float32, device=joint_pos.device)

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


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


class reset_robot_state_from_buffer(ManagerTermBase):
    """Reset root pose and joint positions from the same sampled dataset row.

    Expected .pt format:
        {
            "joint_pos": Tensor[num_samples, num_logged_joints],
            "joint_names": List[str],

            "base_pos": Tensor[num_samples, 3],
            "base_quat": Tensor[num_samples, 4],

            "base_pos_order": ["x", "y", "z"],       # optional metadata
            "base_quat_order": ["w", "x", "y", "z"],  # optional metadata
        }

    Reset behavior:
        - joint position: sampled from the buffer
        - joint velocity: zero
        - root x, y: default root x, y + environment origin
        - root z: sampled base height
        - root roll: zero
        - root pitch: extracted from sampled buffer quaternion
        - root yaw: uniformly randomized
        - root linear/angular velocity: zero
    """

    REQUIRED_KEYS = (
        "joint_pos",
        "joint_names",
        "base_pos",
        "base_quat",
    )

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params.get(
            "asset_cfg",
            SceneEntityCfg("robot"),
        )
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self._device = self.asset.device

        self._loaded = False
        self._buffer_cap = 0

        # Dataset buffers
        self._buffer_joint_pos: torch.Tensor | None = None
        self._buffer_joint_names: list[str] | None = None
        self._buffer_base_z: torch.Tensor | None = None
        self._buffer_base_pitch: torch.Tensor | None = None

        # Mapping:
        # dataset joint index -> IsaacLab articulation joint index
        self._src_indices: torch.Tensor | None = None
        self._dst_indices: torch.Tensor | None = None

    def _load_buffer(self, path: str) -> None:
        path = os.path.abspath(os.path.expanduser(path))

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Initial-state buffer file not found: {path}"
            )

        payload = torch.load(
            path,
            map_location=self._device,
        )

        for key in self.REQUIRED_KEYS:
            if key not in payload:
                raise KeyError(
                    f"{path} is missing required key '{key}'."
                )

        joint_pos = payload["joint_pos"]
        joint_names = payload["joint_names"]
        base_pos = payload["base_pos"]
        base_quat = payload["base_quat"]
        num_samples = joint_pos.shape[0]

        # ------------------------------------------------------------
        # Move tensors to simulation device
        # ------------------------------------------------------------
        joint_pos = joint_pos.to(
            device=self._device,
            dtype=torch.float32,
        )
        base_pos = base_pos.to(
            device=self._device,
            dtype=torch.float32,
        )
        base_quat = base_quat.to(
            device=self._device,
            dtype=torch.float32,
        )

        # ------------------------------------------------------------
        # Reorder base fields according to stored metadata
        # ------------------------------------------------------------
        base_pos_order = list(
            payload.get(
                "base_pos_order",
                ["x", "y", "z"],
            )
        )
        base_quat_order = list(
            payload.get(
                "base_quat_order",
                ["w", "x", "y", "z"],
            )
        )

        base_pos = self._reorder_columns(
            tensor=base_pos,
            source_order=base_pos_order,
            target_order=["x", "y", "z"],
            field_name="base_pos",
        )

        base_quat = self._reorder_columns(
            tensor=base_quat,
            source_order=base_quat_order,
            target_order=["w", "x", "y", "z"],
            field_name="base_quat",
        )

        # ------------------------------------------------------------
        # Normalize quaternion and extract pitch
        # ------------------------------------------------------------
        quat_norm = torch.linalg.vector_norm(
            base_quat,
            dim=-1,
            keepdim=True,
        )

        if torch.any(quat_norm < 1.0e-8):
            invalid_count = int(
                torch.sum(quat_norm.squeeze(-1) < 1.0e-8).item()
            )
            raise ValueError(
                f"Dataset contains {invalid_count} invalid zero-norm "
                "quaternion samples."
            )

        base_quat = base_quat / quat_norm

        _, base_pitch, _ = math_utils.euler_xyz_from_quat(base_quat)

        # Only the sampled absolute height and pitch are needed.
        self._buffer_joint_pos = joint_pos
        self._buffer_joint_names = list(joint_names)
        self._buffer_base_z = base_pos[:, 2]
        self._buffer_base_pitch = base_pitch
        self._buffer_cap = num_samples

        self._build_joint_mapping()

        print(
            "[reset_robot_state_from_buffer] Buffer loaded\n"
            f"  path           : {path}\n"
            f"  samples        : {self._buffer_cap}\n"
            f"  buffer joints  : {len(self._buffer_joint_names)}\n"
            f"  robot joints   : {self.asset.num_joints}\n"
            f"  matched joints : {len(self._src_indices)}\n"
            f"  base z range   : "
            f"[{self._buffer_base_z.min().item():.6f}, "
            f"{self._buffer_base_z.max().item():.6f}] m\n"
            f"  pitch range    : "
            f"[{torch.rad2deg(self._buffer_base_pitch.min()).item():.3f}, "
            f"{torch.rad2deg(self._buffer_base_pitch.max()).item():.3f}] deg"
        )

    @staticmethod
    def _reorder_columns(
        tensor: torch.Tensor,
        source_order: list[str],
        target_order: list[str],
        field_name: str,
    ) -> torch.Tensor:
        """Reorder tensor columns using saved metadata."""

        if len(source_order) != tensor.shape[1]:
            raise ValueError(
                f"{field_name}_order has {len(source_order)} entries, "
                f"but {field_name} has {tensor.shape[1]} columns."
            )

        missing = [
            name
            for name in target_order
            if name not in source_order
        ]

        if missing:
            raise ValueError(
                f"{field_name}_order is missing fields: {missing}. "
                f"Stored order: {source_order}"
            )

        indices = [
            source_order.index(name)
            for name in target_order
        ]

        return tensor[:, indices]

    def _build_joint_mapping(self) -> None:
        """Map dataset joint columns to the articulation joint order."""

        assert self._buffer_joint_names is not None

        robot_joint_names = list(self.asset.joint_names)
        robot_name_to_idx = {
            name: index
            for index, name in enumerate(robot_joint_names)
        }

        src_indices: list[int] = []
        dst_indices: list[int] = []
        missing_in_robot: list[str] = []

        for src_idx, joint_name in enumerate(
            self._buffer_joint_names
        ):
            if joint_name in robot_name_to_idx:
                src_indices.append(src_idx)
                dst_indices.append(
                    robot_name_to_idx[joint_name]
                )
            else:
                missing_in_robot.append(joint_name)

        if not src_indices:
            raise RuntimeError(
                "No dataset joint names match the robot articulation.\n"
                f"Dataset joints: {self._buffer_joint_names}\n"
                f"Robot joints  : {robot_joint_names}"
            )

        if missing_in_robot:
            print(
                "[reset_robot_state_from_buffer] Warning: "
                "dataset joints not found in the robot:\n"
                f"  {missing_in_robot}"
            )

        self._src_indices = torch.tensor(
            src_indices,
            dtype=torch.long,
            device=self._device,
        )
        self._dst_indices = torch.tensor(
            dst_indices,
            dtype=torch.long,
            device=self._device,
        )

    def __call__(
        self,
        env,
        env_ids: torch.Tensor,
        dataset_path: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        yaw_range: tuple[float, float] = (-3.14, 3.14),
    ) -> None:
        """Apply sampled root pose and joint state to selected environments.

        Reset behavior:
            - One dataset row is sampled per environment.
            - Logged joint positions are restored from the sampled row.
            - Joints absent from the dataset, including wheels, use
            default_joint_pos.
            - Joint velocities are initialized to zero.
            - Root x/y use the default position relative to each environment.
            - Root z uses the sampled dataset height relative to env_origin_z.
            - Root roll is zero.
            - Root pitch is extracted from the sampled dataset quaternion.
            - Root yaw is uniformly randomized.
            - Root linear and angular velocities are initialized to zero.
        """

        # asset_cfg is resolved in __init__.
        _ = asset_cfg

        if not self._loaded:
            self._load_buffer(dataset_path)
            self._loaded = True

        if env_ids.numel() == 0:
            return

        assert self._buffer_joint_pos is not None
        assert self._buffer_base_z is not None
        assert self._buffer_base_pitch is not None
        assert self._src_indices is not None
        assert self._dst_indices is not None

        yaw_min = float(yaw_range[0])
        yaw_max = float(yaw_range[1])

        if yaw_max < yaw_min:
            raise ValueError(
                f"Invalid yaw_range={yaw_range}: "
                "maximum is smaller than minimum."
            )

        device = self._device
        num_reset_envs = int(env_ids.numel())
        num_robot_joints = self.asset.num_joints

        # ============================================================
        # 1. Sample one common dataset row for root and joint state
        # ============================================================
        row_ids = torch.randint(
            low=0,
            high=self._buffer_cap,
            size=(num_reset_envs,),
            device=device,
        )

        sampled_joint_pos = self._buffer_joint_pos[row_ids]
        sampled_base_z = self._buffer_base_z[row_ids]
        sampled_base_pitch = self._buffer_base_pitch[row_ids]

        # ============================================================
        # 2. Construct full joint state
        # ============================================================
        # Joints absent from the dataset, such as wheels, always use
        # default_joint_pos.
        joint_pos = self.asset.data.default_joint_pos[env_ids].clone()

        joint_pos[:, self._dst_indices] = (
            sampled_joint_pos[:, self._src_indices]
        )

        joint_vel = torch.zeros(
            (num_reset_envs, num_robot_joints),
            dtype=torch.float32,
            device=device,
        )

        # ============================================================
        # 3. Construct root position
        # ============================================================
        default_root_state = (
            self.asset.data.default_root_state[env_ids].clone()
        )
        env_origins = env.scene.env_origins[env_ids]

        # Default x/y position relative to each vectorized environment.
        root_pos = default_root_state[:, 0:3] + env_origins

        # Dataset z is always interpreted as height relative to the
        # environment origin.
        root_pos[:, 2] = env_origins[:, 2] + sampled_base_z

        # ============================================================
        # 4. Construct root orientation
        # ============================================================
        root_roll = torch.zeros_like(sampled_base_pitch)

        random_yaw = torch.empty(
            num_reset_envs,
            dtype=torch.float32,
            device=device,
        ).uniform_(yaw_min, yaw_max)

        # IsaacLab quaternion order: [w, x, y, z]
        root_quat = math_utils.quat_from_euler_xyz(
            root_roll,
            sampled_base_pitch,
            random_yaw,
        )

        root_pose = torch.cat(
            (root_pos, root_quat),
            dim=-1,
        )

        # [vx, vy, vz, wx, wy, wz]
        root_vel = torch.zeros(
            (num_reset_envs, 6),
            dtype=torch.float32,
            device=device,
        )

        # ============================================================
        # 5. Apply root and joint states
        # ============================================================
        self.asset.write_root_pose_to_sim(
            root_pose,
            env_ids=env_ids,
        )

        self.asset.write_root_velocity_to_sim(
            root_vel,
            env_ids=env_ids,
        )

        self.asset.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            env_ids=env_ids,
    )