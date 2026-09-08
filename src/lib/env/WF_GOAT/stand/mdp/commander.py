"""Sub-module containing command generators for the GOAT tracking tasks."""

from __future__ import annotations
from collections.abc import Sequence

import torch

from isaaclab.utils import configclass
from isaaclab.assets import Articulation

from lib.domain_randomizer.commander import UniformNonHolonomicCommand, UniformVelocityCommandCfg
from lib.env.WF_GOAT.track.mdp.utils import GOATLegFK


# =======================================================================
# Uniform Target Joint Position Command (Left-sampled, Right mirrored)
# =======================================================================
@configclass
class UniformJointPositionCommandCfg():
    """Configuration for the uniform target joint position command generator.

    The command samples an absolute target joint position for the three joints of the
    *left* leg (hip, thigh, knee) from the robot's soft joint position limits. The right
    leg receives the sign-flipped target (``q_target_R = -q_target_L``), which is
    consistent with the sign-symmetric default pose of the GOAT robot.
    """

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""

    resampling_time_range: tuple[float, float] = (1.0, 3.0)
    """Time range (in s) over which a command stays fixed before being resampled."""

    num_envs: int = 1
    """The number of parallel environments."""

    step_dt: float = None
    """Simulation step dt considering decimation ratio for calculating resampling time."""

    prob_default_envs: float = 0.0
    """Probability that an environment is commanded to hold the default pose (zero deviation).
    Defaults to 0.0 (disabled)."""

    decay_factor: float = 1.0
    """Decaying Factor for Stable Joint Tracking.
    Defaults to 1.0 (disabled)."""


class UniformJointPositionCommand():
    """Command generator that samples absolute target joint positions per leg joint.

    The left leg (hip, thigh, knee) is the sampling base; the right leg mirrors it with a
    sign flip. The command is consumed as a tracking reference (observation/reward), it does
    not directly drive the low-level controller.
    """

    def __init__(self,
                 cfg: UniformJointPositionCommandCfg,
                 robot: Articulation,
                 device: torch.device | str):
        self.cfg = cfg
        self.robot = robot
        self.device = device

        self.num_envs = self.cfg.num_envs
        self.step_dt = self.cfg.step_dt
        self.num_joints = self.robot.num_joints
        self.fk = GOATLegFK(self.device)

        # Resolve left/right leg joint indices explicitly (do not rely on even/odd ordering).
        # preserve_order=True guarantees [hip, thigh, knee] ordering within each side.
        left_ids, _ = self.robot.find_joints(["hip_L.*", "thigh_L.*", "knee_L.*"], preserve_order=True)
        right_ids, _ = self.robot.find_joints(["hip_R.*", "thigh_R.*", "knee_R.*"], preserve_order=True)
        self.left_ids = torch.as_tensor(left_ids, device=self.device, dtype=torch.long)
        self.right_ids = torch.as_tensor(right_ids, device=self.device, dtype=torch.long)

        # Joint Specific ID
        self.hip_ids, _ = self.robot.find_joints(["hip_L.*"])

        # buffers
        # independent sample [hip, thigh, knee] of the left leg (observation-friendly view)
        self.joint_command = torch.zeros(self.num_envs, 3, device=self.device)
        # full-dim absolute target aligned with the articulation joint order (reward view)
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        # derived from joint_command via FK
        self.wheel_pos_target = torch.zeros(self.num_envs, 2, 3, device=self.device)

        self.time_left = torch.zeros(self.num_envs, device=self.device)
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # initial sample for all envs
        self.reset()

    @property
    def target(self) -> torch.Tensor:
        """(num_envs, num_joints): full-dim absolute target joint position (left = +s, right = -s)."""
        return self.joint_pos_target

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> dict[str, float]:
        """Call this when those envs are reset."""
        env_ids = self._resolve_env_ids(env_ids)

        # reset counters
        self.command_counter[env_ids] = 0

        # resample immediately
        self._resample(env_ids)
        return {}

    def update(self):
        """Call this once per env-step. Resamples commands whose duration has elapsed."""
        self.time_left -= self.cfg.step_dt
        resample_ids = (self.time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if resample_ids.numel() > 0:
            self._resample(resample_ids)

    # --------------------
    # internals
    # --------------------
    def _resolve_env_ids(self, env_ids):
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        # Sequence[int]
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _resample(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        n = env_ids.numel()

        # duration
        self.time_left[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.resampling_time_range)

        # soft joint position limits for the selected envs: (n, num_joints, 2)
        limits = self.robot.data.soft_joint_pos_limits[env_ids] * self.cfg.decay_factor
        # NOTE: Hip joint is allowed to only rotate toward outside directions
        # limits[:, self.hip_ids, 0] = 0
        limits[:, self.hip_ids, 1] = 0
        left_lo = limits[:, self.left_ids, 0]    # (n, 3)
        left_hi = limits[:, self.left_ids, 1]    # (n, 3)
        right_lo = limits[:, self.right_ids, 0]  # (n, 3)
        right_hi = limits[:, self.right_ids, 1]  # (n, 3)

        # sample absolute target for the left leg, uniform within its soft limits
        s = left_lo + (left_hi - left_lo) * torch.rand(n, 3, device=self.device)  # (n, 3)
        neg_s = torch.clamp(-s, min=right_lo, max=right_hi)

        # write into buffers via broadcasted advanced indexing (in-place, copy-safe)
        rows = env_ids.unsqueeze(-1)  # (n, 1)
        self.joint_command[env_ids] = s
        self.joint_pos_target[env_ids] = 0.0
        self.joint_pos_target[rows, self.left_ids] = s
        self.joint_pos_target[rows, self.right_ids] = neg_s

        # optionally override a subset of envs to hold the default pose (zero deviation)
        if self.cfg.prob_default_envs > 0.0:
            is_default = torch.rand(n, device=self.device) <= self.cfg.prob_default_envs
            default_ids = env_ids[is_default]
            if default_ids.numel() > 0:
                default_q = self.robot.data.default_joint_pos[default_ids] * 0.0 # NOTE : Test for all zero target
                self.joint_pos_target[default_ids] = default_q
                self.joint_command[default_ids] = default_q[:, self.left_ids]

        # compute target wheel positions using forward kinematics
        p_left_b, p_right_b = self.fk.both_wheel_centers_b(self.joint_command[env_ids])
        self.wheel_pos_target[env_ids, 0] = p_left_b
        self.wheel_pos_target[env_ids, 1] = p_right_b

        # count
        self.command_counter[env_ids] += 1


# ===================================================================
# Uniform Velocity + Base Height Command for Non-Holonomic Dynamics Model
# ===================================================================
@configclass
class UniformVelocityHeightCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the velocity command generator extended with a base-height channel."""

    height_resampling_time_range: tuple[float, float] = (2.5, 4.0)
    """Sampling time of which the height command is generated.

    Kept separate from :attr:`resampling_time_range` so that the height and the velocity
    commands never change at the same instant. Otherwise the policy can learn to associate
    a height change with a simultaneous velocity change.
    """

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        """Uniform distribution ranges for the velocity and height commands."""

        height: tuple[float, float] = (0.40, 0.56)
        """Range for the base height command (in m)."""

    ranges: Ranges = Ranges()
    """Distribution ranges for the velocity and height commands."""


class UniformVelocityHeightCommand(UniformNonHolonomicCommand):
    """Command generator that extends the non-holonomic velocity command with a base height.

    The height channel runs on its own resampling clock, so both commands stay decorrelated.
    """

    def __init__(self,
                 cfg: UniformVelocityHeightCommandCfg,
                 robot: Articulation,
                 device: torch.device | str):
        self.height_command_b = torch.zeros(cfg.num_envs, 1, device=device)
        self.height_time_left = torch.zeros(cfg.num_envs, device=device)
        self.vel_height_command_b = torch.zeros(cfg.num_envs, 4, device=device)

        super().__init__(cfg, robot, device)

    @property
    def command_b(self) -> torch.Tensor:
        """(num_envs, 4): [vx, vy, yaw_rate, height] in base frame (ground truth)."""
        self.vel_height_command_b[:, :3] = self.vel_command_b
        self.vel_height_command_b[:, 3:] = self.height_command_b
        return self.vel_height_command_b

    @property
    def height_command(self) -> torch.Tensor:
        """(num_envs, 1): absolute base height target (in m)."""
        return self.height_command_b

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> dict[str, float]:
        """Call this when those envs are reset."""
        metrics = super().reset(env_ids)
        self._resample_height(self._resolve_env_ids(env_ids))
        return metrics

    def update(self):
        """Call this once per env-step."""
        # the height clock is stepped separately from the velocity clock of the base class
        self.height_time_left -= self.step_dt
        resample_ids = (self.height_time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if resample_ids.numel() > 0:
            self._resample_height(resample_ids)

        super().update()

    # --------------------
    # internals
    # --------------------
    def _resample_height(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        r = torch.empty(env_ids.numel(), device=self.device)

        # duration
        self.height_time_left[env_ids] = r.uniform_(*self.cfg.height_resampling_time_range)
        # absolute base height target
        self.height_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)
