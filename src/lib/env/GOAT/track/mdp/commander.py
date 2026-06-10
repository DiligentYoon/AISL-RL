"""Sub-module containing command generators for the joint tracking task."""

from __future__ import annotations
from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass
from isaaclab.assets import Articulation

from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

# =======================================================================
# Uniform Position Command for Holonomic and Non-Holonomic Dynamics Model
# =======================================================================
@configclass
class UniformJointPositionCommandCfg():
    """Configuration for the uniform position command generator."""

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""

    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    """Sampling time of which the commands are generated."""

    num_envs: int = 1
    """The number of parallel environments"""

    step_dt: float = None
    """Simulation step dt considering decimation ratio for calculating resampling time"""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        pos_x: tuple[float, float] = (-1.0, 1.0)
        """Range for the x position command (in m)."""

        pos_y: tuple[float, float] = (-1.0, 1.0)
        """Range for the y position command (in m)."""

        pos_z: tuple[float, float] = (-1.0, 1.0)
        """Range for the z position command (in m)."""

    ranges: Ranges = Ranges()
    """Distribution ranges for the position commands."""


class UniformJointPositionCommand():
    def __init__(self, 
                 cfg: UniformJointPositionCommandCfg, 
                 robot: Articulation, 
                 device: torch.device | str):
        self.cfg = cfg
        self.robot = robot
        self.device = device

        self.num_envs = self.cfg.num_envs
        self.step_dt = self.cfg.step_dt

        # buffers
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.pos_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)

        self.time_left = torch.zeros(self.num_envs, device=self.device)
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # initial sample for all envs
        self.reset()

    @property
    def command_b(self) -> torch.Tensor:
        """(num_envs, 3): [x, v, z] in base frame."""
        vel_w_3d = torch.cat([self.pos_command_w[:, :2], torch.zeros((self.num_envs, 1), device=self.device)], dim=-1)
        self.pos_command_b[:, :2] = quat_apply_inverse(self.robot.data.root_quat_w[:], vel_w_3d)[:, :2]
        self.pos_command_b[:, 2] = self.pos_command_w[:, 2].clone()
        return self.pos_command_b
    
    @property
    def command_w(self) -> torch.Tensor:
        """(num_envs, 3): [x, y, z] in world frame."""
        return self.pos_command_w

    @property
    def heading(self) -> torch.Tensor:
        """(num_envs, 1): yaw angle"""
        return torch.atan2(self.command_w[:, 1], self.command_w[:, 0]).unsqueeze(-1)
    
    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> dict[str, float]:
        """Call this when those envs are reset."""
        env_ids = self._resolve_env_ids(env_ids)

        # reset counters
        self.command_counter[env_ids] = 0

        # resample immediately
        self._resample(env_ids)
        return {}

    def update(self):
        """Call this once per env-step.
        Position commands are sampled only in reset phase."""
        pass

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

        r = torch.empty(env_ids.numel(), device=self.device)
        # duration
        self.time_left[env_ids] = r.uniform_(*self.cfg.resampling_time_range)

        # target pos in body frame
        self.pos_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pos_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pos_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)

        # target pos in world frame
        pos_offset_w = quat_apply(self.robot.data.root_quat_w[env_ids], self.pos_command_b[env_ids])
        self.pos_command_w[env_ids] = self.robot.data.root_pos_w[env_ids] + pos_offset_w
        self.pos_command_w[env_ids, 2] = self.pos_command_b[env_ids, 2]

        # count
        self.command_counter[env_ids] += 1