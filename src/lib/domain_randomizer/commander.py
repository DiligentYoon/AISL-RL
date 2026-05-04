# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch

from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG

from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

# import logger
logger = logging.getLogger(__name__)

# ===================================================================
# Uniform Velocity Command for Holonomic Dynamics Model
# ===================================================================
@configclass
class UniformVelocityCommandCfg():
    """Configuration for the uniform velocity command generator."""

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""

    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    """Sampling time of which the commands are generated."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """

    enable_metrics: bool = True
    """Wheter to use visualization metrics"""

    num_envs: int = 1
    """The number of parallel environments"""

    step_dt: float = None
    """Simulation step dt considering decimation ratio for calculating resampling time"""

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    prob_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    prob_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command). Defaults to 1.0.

    This parameter is only used if :attr:`heading_command` is True.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = (-1.0, 1.0)
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = (-1.0, 1.0)
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.

        This parameter is only used if :attr:`~UniformVelocityCommandCfg.heading_command` is True.
        """

    ranges: Ranges = Ranges()
    """Distribution ranges for the velocity commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


class UniformVelocityCommand():
    """Command generator that generates a velocity command in SE(2) from uniform distribution."""

    def __init__(self, 
                 cfg: UniformVelocityCommandCfg, 
                 robot: Articulation, 
                 device: torch.device | str):
        self.cfg = cfg
        self.robot = robot
        self.device = device

        self.num_envs = self.cfg.num_envs
        self.step_dt = self.cfg.step_dt

        # config checks
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError("heading_command=True but cfg.ranges.heading is None")

        # buffers
        self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_command_yaw = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.time_left = torch.zeros(self.num_envs, device=self.device)
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.enable_metrics = self.cfg.enable_metrics
        self.metrics = {}
        if self.enable_metrics:
            self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

        # initial sample for all envs
        self.reset()

    @property
    def command_b(self) -> torch.Tensor:
        """(num_envs, 3): [vx, vy, yaw_rate] in base frame."""
        vel_w_3d = torch.cat([self.vel_command_w[:, :2], torch.zeros((self.num_envs, 1), device=self.device)], dim=-1)
        self.vel_command_b[:, :2] = quat_apply_inverse(self.robot.data.root_quat_w[:], vel_w_3d)[:, :2]
        self.vel_command_b[:, 2] = self.vel_command_w[:, 2].clone()
        return self.vel_command_b
    
    @property
    def command_w(self) -> torch.Tensor:
        """(num_envs, 3): [vx, vy, yaw_rate] in world frame."""
        return self.vel_command_w
    
    @property
    def command_yaw(self) -> torch.Tensor:
        """(num_envs, 3): [vx, vy, yaw_rate] in robot yaw-frame."""
        vel_w_3d = torch.cat([self.vel_command_w[:, :2], torch.zeros((self.num_envs, 1), device=self.device)], dim=-1)
        yaw_q = yaw_quat(self.robot.data.root_quat_w)
        self.vel_command_yaw[:, :2] = quat_apply_inverse(yaw_q, vel_w_3d)[:, :2]
        self.vel_command_yaw[:, 2] = self.vel_command_w[:, 2].clone()
        return self.vel_command_yaw

    @property
    def heading(self) -> torch.Tensor:
        """(num_envs, 1): yaw angle"""
        return torch.atan2(self.command_w[:, 1], self.command_w[:, 0]).unsqueeze(-1)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> dict[str, float]:
        """Call this when those envs are reset."""
        env_ids = self._resolve_env_ids(env_ids)

        # reset counters/metrics
        self.command_counter[env_ids] = 0
        if self.enable_metrics:
            for k in self.metrics:
                self.metrics[k][env_ids] = 0.0

        # resample immediately
        self._resample(env_ids)
        return {}

    def update(self):
        """Call this once per env-step."""
        if self.enable_metrics:
            self._update_metrics()

        self.time_left -= self.cfg.step_dt
        resample_ids = (self.time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if resample_ids.numel() > 0:
            self._resample(resample_ids)

        self._post_process()

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

        r = torch.empty(env_ids.numel(), device=self.device)

        # duration
        self.time_left[env_ids] = r.uniform_(*self.cfg.resampling_time_range)

        # linear vel in body frame
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # yaw rate (may be overridden by heading post-process)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # commadns in world frame
        vel_b_3d = torch.cat([self.vel_command_b[env_ids, :2], torch.zeros((len(env_ids), 1), device=self.device)], dim=-1)
        self.vel_command_w[env_ids, :2] = quat_apply(self.robot.data.root_quat_w[env_ids], vel_b_3d)[:, :2]
        self.vel_command_w[env_ids, 2] = self.vel_command_b[env_ids, 2].clone()

        # heading env selection (probabilistic)
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.prob_heading_envs
        else:
            self.is_heading_env[env_ids] = False

        # standing env selection (probabilistic)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.prob_standing_envs
        # apply zero command
        standing_ids = env_ids[self.is_standing_env[env_ids]]
        if standing_ids.numel() > 0:
            self.vel_command_b[standing_ids, :] = 0.0
            self.vel_command_w[standing_ids, :] = 0.0

        # count
        self.command_counter[env_ids] += 1

    def _post_process(self):
        # heading-based yaw command
        if self.cfg.heading_command:
            ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            if ids.numel() > 0:
                heading_error = math_utils.wrap_to_pi(self.heading_target[ids] - self.robot.data.heading_w[ids])
                self.vel_command_w[ids, 2] = torch.clip(
                    self.cfg.heading_control_stiffness * heading_error,
                    min=self.cfg.ranges.ang_vel_z[0],
                    max=self.cfg.ranges.ang_vel_z[1],)
                self.vel_command_b[ids, 2] = self.vel_command_w[ids, 2]

    def _update_metrics(self):
        # same normalization as original
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self.step_dt

        self.metrics["error_vel_xy"] += (
            torch.norm(self.command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resolve_xy_velocity_to_arrow(self, scale, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle) # body frame
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat) # arrow rot in body frame -> world frame

        return arrow_scale, arrow_quat

    def _resolve_yaw_velocity_to_arrow(self, scale, yaw_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the Z base angular velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(yaw_velocity.shape[0], 1)
        arrow_scale[:, 0] *= yaw_velocity.squeeze(-1) * 3.0
        # arrow-direction (constant pitch)
        pitch_angle = torch.pi/2 * torch.ones(yaw_velocity.shape[0], device=self.device)
        zeros = torch.zeros_like(pitch_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, pitch_angle, zeros) # world frame

        return arrow_scale, arrow_quat


# ===================================================================
# Uniform Velocity Command for Non-Holonomic Dynamics Model
# ===================================================================
class UniformNonHolonomicCommand(UniformVelocityCommand):
    def __init__(self, cfg: UniformVelocityCommandCfg, robot: Articulation, device: torch.device | str):
        super().__init__(cfg, robot, device)

        if self.enable_metrics:
            self.metrics = {}
            self.metrics["error_vel_x"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_vel_y"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command_b(self) -> torch.Tensor:
        """(num_envs, 3): [vx, vy, yaw_rate] in base frame (ground truth)."""
        return self.vel_command_b
    
    @property
    def command_w(self) -> torch.Tensor:
        """(num_envs, 3): [vx, vy, yaw_rate] in world frame (computed from body frame)."""
        vel_b_3d = torch.cat([self.vel_command_b[:, :2],
                              torch.zeros((self.num_envs, 1), device=self.device)], dim=-1)
        self.vel_command_w[:, :2] = quat_apply(self.robot.data.root_quat_w, vel_b_3d)[:, :2]
        self.vel_command_w[:, 2] = self.vel_command_b[:, 2].clone()
        return self.vel_command_w

    def _resample(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        r = torch.empty(env_ids.numel(), device=self.device)

        self.time_left[env_ids] = r.uniform_(*self.cfg.resampling_time_range)

        # non-holonomic command
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)   # vx
        self.vel_command_b[env_ids, 1] = 0.0                                      # vy = 0 (Non-holonomic constraint)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)   # wz

        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.prob_standing_envs

        # apply zero command
        standing_env_ids = env_ids[self.is_standing_env[env_ids]]
        if standing_env_ids.numel() > 0:
            self.vel_command_b[standing_env_ids, :] = 0.0
            self.vel_command_w[standing_env_ids, :] = 0.0

        self.command_counter[env_ids] += 1

    def _post_process(self):
        pass

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self.step_dt

        self.metrics["error_vel_x"] += (
            torch.abs(self.vel_command_b[:, 0] - self.robot.data.root_lin_vel_b[:, 0]) / max_command_step
        )
        self.metrics["error_vel_y"] += (
            torch.abs(self.robot.data.root_lin_vel_b[:, 1]) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )


# =======================================================================
# Uniform Position Command for Holonomic and Non-Holonomic Dynamics Model
# =======================================================================
@configclass
class UniformPositionCommandCfg():
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

    goal_pos_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal position visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_pos_visualizer_cfg.markers["frame"].scale = (0.5, 0.5, 0.5)


class UniformPositionCommand():
    def __init__(self, 
                 cfg: UniformPositionCommandCfg, 
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