
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.types import ArticulationActions
from isaaclab.actuators.actuator_pd import IdealPDActuator

if TYPE_CHECKING:
    from .torque_delay_actuator_cfg import (
        TorqueDelayedPDActuatorCfg
    )


class TorqueDelayedPDActuator(IdealPDActuator):
    """Ideal PD actuator with delayed control input application."""

    cfg: TorqueDelayedPDActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: TorqueDelayedPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        if self.cfg.max_delay >= self.cfg.decimation:
            raise ValueError(f"Max delay must be less than decimation.")
        # control loop <-> physics step decimation
        self.decimation = self.cfg.decimation
        # instantiate the delay buffers
        self.progress = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.time_lags = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        # torque buffer
        self.previous_applied_torque = torch.zeros((self._num_envs, self.num_joints), device=self._device, dtype=torch.float32)
        self.current_applied_torque = torch.zeros((self._num_envs, self.num_joints), device=self._device, dtype=torch.float32)
        self.previous_computed_torque = torch.zeros((self._num_envs, self.num_joints), device=self._device, dtype=torch.float32)
        self.current_computed_torque = torch.zeros((self._num_envs, self.num_joints), device=self._device, dtype=torch.float32)

        # all of the envs
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            env_ids = self._ALL_INDICES
        else:
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self._device)
        # reset progress
        self.progress[env_ids] = 0
        # reset torque buffer
        self.previous_applied_torque[env_ids] = 0.0
        self.current_applied_torque[env_ids] = 0.0
        self.previous_computed_torque[env_ids] = 0.0
        self.current_computed_torque[env_ids] = 0.0

    def resample_time_lag(self, env_ids: torch.Tensor):

        # set a new random delay for environments in env_ids
        self.time_lags[env_ids] = torch.randint(low=self.cfg.min_delay,
                                                high=self.cfg.max_delay + 1,
                                                size=(env_ids.numel(), ),
                                                dtype=torch.long,
                                                device=self._device,
                                            )

    def compute(self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> ArticulationActions:
        phase = self.progress % self.decimation

        update_mask = phase == 0
        update_ids = torch.nonzero(update_mask, as_tuple=False).squeeze(-1)
        if len(update_ids):
            # update previous torque
            self.previous_computed_torque[update_ids] = self.current_computed_torque[update_ids].clone()
            self.previous_applied_torque[update_ids] = self.current_applied_torque[update_ids].clone()
            # compute current step torque
            error_pos = control_action.joint_positions[update_ids] - joint_pos[update_ids]
            error_vel = control_action.joint_velocities[update_ids] - joint_vel[update_ids]
            computed_torque = self.stiffness[update_ids] * error_pos + self.damping[update_ids] * error_vel + control_action.joint_efforts[update_ids]
            applied_torque = self._clip_effort(computed_torque)
            # update current torque
            self.current_computed_torque[update_ids] = computed_torque
            self.current_applied_torque[update_ids] = applied_torque
            # randomized delay at each torque calculation event
            self.resample_time_lag(update_ids)

        use_current = (phase >= self.time_lags).unsqueeze(-1)

        final_computed_torque = torch.where(use_current, self.current_computed_torque, self.previous_computed_torque) 
        final_applied_torque = torch.where(use_current, self.current_applied_torque, self.previous_applied_torque)
        
        # Update torque buffer
        control_action.joint_efforts = final_applied_torque
        control_action.joint_positions = None
        control_action.joint_velocities = None

        self.computed_effort = final_computed_torque
        self.applied_effort  = final_applied_torque

        # Update progress
        self.progress += 1

        return control_action