from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply

from lib.env.WF_GOAT.stand.WF_GOAT_stand_env import WFGOATStandEnv
from lib.env.WF_GOAT.track.WF_GOAT_track_env_cfg import WFGOATTrackEnvCfg, WFGOATTrackPlayEnvCfg

class WFGOATTrackEnv(WFGOATStandEnv):
    cfg: WFGOATTrackEnvCfg | WFGOATTrackPlayEnvCfg

    def __init__(self, cfg: WFGOATTrackEnvCfg | WFGOATTrackPlayEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Joint position bias
        self.joint_pos_bias = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.biased_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)

    def _apply_action(self):
        # Current state (biased position)
        cmd_joint_pos = self._robot.data.default_joint_pos[:, self.joint_ids] + self.processed_actions[:, self.joint_ids] - self.biased_joint_pos[:, self.joint_ids]
        cmd_wheel_vel = self.processed_actions[:, self.wheel_ids]
        
        # Apply command
        self._robot.set_joint_position_target(cmd_joint_pos, joint_ids=self.joint_ids)
        self._robot.set_joint_velocity_target(cmd_wheel_vel, joint_ids=self.wheel_ids)

    def _get_observations(self) -> torch.Tensor:
        """
        Get sensor data without curriculum Gaussian noise

        Returns:
            Observation space
        """
        observation = torch.cat((self.base_ang_vel,                                                                     # [E, 3]
                                 self.gravity_vector,                                                                   # [E, 3]
                                 self.command_inputs_b,                                                                 # [E, 4]
                                 self.biased_joint_pos[:, self.joint_ids] - self.default_joint_pos[:, self.joint_ids],  # [E, 4]
                                 self.joint_vel,                                                                        # [E, 6]
                                 self.previous_actions,                                                                 # [E, 6]
                                ), dim=1) 

        return observation
    
    def _get_states(self) -> torch.Tensor:
        """"
        Get State space using previleged information

        Returns
            State space
        """
        observation = torch.cat((self.base_ang_vel,                                                               # [E, 3]
                                 self.gravity_vector,                                                             # [E, 3]
                                 self.command_inputs_b,                                                           # [E, 4]
                                 self.joint_pos[:, self.joint_ids] - self.default_joint_pos[:, self.joint_ids],   # [E, 4]
                                 self.joint_vel,                                                                  # [E, 6]
                                 self.previous_actions,                                                           # [E, 6]
                                 ), dim=1)                             
        
        privileged_info = torch.cat((self.base_lin_vel,                                      # [E, 3]
                                     self.base_height,                                       # [E, 1]
                                     self.joint_pos_bias[:, self.joint_ids],                 # [E, 4]
                                     self.friction_coefficient), dim=1)                      # [E, 2]
        
        state = torch.cat([observation, privileged_info], dim=-1)

        return state

    def _get_rewards(self) -> torch.Tensor:
        # Orientation Reward (Projected Gravity Alignment)
        upright_error = torch.sum(torch.square(self.gravity_vector[:, :2]), dim=1)
        r_upright = torch.exp(-upright_error / 0.05)

        # Height tracking Reward
        height_error = torch.reshape(torch.abs(self.base_height - self.command_inputs_b[:, 3:]), (-1,))
        r_height = torch.exp(-height_error / 0.05)

        # Lin vel Tracking Reward
        lin_vel_error = torch.sum(torch.square(self.base_lin_vel[:, :2] - self.command_inputs_b[:, :2]), dim=1)
        r_lin_vel_tracking = torch.exp(-lin_vel_error / 0.05)

        # Ang vel Tracking Reward
        ang_vel_error = torch.abs(self.base_ang_vel[:, 2] - self.command_inputs_b[:, 2])
        r_ang_vel_tracking = torch.exp(-ang_vel_error / 0.05)

        # COM align Reward
        wheel_xy = torch.mean(self.wheel_pos[:, :, :2], dim=1)
        align_error = torch.sum(torch.square(self.base_pos_w[:, :2] - wheel_xy), dim=1)
        r_com_align = torch.exp(-align_error / 0.02)

        # Regularization Penalty
        p_ang_vel            = -torch.norm(self.base_ang_vel[:, :2], dim=-1)        
        p_illegal_contact    = -torch.sum(self.illegal_force, dim=1)
        
        p_velocity_limit     = -torch.sum(self.out_of_limits_velocity[:, self.joint_ids], dim=1)     # wheel is not included
        p_all_torque_limit   = -torch.sum(self.out_of_limits_torque, dim=1)
        p_all_torque         = -torch.sum(torch.square(self.applied_torque), dim=1)
        p_joint_velocity     = -torch.sum(torch.square(self.joint_vel[:, self.joint_ids]), dim=1)    # wheel is not included
        p_joint_accel        = -torch.sum(torch.square(self.joint_acc), dim=1)                       # [NOTE] wheel is included
        p_joint_deviation_lr = -torch.sum(torch.abs(self.joint_deviation_lr), dim=-1)
        p_action_rate        = -torch.sum(torch.abs((self.actions - self.previous_actions)), dim=1)
        p_terminated         = -self.reset_terminated.float()

        # Total Reward Summation
        total_reward = (
            self.cfg.r_upright_weight * r_upright                           +
            self.cfg.r_height_weight * r_height                             +
            self.cfg.r_lin_vel_tracking_weight * r_lin_vel_tracking         +
            self.cfg.r_ang_vel_tracking_weight * r_ang_vel_tracking         +
            self.cfg.r_com_align_weight * r_com_align                       +
            self.cfg.p_ang_vel_weight * p_ang_vel                           +
            self.cfg.p_illegal_contact_weight * p_illegal_contact           + 
            self.cfg.p_all_torque_limit_weight * p_all_torque_limit         +
            self.cfg.p_all_torque_weight * p_all_torque                     +
            self.cfg.p_joint_vel_limit_weight * p_velocity_limit            +
            self.cfg.p_joint_velocity_weight * p_joint_velocity             +
            self.cfg.p_joint_accel_weight * p_joint_accel                   +
            self.cfg.p_joint_deviation_lr_weight * p_joint_deviation_lr     +
            self.cfg.p_action_rate_weight * p_action_rate                   +
            self.cfg.p_terminated_weight * p_terminated
        )

        self.extras["reward"] = {
            # ==========================================
            # Task Reward (+)
            # ==========================================
            "Task Reward / Upright"             : r_upright,
            "Task Reward / Height"              : r_height,
            "Task Reward / Lin_Vel_Tracking"    : r_lin_vel_tracking,
            "Task Reward / Ang_Vel_Tracking"    : r_ang_vel_tracking,
            "Task Rweard / COM_Align"           : r_com_align, 
            # ==========================================
            # Task Penalty (-)
            # ==========================================
            "Task Penalty / Ang_Vel"            : p_ang_vel,
            "Task Penalty / Contact"            : p_illegal_contact,
            "Task Penalty / Torque_Limit"       : p_all_torque_limit,
            "Task Penalty / Torque"             : p_all_torque,
            "Task Penalty / Vel_Limit"          : p_velocity_limit, 
            "Task Penalty / Joint_Vel"          : p_joint_velocity,
            "Task Penalty / Joint_Acc"          : p_joint_accel,
            "Task Penalty / Joint_Deviation_LR" : p_joint_deviation_lr,
            "Task Penalty / Action_Rate"        : p_action_rate,
        }

        self.previous_actions = self.actions.clone()

        return total_reward

    def _compute_intermediate_values(self, env_ids = None):
        super()._compute_intermediate_values(env_ids)
        i = env_ids if env_ids is not None else self._robot._ALL_INDICES

        self.biased_joint_pos[i] = self.joint_pos[i] + self.joint_pos_bias[i]

        