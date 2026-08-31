from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply

from lib.env.WF_GOAT.stand.WF_GOAT_stand_env import WFGOATStandEnv
from lib.env.WF_GOAT.track.WF_GOAT_track_env_cfg import WFGOATTrackEnvCfg, WFGOATTrackPlayEnvCfg
from lib.env.WF_GOAT.track.mdp.commander import UniformVelocityHeightCommand

class WFGOATTrackEnv(WFGOATStandEnv):
    cfg: WFGOATTrackEnvCfg | WFGOATTrackPlayEnvCfg

    def __init__(self, cfg: WFGOATTrackEnvCfg | WFGOATTrackPlayEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Commands for reference generator
        self.commands = UniformVelocityHeightCommand(self.cfg.commands, self._robot, self.device)
        self.command_inputs_b   = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device) # [vx, vy, vz]

        # Jig release
        self.stand_up_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.jig_release = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_release = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Joint ids
        # self.hip_joint_ids, _ = self._robot.find_joints(["hip_.*"])

    def _debug_vis_callback(self, event):
        if not self._robot.is_initialized:
            return

        # ============== Arrow ================ #
        # Arrow: get marker location
        x_offset_w = quat_apply(self._robot.data.root_quat_w, 0.3 * self.forward_vec)
        goal_pos_w = self._robot.data.root_pos_w.clone() + x_offset_w
        goal_pos_w[:, 2] = self.commands.height_command.squeeze(-1)
        curr_pos_w = self._robot.data.root_pos_w.clone() + x_offset_w
        # Arrow: resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self.commands._resolve_xy_velocity_to_arrow(scale=self.goal_vel_visualizer.cfg.markers["arrow"].scale,
                                                                                              xy_velocity=self.commands.command_b[:, :2])
        vel_arrow_scale, vel_arrow_quat = self.commands._resolve_xy_velocity_to_arrow(scale=self.current_vel_visualizer.cfg.markers["arrow"].scale,
                                                                                      xy_velocity=self._robot.data.root_lin_vel_b[:, :2])

        if hasattr(self, "goal_vel_visualizer"):
            self.goal_vel_visualizer.visualize(goal_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        if hasattr(self, "current_vel_visualizer"):
            self.current_vel_visualizer.visualize(curr_pos_w, vel_arrow_quat, vel_arrow_scale)
    
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
        r_ang_vel_tracking = torch.exp(-ang_vel_error / 0.1)

        # Regularization Penalty
        p_ang_vel            = -torch.norm(self.base_ang_vel[:, :2], dim=-1)        
        # p_hip_deviation      = -torch.sum(torch.abs(self.joint_deviation[:, self.hip_joint_ids]), dim=1)
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
            self.cfg.p_ang_vel_weight * p_ang_vel                           +
            # self.cfg.p_hip_deviation_weight * p_hip_deviation               +
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
            # ==========================================
            # Task Penalty (-)
            # ==========================================
            "Task Penalty / Ang_Vel"            : p_ang_vel,
            # "Task Penalty / Hip_Deviation"      : p_hip_deviation,
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

    def _get_dones(self):
        self._compute_intermediate_values()

        base_fall = (self.base_height <= self.cfg.height_reset_condition).squeeze(-1)

        projected_gravity_x = self.gravity_vector[:, 0]
        projected_gravity_y = self.gravity_vector[:, 1]
        tilt_fall = torch.logical_or(torch.abs(projected_gravity_x) >= self.cfg.terminated_tilt,
                                     torch.abs(projected_gravity_y) >= self.cfg.terminated_tilt)
        exceed_joint_vel = torch.any(torch.abs(self.joint_vel[:, self.joint_ids]) > self.cfg.terminated_joint_vel_limit, dim=-1)
        exceed_base_vel_z = torch.abs(self.base_lin_vel[:, 2]) > self.cfg.terminated_lin_vel_limit_z
        
        terminated = base_fall | tilt_fall | exceed_joint_vel | exceed_base_vel_z
        truncated = self.episode_length_buf >= (self.cfg.max_episode_length - 1)

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        # Reset previous action observation
        self.jig_release[env_ids] = False
        self.is_release[env_ids] = False
        self.stand_up_counter[env_ids] = 0
        super()._reset_idx(env_ids)

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        super()._compute_intermediate_values(env_ids)

        # The jig is only handled on the full-env path
        if env_ids is not None:
            return

        # Latch the release once the robot has held itself above the support height
        self.stand_up_counter = torch.where(self.base_height.squeeze(-1) > self.cfg.jig_release_height,
                                            self.stand_up_counter + 1,
                                            torch.zeros_like(self.stand_up_counter))
        self.jig_release |= self.stand_up_counter >= self.cfg.jig_release_hold_step

        # Drop the jig out of the world exactly once per episode
        release_ids = (self.jig_release & ~self.is_release).nonzero(as_tuple=False).flatten()
        if release_ids.numel() > 0:
            jig_pose = self._jig.data.root_state_w[release_ids, :7].clone()
            jig_pose[:, 2] = self.cfg.jig_release_depth
            self._jig.write_root_pose_to_sim(jig_pose, env_ids=release_ids)
            self.is_release[release_ids] = True
