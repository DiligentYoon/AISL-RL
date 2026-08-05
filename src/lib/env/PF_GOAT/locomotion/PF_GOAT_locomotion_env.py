# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import copy

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter
from isaaclab.markers import VisualizationMarkers

from isaaclab.utils.math import quat_apply_inverse, yaw_quat, euler_xyz_from_quat, quat_apply

from lib.domain_randomizer.commander import UniformNonHolonomicCommand
from lib.env.PF_GOAT.locomotion.PF_GOAT_locomotion_env_cfg import PFGOATLocomotionEnvCfg, PFGOATLocomotionPlayEnvCfg
from lib.env.PF_GOAT.base.PF_GOAT_base_env import PFGOATBaseEnv

class PFGOATLocomotionEnv(PFGOATBaseEnv):
    cfg: PFGOATLocomotionEnvCfg | PFGOATLocomotionPlayEnvCfg

    def __init__(self, cfg: PFGOATLocomotionEnvCfg | PFGOATLocomotionPlayEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Commands for reference generator
        self.commands = UniformNonHolonomicCommand(self.cfg.commands, self._robot, self.device)

        # Action scale factor
        self.cfg.action_scale_factor["joint"][1] = self.joint_ids

        # Intermediate values
        self.base_pos_w         = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.base_rot_w         = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.base_lin_vel       = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.base_ang_vel       = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.base_height        = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.gravity_vector     = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.joint_pos          = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
        self.joint_vel          = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
        self.command_inputs_b   = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.air_time           = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self.contact_time       = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self.in_contact         = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self.foot_pos_w         = torch.zeros((self.num_envs, 2, 3), dtype=torch.float, device=self.device)
        self.foot_rot_w         = torch.zeros((self.num_envs, 2, 4), dtype=torch.float, device=self.device)

        # Contact states
        self.is_contacts = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)

        # Gait guidance (Gait scheduler)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) 
        self.update_phase_ids = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.command_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.update_command_ids = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.step_period = int(self.cfg.time_period / self.step_dt) * torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.full_step_period = int(2 * self.cfg.time_period / self.step_dt) * torch.ones(self.num_envs, dtype=torch.long, device=self.device)

        self.contact_schedule = torch.zeros(self.num_envs, device=self.device)
        self.phase_sin = torch.zeros(self.num_envs, device=self.device)
        self.phase_cos = torch.zeros(self.num_envs, device=self.device)

        # Gait guidance (Foot State)
        self.foot_on_swing = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device) # True foot is on command (=swing)

        self.support_foot_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.support_foot_rot = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device) 

        # Prev action
        self.previous_actions = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # Regularization
        self.out_of_limits_joint    = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
        self.out_of_limits_torque   = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
        self.out_of_limits_velocity = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.applied_torque         = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.joint_acc              = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.deviation_hip          = torch.zeros((self.num_envs, len(self.hip_joint_ids)), dtype=torch.float, device=self.device)

        # Visualization
        debug_vis = self.num_envs <= 32
        self.set_debug_vis(debug_vis)


    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
            if not hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
            if hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self._robot.is_initialized:
            return
        # ============== Arrow ================ # 
        # Arrow: get marker location
        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # Arrow: resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self.commands._resolve_xy_velocity_to_arrow(scale=self.goal_vel_visualizer.cfg.markers["arrow"].scale,
                                                                                              xy_velocity=self.commands.command_b)
        
        vel_arrow_scale, vel_arrow_quat = self.commands._resolve_xy_velocity_to_arrow(scale=self.current_vel_visualizer.cfg.markers["arrow"].scale,
                                                                                      xy_velocity=self._robot.data.root_lin_vel_b[:, :2])

        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _setup_scene(self):
        super()._setup_scene()
        # Terrain
        self.terrain = TerrainImporter(self.cfg.terrain)
        self.cfg.dome_light_cfg.spawn.func(self.cfg.dome_light_cfg.prim_path,
                                           self.cfg.dome_light_cfg.spawn)
        # Commands cfg
        self.cfg.commands.num_envs = self.scene.num_envs
        self.cfg.commands.step_dt = self.step_dt
        # Collision filtering
        global_prim_paths = []
        if hasattr(self.cfg, "terrain") and hasattr(self.cfg.terrain, "prim_path"):
            global_prim_paths.append(self.cfg.terrain.prim_path)
        self.scene.filter_collisions(global_prim_paths=global_prim_paths)

    def _get_observations(self) -> torch.Tensor:
        """
        Get sensor data without curriculum Gaussian noise

        Returns:
            Observation space
        """
        observation = torch.cat((self.base_ang_vel,                         # [E, 3]
                                 self.base_rot_w,                           # [E, 4]
                                 self.command_inputs_b,                     # [E, 3]
                                 self.joint_pos - self.default_joint_pos,   # [E, 6]
                                 self.joint_vel,                            # [E, 6]
                                 self.previous_actions,                     # [E, 6]
                                ), dim=1) 

        return observation


    def _get_states(self) -> torch.Tensor:
        """"
        Get State space using previleged information

        Returns
            State space
        """
        observation = torch.cat((self.base_ang_vel,                            # [E, 3]
                                 self.base_rot_w,                              # [E, 4]
                                 self.command_inputs_b,                        # [E, 3]
                                 self.joint_pos - self.default_joint_pos,      # [E, 6]
                                 self.joint_vel,                               # [E, 6]
                                 self.previous_actions,                        # [E, 6]
                                 ), dim=1)                             
        
        privileged_info = torch.cat((self.base_lin_vel,                             # [E, 3]
                                     self.base_height), dim=1)                      # [E, 1]
        
        state = torch.cat([observation, privileged_info], dim=-1)

        return state
    

    def _get_rewards(self) -> torch.Tensor:
        # Orientation Reward
        upright_error = torch.sum(torch.square(self.gravity_vector[:, :2]), dim=1)
        r_upright = torch.exp(-upright_error / 0.25)

        # Height tracking Reward
        height_error = torch.reshape(torch.abs(self.base_height - self.cfg.target_height), (-1,))
        r_height = torch.exp(-height_error / 0.03)

        # Command Tracking Reward
        lin_vel_error = torch.sum(torch.square(self.command_inputs_b[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.command_inputs_b[:, 2] - self.base_ang_vel[:, 2])
        r_lin_vel_tracking = torch.exp(-lin_vel_error / 0.05)
        r_ang_vel_tracking = torch.exp(-ang_vel_error / 0.05)

        # Gait rewards
        diff = self.in_contact[:, 1].float() - self.in_contact[:, 0].float()  # right support (+), left support (-), double support (0)
        r_gait = diff * self.contact_schedule

        # Regularization Penalty
        p_ang_vel_xy         = -torch.norm(self.base_ang_vel[:, :2], dim=-1)           
        p_all_torque         = -torch.sum(torch.square(self.applied_torque), dim=1)
        p_joint_velocity     = -torch.sum(torch.square(self.joint_vel), dim=1)   
        p_joint_accel        = -torch.sum(torch.square(self.joint_acc), dim=1)                      
        p_action_rate        = -torch.sum(torch.abs((self.actions - self.previous_actions)), dim=1)
        p_terminated         = -self.reset_terminated.float()

        # Total Reward Summation
        total_reward = (
            self.cfg.r_upright_weight * r_upright                 +
            self.cfg.r_height_weight * r_height                   +
            self.cfg.r_lin_vel_weight * r_lin_vel_tracking        +
            self.cfg.r_ang_vel_weight * r_ang_vel_tracking        +
            self.cfg.r_gait_weight * r_gait                       + 
            self.cfg.p_ang_vel_xy_weight * p_ang_vel_xy           +
            self.cfg.p_all_torque_weight * p_all_torque           +
            self.cfg.p_joint_velocity_weight * p_joint_velocity   +
            self.cfg.p_joint_accel_weight * p_joint_accel         +
            self.cfg.p_action_rate_weight * p_action_rate         +
            self.cfg.p_termination_weight * p_terminated
        )

        self.extras["reward"] = {
            # ==========================================
            # Task Reward (+)
            # ==========================================
            "Task Reward / Upright"             : self.cfg.r_upright_weight * r_upright,
            "Task Reward / Height"              : self.cfg.r_height_weight * r_height,
            "Task Reward / Lin_Vel_Tracking"    : self.cfg.r_lin_vel_weight * r_lin_vel_tracking,
            "Task Reward / Ang_Vel_Tracking"    : self.cfg.r_ang_vel_weight * r_ang_vel_tracking,
            "Task Reward / Gait"                : self.cfg.r_gait_weight * r_gait,
            # ==========================================
            # Task Penalty (-)
            # ==========================================
            "Task Penalty / Ang_Vel_XY"         : self.cfg.p_ang_vel_xy_weight * p_ang_vel_xy,
            "Task Penalty / Torque"             : self.cfg.p_all_torque_weight * p_all_torque,
            "Task Penalty / Joint_Vel"          : self.cfg.p_joint_velocity_weight * p_joint_velocity,
            "Task Penalty / Joint_Acc"          : self.cfg.p_joint_accel_weight * p_joint_accel,
            "Task Penalty / Action_Rate"        : self.cfg.p_action_rate_weight * p_action_rate,
        }

        self.previous_actions = self.actions.clone()

        
        return total_reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        projected_gravity_x = self.gravity_vector[:, 0]
        projected_gravity_y = self.gravity_vector[:, 1]

        died_fall   = self.base_height[:, 0] <= self.cfg.termination_height
        died_fall_2 = torch.logical_or(torch.abs(projected_gravity_x) >= self.cfg.termination_gravity,
                                       torch.abs(projected_gravity_y) >= self.cfg.termination_gravity)
        died_ang = torch.norm(self.base_ang_vel[:, :3], dim=-1) >= self.cfg.termination_ang_vel
        
        died = died_fall | died_fall_2 | died_ang
        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        # Randomization by Event-based randomizer
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Gait guidance (Phase scheduler)
        self.phase[env_ids] = 0
        self.phase_count[env_ids] = 0
        self.update_phase_ids[env_ids] = False
        self.command_count[env_ids] = 0
        self.update_command_ids[env_ids] = False

        # Gait guidance (Foot state)
        self.foot_on_swing[env_ids] = 0
        self.foot_on_swing[env_ids, 0] = 1 # Initial swing feet is the left feet

        # Prev actions
        self.previous_actions[env_ids] = 0.0

        # Command resampling
        self.commands.reset(env_ids)

        self._compute_intermediate_values(env_ids)


    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        i = env_ids if env_ids is not None else self._robot._ALL_INDICES
        # Robot data
        self.base_pos_w[i] = self._robot.data.root_pos_w[i]
        self.base_rot_w[i] = self._robot.data.root_quat_w[i] # (w, x, y, z)
        self.base_ang_vel[i] = self._robot.data.root_ang_vel_b[i]
        self.gravity_vector[i] = self._robot.data.projected_gravity_b[i]                  
        self.joint_pos[i] = self._robot.data.joint_pos[i]
        self.joint_vel[i] = self._robot.data.joint_vel[i]
        # Privileged data
        self.base_lin_vel[i] = self._robot.data.root_lin_vel_b[i]
        self.base_height[i] = self._robot.data.root_pos_w[i, 2].unsqueeze(-1)
        # Joint Angle & Velocity
        self.joint_pos[i], self.joint_vel[i] = self._robot.data.joint_pos[i], self._robot.data.joint_vel[i]
        # Information related to Commands Tracking
        self.command_inputs_b[i] = self.commands.command_b[i]
        # Information related to Contact
        self.air_time[i] = self.contact_sensors.data.current_air_time[i][:, self.contact_calf_link_ids] # [Left, Right]
        self.contact_time[i] = self.contact_sensors.data.current_contact_time[i][:, self.contact_calf_link_ids] # [Left, Right]
        self.in_contact[i] = self.contact_time[i] > 0.0 # [E, 2 (Left, Right)]
        # Feet Slide
        self.is_contacts[i] = self.contact_sensors.data.net_forces_w_history[i][:, :, self.contact_calf_link_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

        # Gait guidance (Phase scheduler)
        self.foot_pos_w[i] = self._robot.data.body_link_pos_w[i][:, self.calf_link_ids] # [Left, Right]
        self.foot_rot_w[i] = self._robot.data.body_link_quat_w[i][:, self.calf_link_ids] # [Left, Right]
        if env_ids is not None:
            # Update only reset env
            self.support_foot_pos[i] = self.foot_pos_w[i, 1, :3] # Left swing and Right support
            self.support_foot_rot[i] = self.foot_rot_w[i, 1, :4] # Left swing and Right support
        else:
            # Only full progress env
            # Phase update signal
            self.phase += 1 / self.full_step_period
            self.phase_count += 1
            self.update_phase_ids = (self.phase_count >= self.full_step_period)
            # Step command update signal
            self.command_count += 1
            self.update_command_ids = (self.command_count >= self.step_period)
            # Schedule variables
            phase_update_mask = self.update_phase_ids.clone()
            command_update_mask = self.update_command_ids.clone()
            self.phase[phase_update_mask] = 0
            self.phase_count[phase_update_mask] = 0
            self.command_count[command_update_mask] = 0
            # Foot switching
            if torch.any(self.update_command_ids):
                # Switch the swing foot
                self.foot_on_swing[self.update_command_ids] = ~self.foot_on_swing[self.update_command_ids]
                # Switch the support foot
                left_support_mask  = (self.foot_on_swing[:, 0] == 0) 
                right_support_mask = (self.foot_on_swing[:, 1] == 0)
                # Envs which require left stance state
                left_combined_mask = self.update_command_ids & left_support_mask
                # Envs which require right stance state
                right_combined_mask = self.update_command_ids & right_support_mask
                # Update support foot pos and rot
                self.support_foot_pos[left_combined_mask]  = self.foot_pos_w[left_combined_mask, 0, :3]
                self.support_foot_rot[left_combined_mask]  = self.foot_rot_w[left_combined_mask, 0, :4]
                self.support_foot_pos[right_combined_mask] = self.foot_pos_w[right_combined_mask, 1, :3]
                self.support_foot_rot[right_combined_mask] = self.foot_rot_w[right_combined_mask, 1, :4]

        # Contact schedule
        self.contact_schedule[i] = smooth_sqr_wave(self.phase[i])
        # Phase variable
        self.phase_sin[i] = torch.sin(2*torch.pi*self.phase[i])
        self.phase_cos[i] = torch.cos(2*torch.pi*self.phase[i])

        # Regularization Parameter
        self.out_of_limits_joint[i]  = -(self.joint_pos[i] - self._robot.data.soft_joint_pos_limits[i, :, 0]).clip(max=0.0) + \
                                        (self.joint_pos[i] - self._robot.data.soft_joint_pos_limits[i, :, 1]).clip(min=0.0)
        self.out_of_limits_torque[i] = (torch.abs(self._robot.data.applied_torque[i]) - self.torque_limits[i] * self.cfg.soft_torque_limit).clip(min=0.0)
        self.out_of_limits_velocity[i] = (torch.abs(self.joint_vel[i]) - self.cfg.joint_vel_limit).clip(min=0.0)
        self.applied_torque[i]       = self._robot.data.applied_torque[i]
        self.deviation_hip[i]     = self.joint_pos[i][:, self.hip_joint_ids] - self._robot.data.default_joint_pos[i][:, self.hip_joint_ids]
        self.joint_acc[i] = self._robot.data.joint_acc[i]

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*torch.pi
    angles -= 2*torch.pi * (angles > torch.pi)
    return angles

@torch.jit.script
def smooth_sqr_wave(phase):
    p = 2.*torch.pi*phase
    eps = 0.2
    return torch.sin(p) / torch.sqrt(torch.sin(p)**2. + eps**2.)