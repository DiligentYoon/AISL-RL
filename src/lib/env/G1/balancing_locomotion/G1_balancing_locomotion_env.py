# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import copy

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter
from isaaclab.markers import VisualizationMarkers

from isaaclab.utils.math import quat_apply_inverse, yaw_quat, euler_xyz_from_quat, quat_apply, quat_from_euler_xyz

from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg

from lib.domain_randomizer.commander import UniformVelocityCommand
from lib.env.G1.base.G1_base_env import G1BaseEnv
from lib.env.G1.balancing_locomotion.G1_balancing_locomotion_env_cfg import G1BalancingLocomotionEnvCfg

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class G1BalancingLocomotionEnv(G1BaseEnv):
    cfg: G1BalancingLocomotionEnvCfg

    def __init__(self, cfg: G1BalancingLocomotionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Commands for reference generator
        self.commands = UniformVelocityCommand(self.cfg.commands, self._robot, self.device)

        # Action Mapping
        self.mapping_sort_ids = torch.argsort(torch.tensor(self.total_arm_joint_ids + self.total_leg_joint_ids, device=self.device))

        # Joint Ids (Leg)
        self.ankle_joint_ids, _ = self._robot.find_joints([".*_ankle_pitch_joint", 
                                                           ".*ankle_roll_joint"])
        
        self.knee_joint_ids, _ = self._robot.find_joints([".*_knee_joint"])

        self.hip_joint_ids, _ = self._robot.find_joints([".*_hip_yaw_joint", 
                                                         ".*_hip_roll_joint"])
        
        self.hip_knee_joint_ids, _ = self._robot.find_joints([".*_hip_.*", 
                                                               ".*_knee_joint"])

        # Joint Ids (Arm)
        self.arm_joint_ids, _ = self._robot.find_joints([".*_shoulder_pitch_joint",
                                                         ".*_shoulder_roll_joint",
                                                         ".*_shoulder_yaw_joint",
                                                         ".*_elbow_pitch_joint",
                                                         ".*_elbow_roll_joint"])
        
        self.finger_joint_ids, _ = self._robot.find_joints([".*_five_joint",
                                                            ".*_three_joint",
                                                            ".*_six_joint",
                                                            ".*_four_joint",
                                                            ".*_zero_joint",
                                                            ".*_one_joint",
                                                            ".*_two_joint"])
        
        self.torso_joint_ids, _ = self._robot.find_joints("torso_joint")
        
        # Joint Limits
        self.leg_joint_limits = self.joint_pos_limits[:, self.total_leg_joint_ids]
        self.arm_joint_limits = self.joint_pos_limits[:, self.total_arm_joint_ids]
        
        # Link ids
        self.torso_link_ids, _ = self._robot.find_bodies("torso_link")
        self.ankle_roll_link_ids, _ = self._robot.find_bodies(".*_ankle_roll_link")

        # Contact Link ids
        self.torso_contact_link_ids, _ = self.contact_sensors.find_bodies("torso_link")
        self.ankle_contact_roll_link_ids, _ = self.contact_sensors.find_bodies(".*_ankle_roll_link")

        # Action scale factor
        self.cfg.action_scale_factor["arm"][1] = self.total_arm_joint_ids
        self.cfg.action_scale_factor["leg"][1] = self.total_leg_joint_ids

        # Foot states
        self.foot_pos_w = torch.zeros((self.num_envs, 2, 3), dtype=torch.float, device=self.device)
        self.foot_pos_b = torch.zeros((self.num_envs, 2, 3), dtype=torch.float, device=self.device)

        # Robot Property
        self.robot_mass = self._robot.data.default_mass.to(self.device)
        self.total_mass = self._robot.data.default_mass.sum(dim=-1).to(self.device)

        # Gait Guidance
        self.z_c = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.foot_on_swing = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device) # True foot is on command (=swing)

        self.phase_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) # Number of phase progress
        self.update_phase_ids = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # envs whose phases are updated
        self.phase = torch.zeros(self.num_envs, device=self.device) # phase of current step in a whole gait cycle
        
        self.update_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.update_command_ids = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.step_period = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.full_step_period = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) # full_step_period = 2 * step_period
        self.dstep_width = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.support_foot_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device) # position of the support foot
        self.support_foot_rot = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device) # rotation of the support foot
        self.prev_support_foot_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device) # position of the support foot
        
        self.prev_target_footstep_w = torch.zeros((self.num_envs, 2, 3), dtype=torch.float, device=self.device)
        self.target_footstep_w = torch.zeros((self.num_envs, 2, 3), dtype=torch.float, device=self.device)
        self.prev_target_footstep_b = torch.zeros((self.num_envs, 2, 3), dtype=torch.float, device=self.device)
        self.target_footstep_b = torch.zeros((self.num_envs, 2, 3), dtype=torch.float, device=self.device)
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # Reward Info for logging
        self.extras["reward"] = {
            "arm": {},
            "leg": {}
        }

        debug_vis = self.num_envs <= 32
        self.set_debug_vis(debug_vis)


    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
            if not hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            if not hasattr(self, "target_foot_visualizer"):
                self.target_foot_visualizer = VisualizationMarkers(self.cfg.target_foot_visualizer_cfg)
            if not hasattr(self, "target_foot_rotation_visualizer"):
                self.target_foot_rotation_visalizer = VisualizationMarkers(self.cfg.target_foot_rotation_visualizer_cfg)
            if not hasattr(self, "foot_rotation_visualizer"):
                self.foot_rotation_visalizer = VisualizationMarkers(self.cfg.target_foot_rotation_visualizer_cfg)
            if not hasattr(self, "torso_rotation_visualizer"):
                self.torso_rotation_visalizer = VisualizationMarkers(self.cfg.torso_rotation_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
            self.target_foot_visualizer.set_visibility(True)
            self.target_foot_rotation_visalizer.set_visibility(True)
            self.foot_rotation_visalizer.set_visibility(True)
            self.torso_rotation_visalizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
            if hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer.set_visibility(False)
            if hasattr(self, "target_foot_visualizer"):
                self.target_foot_visualizer.set_visibility(False)
            if hasattr(self, "target_foot_rotation_visualizer"):
                self.target_foot_rotation_visalizer.set_visibility(False)
                self.target_foot_visualizer.set_visibility(False)
            if hasattr(self, "foot_rotation_visualizer"):
                self.foot_rotation_visalizer.set_visibility(False)
            if hasattr(self, "torso_rotation_visualizer"):
                self.torso_rotation_visalizer.set_visibility(False)

    

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
        
        # ============== Target Foot Cube and Rotation Frame ================ # 
        pos = self.target_footstep_w[..., :2].clone()               # [E, 2, 2]
        z_height = 0.01 * torch.ones_like(pos[..., :1])             # [E, 2, 1]
        target_pos = torch.cat([pos, z_height], dim=-1).view(-1, 3) # [E*2, 3]

        target_yaw = self.target_footstep_w[..., 2].view(-1).clone()    # [E*2, 1]
        target_quat = quat_from_euler_xyz(torch.zeros_like(target_yaw), # [E*2, 1]
                                          torch.zeros_like(target_yaw), # [E*2, 1]
                                          target_yaw)                   # [E*2, 1]
        
        # Color difference between support (green) and swing (red) foot
        marker_indices = torch.where(self.foot_on_swing.view(-1), 0, 1)

        
        # =============== Foot Rotation Frame =============== #
        foot_pos = self.foot_pos_w.clone().view(-1, 3)
        foot_rot = self.foot_rot_w.clone().view(-1, 4)
        foot_marker_indices = torch.tensor([0, 0], device=self.device).repeat(self.num_envs)

        # =============== Torso ================
        torso_pos = self.torso_pos_w
        torso_rot = self.torso_rot_w
        

        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)
        self.target_foot_visualizer.visualize(translations=target_pos, 
                                              orientations=target_quat, 
                                              marker_indices=marker_indices)
        self.target_foot_rotation_visalizer.visualize(translations=target_pos,
                                                      orientations=target_quat,
                                                      marker_indices=foot_marker_indices)
        self.foot_rotation_visalizer.visualize(translations=foot_pos,
                                               orientations=foot_rot,
                                               marker_indices=foot_marker_indices)
        self.torso_rotation_visalizer.visualize(translations=torso_pos,
                                                orientations=torso_rot)


    def _setup_scene(self):
        super()._setup_scene()
        # sensor
        self.scene.sensors["contact_forces"] = ContactSensor(self.cfg.contact_forces)
        # self.scene.sensors["height_scanner"] = RayCaster(self.cfg.height_scanner)
        self.contact_sensors = self.scene.sensors["contact_forces"]
        # self.height_scanner = self.scene.sensors["height_scanner"]
        # self.height_scanner.update_period = self.cfg.decimation * self.cfg.sim_dt
        self.contact_sensors.update_period = self.cfg.sim_dt * self.cfg.decimation
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add ground plane
        self.cfg.terrain_importer_cfg.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain_importer_cfg.env_spacing = self.scene.cfg.env_spacing
        self.terrain = TerrainImporter(self.cfg.terrain_importer_cfg)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # add commands cfg
        self.cfg.commands.num_envs = self.scene.num_envs
        self.cfg.commands.step_dt = self.step_dt


    def _pre_physics_step(self, actions: dict[str, torch.Tensor] | torch.Tensor):
        self.actions = actions


    def _apply_action(self):
        if self.cfg.num_agents > 1:
            # Multi Agent
            arm_actions = self.actions["arm"]
            leg_actions = self.actions["leg"]

            self._robot.set_joint_position_target(
                target=torch.clamp(self._robot.data.default_joint_pos[:, self.total_arm_joint_ids] + arm_actions,
                                   min=self.arm_joint_limits[:, :, 0],
                                   max=self.arm_joint_limits[:, :, 1]),
                joint_ids=self.total_arm_joint_ids
            )

            self._robot.set_joint_position_target(
                target=torch.clamp(self._robot.data.default_joint_pos[:, self.total_leg_joint_ids] + leg_actions,
                                   min=self.leg_joint_limits[:, :, 0],
                                   max=self.leg_joint_limits[:, :, 1]),
                joint_ids=self.total_leg_joint_ids)
            
        else:
            self._robot.set_joint_position_target(
                target=torch.clamp(self._robot.data.default_joint_pos[:, self._joint_dof_ids] + self.actions,
                                   min=self._robot.data.joint_pos_limits[:, self._joint_dof_ids, 0],
                                   max=self._robot.data.joint_pos_limits[:, self._joint_dof_ids, 1]),
                joint_ids=self._joint_dof_ids
            )


    def _get_observations(self) -> dict[str, torch.Tensor] | torch.Tensor:
        if self.cfg.num_agents > 1:
            # Multi Agent
            observations = {
                "arm": torch.cat(
                    [
                        self.CoM[:, 2:3],                                # [E, 1]
                        self.torso_lin_vel_b,                            # [E, 3]
                        self.torso_ang_vel_b,                            # [E, 3]
                        self.projected_gravity,                          # [E, 3]
                        self.command_inputs_b,                           # [E, 3]
                        self.joint_pos[:, self.total_arm_joint_ids],     # [E, 25]
                        self.joint_vel[:, self.total_arm_joint_ids],     # [E, 25]
                        self.actions["arm"]                              # [E, 25]
                    ],
                    dim=-1
                ),
                "leg": torch.cat(
                    [
                        self.CoM[:, 2:3],                                   # [E, 1]
                        self.torso_lin_vel_b,                               # [E, 3]
                        self.torso_ang_vel_b,                               # [E, 3]    
                        self.projected_gravity,                             # [E, 3]
                        self.command_inputs_b,                              # [E, 3]
                        self.full_step_period.unsqueeze(-1) * self.step_dt, # [E, 1]
                        self.z_c.unsqueeze(-1),                             # [E, 1]
                        self.phase_sin.unsqueeze(-1),                       # [E, 1]
                        self.phase_cos.unsqueeze(-1),                       # [E, 1]
                        self.foot_pos_b.view(self.num_envs, -1),            # [E, 6]
                        self.foot_rot_yaw_b.view(self.num_envs, -1),        # [E, 2]
                        self.target_footstep_b.view(self.num_envs, -1),     # [E, 6] 
                        self.target_footstep_yaw_b.view(self.num_envs, -1), # [E, 2] 
                        self.joint_pos[:, self.total_leg_joint_ids],        # [E, 12]
                        self.joint_vel[:, self.total_leg_joint_ids],        # [E, 12]
                        self.actions["leg"]                                 # [E, 12]
                    ],
                    dim=-1
                ),
            }
        else:
            actions = torch.cat([self.actions["arm"], self.actions["leg"]], dim=-1)
            sorted_actions = actions[:, self.mapping_sort_ids]
            observations = torch.cat(
                [
                    self.CoM[:, 2:3],                                   # [E, 1]
                    self.torso_lin_vel_b,                               # [E, 3]
                    self.torso_ang_vel_b,                               # [E, 3]
                    self.projected_gravity,                             # [E, 3]
                    self.command_inputs_b,                              # [E, 3]
                    self.full_step_period.unsqueeze(-1) * self.step_dt, # [E, 1]
                    self.z_c.unsqueeze(-1),                             # [E, 1]
                    self.phase_sin.unsqueeze(-1),                       # [E, 1]
                    self.phase_cos.unsqueeze(-1),                       # [E, 1]
                    self.foot_pos_b.view(self.num_envs, -1),            # [E, 6]
                    self.foot_rot_yaw_b.view(self.num_envs, -1),        # [E, 2]
                    self.target_footstep_b.view(self.num_envs, -1),     # [E, 6] 
                    self.target_footstep_yaw_b.view(self.num_envs, -1), # [E, 2] 
                    self.joint_pos,                                     # [E, 37]
                    self.joint_vel,                                     # [E, 37]
                    sorted_actions                                      # [E, 37]  
                ],
                dim=-1
            )

        return observations

    def _get_states(self) -> dict[str, torch.Tensor] | torch.Tensor:
        if self.cfg.num_agents > 1:
            # Multi Agent
            actions = torch.cat([self.actions["arm"], self.actions["leg"]], dim=-1)
            sorted_actions = actions[:, self.mapping_sort_ids]
            shared_states = torch.cat(
                [
                    self.CoM[:, 2:3],                                   # [E, 1]
                    self.torso_lin_vel_b,                               # [E, 3]
                    self.torso_ang_vel_b,                               # [E, 3]
                    self.projected_gravity,                             # [E, 3]
                    self.command_inputs_b,                              # [E, 3]
                    self.full_step_period.unsqueeze(-1) * self.step_dt, # [E, 1]
                    self.z_c.unsqueeze(-1),                             # [E, 1]
                    self.phase_sin.unsqueeze(-1),                       # [E, 1]
                    self.phase_cos.unsqueeze(-1),                       # [E, 1]
                    self.foot_pos_b.view(self.num_envs, -1),            # [E, 6]
                    self.foot_rot_yaw_b.view(self.num_envs, -1),        # [E, 2]
                    self.target_footstep_b.view(self.num_envs, -1),     # [E, 6] 
                    self.target_footstep_yaw_b.view(self.num_envs, -1), # [E, 2] 
                    self.joint_pos,                                     # [E, 37]
                    self.joint_vel,                                     # [E, 37]
                    sorted_actions                                      # [E, 37] 
                ], dim=-1) 
            
            states = {
                "arm": shared_states,
                "leg": shared_states,
            }
        else:
            # Single Agent (Syncronous Actor Critic)
            states = None
            
        return states


    def _get_rewards(self) -> torch.Tensor:
        # Tracking Rewards (Torso)
        lin_vel_error = torch.sum(torch.square(self.command_inputs_b[:, :2] - self.vel_yaw[:, :2]), dim=1)
        ang_vel_error = torch.square(self.command_inputs_w[:, 2] - self.torso_ang_vel_w[:, 2])
        heading_error = torch.square(wrap_to_pi(self.commands.heading - self.torso_heading))
        height_error  = torch.square(self.CoM[:, 2] - self.z_c)
        lin_vel_rewards = torch.exp(-lin_vel_error / 0.5**2)
        ang_vel_rewards = torch.exp(-ang_vel_error / 0.5**2)
        heading_rewards = torch.exp(-heading_error / 0.5**2)
        height_rewards  = torch.exp(-height_error / 0.5**2)
        
        # Attitute rewards (Torso)
        tilting = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        flat_rewards = torch.exp(-tilting / 0.5**2)

        # Control Penalty (Total)
        ang_vel_xy_penalty = -torch.sum(torch.square(self.torso_ang_vel_b[:, :2]), dim=1)
        joint_limit_penalty   = -(torch.sum(self.out_of_limits_joint, dim=1).clip(0, 1))
        joint_torque_penalty = -torch.sum(torch.square(self._robot.data.applied_torque[:, self.hip_knee_joint_ids]), dim=1)
        joint_acc_penalty = -torch.sum(torch.square(self._robot.data.joint_acc[:, self.hip_knee_joint_ids]), dim=1)

        # Termination (Torso)
        terminate_penalty = -self.reset_terminated.float()

        # Gait Rewards (Leg)
        footstep_loc_error = torch.sum(self.step_location_offset * self.foot_on_swing.float(), dim=1)
        footstep_rot_error = torch.sum(self.step_rotation_offset * self.foot_on_swing.float(), dim=1)
        contact_schedule = (self.in_contact[:, 1].int() - self.in_contact[:, 0].int()) * self.contact_schedule
        footstep_tracking = torch.exp(-footstep_loc_error / 0.5**2) + torch.exp(-footstep_rot_error / 0.5**2)
        gait_reward = contact_schedule * footstep_tracking

        # print(f"contact (left) | contact (right) | contact_schedule | footstep tracking : {self.in_contact[:, 0].int().item()} | {self.in_contact[:, 1].int().item()} | {contact_schedule.item():.2f} | {footstep_tracking.item():.2f}")

        # Sliding Penalty (Leg)
        slide_penalty = -torch.sum(self._robot.data.body_link_lin_vel_w[:, self.ankle_roll_link_ids, :2].norm(dim=-1) * self.is_contacts, dim=1)

        # Joint Deviation Penalty (Arm)
        joint_pos_penalty_arms    = -torch.sum(torch.abs(self.deviation_arms), dim=1)    # Arm
        joint_pos_penalty_fingers = -torch.sum(torch.abs(self.deviation_fingers), dim=1) # Arm

        if self.cfg.possible_agents is not None:
            # Control Penalty (Leg and Arm)
            action_rate_penalty_leg = -torch.sum(torch.square(self.actions["leg"] - self.prev_actions["leg"]), dim=1)
            action_rate_penalty_arm = -torch.sum(torch.square(self.actions["arm"] - self.prev_actions["arm"]), dim=1)

            # Multi Agent
            common_rewards = self.cfg.w_track_lin_vel * lin_vel_rewards + \
                             self.cfg.w_track_ang_vel * ang_vel_rewards + \
                             self.cfg.w_track_heading * heading_rewards + \
                             self.cfg.w_track_height  * height_rewards + \
                             self.cfg.w_ang_vel_xy    * ang_vel_xy_penalty + \
                             self.cfg.w_flat          * flat_rewards + \
                             self.cfg.w_limits        * joint_limit_penalty + \
                             self.cfg.w_termination   * terminate_penalty

            arm_rewards = self.cfg.w_limits_arm     * joint_pos_penalty_arms + \
                          self.cfg.w_limits_fingers * joint_pos_penalty_fingers + \
                          self.cfg.w_action_rate    * action_rate_penalty_arm 
            
            leg_rewards = common_rewards + \
                          self.cfg.w_feet_gait     * gait_reward + \
                          self.cfg.w_feet_slide    * slide_penalty + \
                          self.cfg.w_joint_torque  * joint_torque_penalty + \
                          self.cfg.w_joint_acc     * joint_acc_penalty + \
                          self.cfg.w_action_rate   * action_rate_penalty_leg
            
            self.extras["reward"]["arm"] = {
                "Penalty / arm_deviation": torch.mean(joint_pos_penalty_arms).item(),
                "Penalty / finger_deviation": torch.mean(joint_pos_penalty_fingers).item()}
                          
            # Dictionary key order (alphabetical order in dictionary)
            rewards = torch.stack([arm_rewards, leg_rewards], dim=-1) # [E, 2]

            # Update Prev Actions (Multi Agent)
            self.prev_actions = {k: v.clone() for k, v in self.actions.items()}

        else:
            # Control Penalty (2)
            action_rate_penalty = -torch.sum(torch.square(self.actions - self.prev_actions), dim=1)

            # Single Agent
            rewards = self.cfg.w_track_lin_vel  * lin_vel_rewards + \
                      self.cfg.w_track_ang_vel  * ang_vel_rewards + \
                      self.cfg.w_track_heading  * heading_rewards + \
                      self.cfg.w_track_height   * height_rewards + \
                      self.cfg.w_feet_gait      * gait_reward + \
                      self.cfg.w_feet_slide     * slide_penalty + \
                      self.cfg.w_flat           * flat_rewards + \
                      self.cfg.w_limits         * joint_limit_penalty + \
                      self.cfg.w_limits_arm     * joint_pos_penalty_arms + \
                      self.cfg.w_limits_fingers * joint_pos_penalty_fingers + \
                      self.cfg.w_termination    * terminate_penalty + \
                      self.cfg.w_joint_torque   * joint_torque_penalty + \
                      self.cfg.w_joint_acc      * joint_acc_penalty + \
                      self.cfg.w_action_rate    * action_rate_penalty

            # Update Prev Actions (Single Agent)
            self.prev_actions = self.actions.clone()
        
        return rewards


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        torso_contact_forces = self.contact_sensors.data.net_forces_w_history[:, :, self.torso_contact_link_ids]
        projected_gravity_x = self.projected_gravity[:, 0]
        projected_gravity_y = self.projected_gravity[:, 1]

        swing_foot_pos = self.foot_pos_w[self.foot_on_swing]
        target_foot_pos = self.target_footstep_w[self.foot_on_swing]

        died_fall   = torch.any(torch.max(torch.norm(torso_contact_forces, dim=-1), dim=1)[0] > 1.0, dim=1)
        died_fall_2 = torch.logical_or(projected_gravity_x >= self.cfg.termination_gravity, projected_gravity_y >= self.cfg.termination_gravity)
        died_ang = torch.norm(self.torso_ang_vel_b, dim=-1) >= self.cfg.termination_ang_vel
        died_fall_3 = torch.norm(target_foot_pos[:, :2] - swing_foot_pos[:, :2], dim=-1) >= self.cfg.termination_target_foot
        died = died_fall | died_fall_2 | died_fall_3 | died_ang
        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        # Randomization by Event-based randomizer
        self._robot.reset(env_ids)
        self.commands.reset(env_ids)
        super()._reset_idx(env_ids)

        # Prev State Initialization
        if hasattr(self, "prev_actions"):
            if self.cfg.possible_agents is not None:
                # Multi Agent
                self.prev_actions["leg"][env_ids] = 0.0
                self.prev_actions["arm"][env_ids] = 0.0
            else:
                # Single Agent
                self.prev_actions[env_ids] = 0.0
        else:
            if self.cfg.possible_agents is not None:
                # Multi Agent
                self.prev_actions = {
                    "leg": torch.zeros((self.num_envs, len(self.total_leg_joint_ids)), device=self.device),
                    "arm": torch.zeros((self.num_envs, len(self.total_arm_joint_ids)), device=self.device)
                }
            else:
                # Single Agent
                self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        
        # Gait Phase Parameters
        self.z_c[env_ids] = self.cfg.z_c_min + (self.cfg.z_c_max - self.cfg.z_c_min) * torch.rand(env_ids.shape[0], device=self.device)# TODO: Randomizer에 기능 추가
        self.foot_on_swing[env_ids] = 0
        self.foot_on_swing[env_ids, 0] = 1 # Initial swing feet is the left feet
        self.update_phase_ids[env_ids] = True
        self.phase_count[env_ids] = 0
        self.phase[env_ids] = 0
        self.update_command_ids[env_ids] = True
        self.update_count[env_ids] = 0
        self.step_period, self.full_step_period, self.dstep_width = resample_commands(self.step_period,
                                                                                      self.full_step_period,
                                                                                      self.dstep_width,
                                                                                      env_ids,
                                                                                      self.step_dt,
                                                                                      self.cfg.time_period_min, self.cfg.time_period_max,
                                                                                      self.cfg.dstep_min, self.cfg.dstep_max) # TODO: Randomizer에 기능 추가

        self._compute_intermediate_values(env_ids)


    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        # Root Pose & Velocity
        self.torso_pos_w, self.torso_rot_w = self._robot.data.root_pos_w, self._robot.data.root_quat_w
        self.torso_heading = euler_xyz_from_quat(self.torso_rot_w)[2]
        self.torso_lin_vel_w, self.torso_ang_vel_w = self._robot.data.root_lin_vel_w, self._robot.data.root_ang_vel_w
        self.torso_lin_vel_b, self.torso_ang_vel_b = self._robot.data.root_lin_vel_b, self._robot.data.root_ang_vel_b
        # Attitude
        self.projected_gravity = self._robot.data.projected_gravity_b
        # Joint Angle & Velocity
        self.joint_pos, self.joint_vel = self._robot.data.joint_pos, self._robot.data.joint_vel
        # Height (For rough terrain)
        # self.height_scan = (self.height_scanner.data.pos_w[:, 2].unsqueeze(1) - self.height_scanner.data.ray_hits_w[..., 2] - 0.5).clip(min=-1.0, max=1.0)
        # Information related to Commands Tracking
        self.command_inputs_b = self.commands.command_b
        self.command_inputs_w = self.commands.command_w
        self.command_inputs_yaw = self.commands.command_yaw
        self.vel_yaw = quat_apply_inverse(yaw_quat(self.torso_rot_w), self.torso_lin_vel_w[:, :3]) # yaw of rot_w : (body -> world)
        # Information related to Contact
        self.air_time = self.contact_sensors.data.current_air_time[:, self.ankle_contact_roll_link_ids] # [Left, Right]
        self.contact_time = self.contact_sensors.data.current_contact_time[:, self.ankle_contact_roll_link_ids] # [Left, Right]
        self.in_contact = self.contact_time > 0.0 # [E, 2 (Left, Right)]

        # ==== Information related to Gait Guidance ====
        # Foot pos (World Frame)
        self.foot_pos_w = self._robot.data.body_link_pos_w[:, self.ankle_roll_link_ids] # [Left, Right]
        self.foot_rot_w = self._robot.data.body_link_quat_w[:, self.ankle_roll_link_ids] # [Left, Right]
        # Counting variables
        if env_ids is not None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_ids] = False
            if torch.any(mask):
                # Update without reset env
                self.phase[mask] += 1 / self.full_step_period[mask]
                self.phase_count[mask] += 1
                self.update_count[mask] += 1
                # Phase and command update signal
                self.update_phase_ids[mask] = (self.phase_count[mask] >= self.full_step_period[mask])
                self.update_command_ids[mask] = (self.update_count[mask] >= self.step_period[mask])
                # Counting variables
                combined_mask = self.update_command_ids & mask
                self.phase_count[combined_mask] = 0
                self.phase[combined_mask] = 0
                self.update_count[combined_mask] = 0
                # Prev commands
                self.prev_target_footstep_w[combined_mask] = self.target_footstep_w[combined_mask].clone()
                self.prev_target_footstep_b[combined_mask] = self.target_footstep_b[combined_mask].clone()
            # Update only reset env
            self.support_foot_pos[env_ids] = self.foot_pos_w[env_ids, 1, :3] # Left swing and Right support
            self.support_foot_rot[env_ids] = self.foot_rot_w[env_ids, 1, :4] # Left swing and Right support
            self.prev_target_footstep_w[env_ids] = 0
            self.prev_target_footstep_b[env_ids] = 0
        else:
            # Total Update
            self.phase += 1 / self.full_step_period
            self.phase_count += 1
            self.update_count += 1
            # Phase and command update signal
            self.update_phase_ids = (self.phase_count >= self.full_step_period)
            self.update_command_ids = (self.update_count >= self.step_period)
            # Counting variables
            self.phase_count[self.update_phase_ids] = 0
            self.phase[self.update_phase_ids] = 0
            self.update_count[self.update_command_ids] = 0
            # Prev commands
            self.prev_target_footstep_w[self.update_command_ids] = self.target_footstep_w[self.update_command_ids].clone()
            self.prev_target_footstep_b[self.update_command_ids] = self.target_footstep_b[self.update_command_ids].clone()

        # Center of Mass (CoM)
        self.CoM = (self._robot.data.body_link_pos_w * self.robot_mass.unsqueeze(-1)).sum(dim=1) / self.total_mass.unsqueeze(-1)

        # Target foot pos update
        if torch.any(self.update_command_ids):
            update_commands_mask = self.target_footstep_w[self.update_command_ids].clone()
            # Switch the swing foot
            if env_ids is not None:
                combined_mask = self.update_command_ids & mask # Not reset env and command updated env
            else:
                combined_mask = self.update_command_ids # command updated env

            self.foot_on_swing[combined_mask] = ~self.foot_on_swing[combined_mask] # NOTE: Assume single stance (double stance -> guide left single, right stand)

            # Switch the support foot
            left_support_mask  = (self.foot_on_swing[:, 0] == 0) 
            right_support_mask = (self.foot_on_swing[:, 1] == 0)
            
            left_combined_mask = combined_mask & left_support_mask   # valid ``env and left support env
            right_combined_mask = combined_mask & right_support_mask # valid env and right support env

            # ============== Support Foot Update ==============
            self.support_foot_pos[left_combined_mask] = self.foot_pos_w[left_combined_mask, 0, :3]
            self.support_foot_pos[right_combined_mask] = self.foot_pos_w[right_combined_mask, 1, :3]
            self.support_foot_rot[left_combined_mask] = self.foot_rot_w[left_combined_mask, 0, :4]
            self.support_foot_rot[right_combined_mask] = self.foot_rot_w[right_combined_mask, 1, :4]

            # Update footstep command
            support_yaw = euler_xyz_from_quat(self.support_foot_rot[self.update_command_ids])[2]
            update_commands_mask[~self.foot_on_swing[self.update_command_ids]] = torch.cat([self.support_foot_pos[self.update_command_ids, :2], support_yaw.unsqueeze(-1)], dim=-1)
            update_commands_mask[self.foot_on_swing[self.update_command_ids]] = self.compute_target_footstep()

            foot_collision_ids = (update_commands_mask[:, 0, :2] - update_commands_mask[:, 1, :2]).norm(dim=1) < self.cfg.self_collision_threshold
            if torch.any(foot_collision_ids):
                update_commands_mask[foot_collision_ids, :, :2] = adjust_foot_collision(update_commands_mask[foot_collision_ids, :, :2], 
                                                                                        self.foot_on_swing[self.update_command_ids][foot_collision_ids],
                                                                                        self.cfg.self_collision_threshold)
            self.target_footstep_w[self.update_command_ids] = update_commands_mask

        # Phase variable
        self.phase_sin = torch.sin(2*torch.pi*self.phase)
        self.phase_cos = torch.cos(2*torch.pi*self.phase)

        # Foot states (Body Frame)
        foot_forward_w_left = quat_apply(self.foot_rot_w[:, 0], self.forward_vec) # rot_w : (base -> world)
        foot_forward_w_right = quat_apply(self.foot_rot_w[:, 1], self.forward_vec) # rot_w : (base -> world)
        foot_forward_w = torch.cat([foot_forward_w_left.unsqueeze(1), foot_forward_w_right.unsqueeze(1)], dim=1)
        foot_forward_b_left = quat_apply_inverse(self.torso_rot_w, foot_forward_w_left) # rot_w (base -> world) [E, 3]
        foot_forward_b_right = quat_apply_inverse(self.torso_rot_w, foot_forward_w_right) # rot_w (base -> world) [E, 3]
        foot_forward_b = torch.cat([foot_forward_b_left.unsqueeze(1), foot_forward_b_right.unsqueeze(1)], dim=1) # [E, 2, 3]
        self.foot_rot_yaw_w = torch.atan2(foot_forward_w[..., 1], foot_forward_w[..., 0])
        self.foot_rot_yaw_b = torch.atan2(foot_forward_b[..., 1], foot_forward_b[..., 0])

        left_foot_pos_b = quat_apply_inverse(self.torso_rot_w, self.foot_pos_w[:, 0, :3] - self.torso_pos_w) # [E, 3]
        right_foot_pos_b = quat_apply_inverse(self.torso_rot_w, self.foot_pos_w[:, 1, :3] - self.torso_pos_w) # [E, 3]
        self.foot_pos_b = torch.cat([left_foot_pos_b.unsqueeze(1), right_foot_pos_b.unsqueeze(1)], dim=1)

        if env_ids is not None:
            self.target_footstep_w[env_ids, 1, :2] = self.foot_pos_w[env_ids, 1, :2] # NOTE: Right foot is support foot at initial state
            self.target_footstep_w[env_ids, 1, 2]  = self.foot_rot_yaw_w[env_ids, 1] # Right foot

        # Contact schedule
        self.contact_schedule = smooth_sqr_wave(self.phase)
        self.step_location_offset = torch.norm(self.foot_pos_w[:, :, :3] - \
                                               torch.cat([self.target_footstep_w[:, :, :2], torch.zeros((self.num_envs, 2, 1), device=self.device)], dim=-1), dim=-1) # [E, 2]
        self.step_rotation_offset = torch.abs(
            wrap_to_pi(self.target_footstep_w[:, :, 2] - self.foot_rot_yaw_w)) # [E, 2]

        # Command pos (Body Frame)
        target_left_footstep_b  = quat_apply_inverse(self.torso_rot_w, 
                                                     torch.cat([self.target_footstep_w[:, 0, :2], torch.zeros((self.num_envs, 1), device=self.device)], dim=-1) - self.torso_pos_w)
        target_right_footstep_b = quat_apply_inverse(self.torso_rot_w,
                                                     torch.cat([self.target_footstep_w[:, 1, :2], torch.zeros((self.num_envs, 1), device=self.device)], dim=-1) - self.torso_pos_w)
        self.target_footstep_b = torch.cat([target_left_footstep_b.unsqueeze(1), target_right_footstep_b.unsqueeze(1)], dim=1)

        # Command yaw (Body Frame)
        target_yaw_w = self.target_footstep_w[:, :, 2]
        target_forward_w = torch.stack([torch.cos(target_yaw_w), torch.sin(target_yaw_w), torch.zeros_like(target_yaw_w)], dim=-1)
        target_forward_b_left  = quat_apply_inverse(self.torso_rot_w, target_forward_w[:, 0, :3])
        target_forward_b_right = quat_apply_inverse(self.torso_rot_w, target_forward_w[:, 1, :3])
        target_forward_b = torch.cat([target_forward_b_left.unsqueeze(1), target_forward_b_right.unsqueeze(1)], dim=1)
        self.target_footstep_yaw_b = torch.atan2(target_forward_b[..., 1], target_forward_b[..., 0])

        # Feet Slide
        self.is_contacts = self.contact_sensors.data.net_forces_w_history[:, :, self.ankle_contact_roll_link_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        # Joint Limits
        self.out_of_limits_joint = -(self.joint_pos - self._robot.data.soft_joint_pos_limits[:, :, 0]).clip(max=0.0) + \
                                    (self.joint_pos - self._robot.data.soft_joint_pos_limits[:, :, 1]).clip(min=0.0)
        self.deviation_hip = self.joint_pos[:, self.hip_joint_ids] - self._robot.data.default_joint_pos[:, self.hip_joint_ids]
        self.deviation_arms = self.joint_pos[:, self.arm_joint_ids] - self._robot.data.default_joint_pos[:, self.arm_joint_ids]
        self.deviation_fingers = self.joint_pos[:, self.finger_joint_ids] - self._robot.data.default_joint_pos[:, self.finger_joint_ids]
        self.deviation_torso = self.joint_pos[:, self.torso_joint_ids] - self._robot.data.default_joint_pos[:, self.torso_joint_ids]


    def compute_target_footstep(self):
        update_ids = self.update_command_ids
        step_period = self.step_period[update_ids]
        command = self.command_inputs_w[update_ids]
        T = step_period * self.step_dt
        CoM = self.CoM[update_ids]
        z_c = CoM[:, 2]

        w0 = torch.sqrt(9.81 / z_c)
        dstep_length = torch.norm(command[:, :2], dim=1, keepdim=True).squeeze(-1) * T
        dstep_width = self.dstep_width[update_ids]
        theta = torch.atan2(command[:, 1:2], command[:, 0:1]).squeeze(-1)
        
        # Support Foot pos
        support_foot_pos = self.support_foot_pos[update_ids]

        # Relative CoM pos
        x_com_rel = CoM[:, 0] - support_foot_pos[:, 0]
        y_com_rel = CoM[:, 1] - support_foot_pos[:, 1]
        
        # Linear CoM velocity in world frame
        # vel_x_com = command[:, 0]
        # vel_y_com = command[:, 1]
        vel_x_com = self.torso_lin_vel_w[update_ids, 0]
        vel_y_com = self.torso_lin_vel_w[update_ids, 1]

        # Final step COM pos
        x_com_f  = x_com_rel * torch.cosh(w0 * T) + (vel_x_com / w0) * torch.sinh(w0 * T)
        vx_com_f = x_com_rel * w0 * torch.sinh(w0 * T) + vel_x_com * torch.cosh(w0 * T)
        y_com_f  = y_com_rel * torch.cosh(w0 * T) + (vel_y_com / w0) * torch.sinh(w0 * T)
        vy_com_f = y_com_rel * w0 * torch.sinh(w0 * T) + vel_y_com * torch.cosh(w0 * T)

        # Final ICP
        xi_f_x = (x_com_f + support_foot_pos[:, 0]) + vx_com_f / w0
        xi_f_y = (y_com_f + support_foot_pos[:, 1]) + vy_com_f / w0

        # s_d & w_d
        s_d = dstep_length
        w_d = dstep_width

        # Offset
        b_x = s_d / (torch.exp(w0 * T) - 1)
        b_y = w_d / (torch.exp(w0 * T) + 1)

        original_offset_x = -b_x
        original_offset_y = -torch.where(self.foot_on_swing[update_ids, 0] == 1, -b_y, b_y) # Left Swing : +b_y, Right Swing : -b_y

        offset_x = torch.cos(theta) * original_offset_x - torch.sin(theta) * original_offset_y
        offset_y = torch.sin(theta) * original_offset_x + torch.cos(theta) * original_offset_y

        # Target Foot Positions (World Frame)
        p_x = (xi_f_x + offset_x).reshape(-1, 1)
        p_y = (xi_f_y + offset_y).reshape(-1, 1)
        theta = theta.reshape(-1, 1)

        return torch.cat([p_x, p_y, theta], dim=-1)
    

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

@torch.jit.script
def adjust_foot_collision(collision_commands: torch.Tensor, 
                          collision_on_swing: torch.Tensor,
                          self_collision_threshold: float):
    """ Adjust foot collision by moving the foot to the nearest point on the boundary """
    collision_distance = torch.linalg.norm(collision_commands[:, 0] - collision_commands[:, 1], dim=1, ord=2, keepdim=True)
    adjust_step_commands = torch.clone(collision_commands)
    adjust_step_commands[collision_on_swing] = collision_commands[~collision_on_swing] + \
                                               self_collision_threshold * (collision_commands[collision_on_swing] - collision_commands[~collision_on_swing]) / collision_distance 
    return adjust_step_commands


@torch.jit.script
def resample_commands(step_period: torch.Tensor,
                      full_step_period: torch.Tensor,
                      dstep_width: torch.Tensor,
                      env_ids: torch.Tensor,
                      sim_dt: float,
                      time_period_min: float, time_period_max: float, 
                      dstep_width_min: float, dstep_width_max: float):
    """ 
    Randomly select foot step commands one/two steps ahead
    """
    period_min = int(time_period_min / sim_dt)
    period_max = int(time_period_max / sim_dt)

    step_period[env_ids] = torch.randint(low=period_min, 
                                         high=period_max,
                                         size=(len(env_ids),), device=step_period.device)
    
    full_step_period[env_ids] = 2 * step_period[env_ids]
    

    dstep_width[env_ids] = dstep_width_min + (dstep_width_max - dstep_width_min) * torch.rand(size=(len(env_ids),), device=dstep_width.device)

    return step_period, full_step_period, dstep_width
