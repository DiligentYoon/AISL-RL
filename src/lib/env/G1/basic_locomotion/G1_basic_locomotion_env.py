# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter
from isaaclab.markers import VisualizationMarkers

from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.managers import SceneEntityCfg

from lib.domain_randomizer.commander import UniformVelocityCommand
from lib.env.G1.base.G1_base_env import G1BaseEnv
from lib.env.G1.basic_locomotion.G1_basic_locomotion_env_cfg import G1BasicLocomotionEnvCfg


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class G1BasicLocomotionEnv(G1BaseEnv):
    cfg: G1BasicLocomotionEnvCfg

    def __init__(self, cfg: G1BasicLocomotionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # default data
        self.action_scale = self.cfg.action_scale
        self._joint_dof_ids, _ = self._robot.find_joints(".*")

        # Commands for reference generator
        self.commands = UniformVelocityCommand(self.cfg.commands, self._robot, self.device)

        # Joint Ids
        self.total_leg_joint_ids, _ = self._robot.find_joints([r".*_hip_(pitch|roll|yaw)_joint",
                                                               r".*_knee_joint",
                                                               r".*_ankle_(pitch|roll)_joint"])
        
        self.total_arm_joint_ids, _ = self._robot.find_joints([r"torso_joint", 
                                                               r".*_shoulder_(pitch|roll|yaw)_joint",
                                                               r".*_elbow_(pitch|roll)_joint",
                                                               r".*_(zero|one|two|three|four|five|six)_joint"])

        self.ankle_joint_ids, _ = self._robot.find_joints([".*_ankle_pitch_joint", 
                                                           ".*ankle_roll_joint"])
        
        self.hip_joint_ids, _ = self._robot.find_joints([".*_hip_yaw_joint", 
                                                         ".*_hip_roll_joint"])
        
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
        self.leg_joint_limits = self._robot.data.joint_pos_limits[:, self.total_leg_joint_ids]
        self.arm_joint_limits = self._robot.data.joint_pos_limits[:, self.total_arm_joint_ids]
        
        # Link ids
        self.torso_link_ids, _ = self._robot.find_bodies("torso_link")
        self.torso_contact_link_ids, _ = self.contact_sensors.find_bodies("torso_link")
        self.ankle_roll_link_ids, _ = self._robot.find_bodies(".*_ankle_roll_link")

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
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self._robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self.commands._resolve_xy_velocity_to_arrow(scale=self.goal_vel_visualizer.cfg.markers["arrow"].scale,
                                                                                              xy_velocity=self.command_inputs[:, :2])
        vel_arrow_scale, vel_arrow_quat = self.commands._resolve_xy_velocity_to_arrow(scale=self.current_vel_visualizer.cfg.markers["arrow"].scale,
                                                                                      xy_velocity=self._robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)


    def _setup_scene(self):
        super()._setup_scene()
        # sensor
        self.scene.sensors["contact_forces"] = ContactSensor(self.cfg.contact_forces)
        # self.scene.sensors["height_scanner"] = RayCaster(self.cfg.height_scanner)
        self.contact_sensors = self.scene.sensors["contact_forces"]
        # self.height_scanner = self.scene.sensors["height_scanner"]
        # self.height_scanner.update_period = self.cfg.decimation * self.cfg.sim_dt
        self.contact_sensors.update_period = self.cfg.sim_dt
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
            leg_actions = self.action_scale * self.actions["leg"]
            arm_actions = self.action_scale * self.actions["arm"]

            self._robot.set_joint_position_target(
                target=torch.clamp(self._robot.data.joint_pos[:, self.total_leg_joint_ids] + leg_actions,
                                min=self.leg_joint_limits[:, :, 0],
                                max=self.leg_joint_limits[:, :, 1]),
                joint_ids=self.total_leg_joint_ids)
            
            self._robot.set_joint_position_target(
                target=torch.clamp(self._robot.data.joint_pos[:, self.total_arm_joint_ids] + arm_actions,
                                min=self.arm_joint_limits[:, :, 0],
                                max=self.arm_joint_limits[:, :, 1]),
                joint_ids=self.total_arm_joint_ids
            )
        else:
            # Single Agent
            # self._robot.set_joint_position_target(
            #     target=torch.clamp(self._robot.data.joint_pos[:, self._joint_dof_ids] + self.action_scale * self.actions,
            #                        min=self._robot.data.joint_pos_limits[:, self._joint_dof_ids, 0],
            #                        max=self._robot.data.joint_pos_limits[:, self._joint_dof_ids, 1]),
            #     joint_ids=self._joint_dof_ids
            # )
            self._robot.set_joint_position_target(
                target=torch.clamp(self._robot.data.default_joint_pos[:, self._joint_dof_ids] + self.action_scale * self.actions,
                                   min=self._robot.data.joint_pos_limits[:, self._joint_dof_ids, 0],
                                   max=self._robot.data.joint_pos_limits[:, self._joint_dof_ids, 1]),
                joint_ids=self._joint_dof_ids
            )


    def _get_observations(self) -> dict[str, torch.Tensor] | torch.Tensor:
        if self.cfg.num_agents > 1:
            # Multi Agent
            observations = {
                "leg": torch.cat(
                    [
                        self.torso_lin_vel_b,                        # [E, 3]
                        self.torso_ang_vel_b,                        # [E, 3]    
                        self.projected_gravity,                      # [E, 3]
                        self.joint_pos[:, self.total_leg_joint_ids], # [E, 12]
                        self.joint_vel[:, self.total_leg_joint_ids], # [E, 12]
                        self.actions["leg"]                          # [E, 12]
                    ],
                    dim=-1
                ),
                "arm": torch.cat(
                    [
                        self.torso_lin_vel_b,                        # [E, 3]
                        self.torso_ang_vel_b,                        # [E, 3]
                        self.projected_gravity,                      # [E, 3]
                        self.joint_pos[:, self.total_arm_joint_ids], # [E, 25]
                        self.joint_vel[:, self.total_arm_joint_ids], # [E, 25]
                        self.actions["arm"]                          # [E, 25]
                    ],
                    dim=-1
                ),
            }
        else:
            observations = torch.cat(
                [
                    self.torso_lin_vel_b,           # [E, 3]
                    self.torso_ang_vel_b,           # [E, 3]
                    self.projected_gravity,         # [E, 3]
                    self.command_inputs,            # [E, 3]
                    self.joint_pos_rel,             # [E, 37]
                    self.joint_vel_rel,             # [E, 37]
                    self.actions                    # [E, 37]  
                ],
                dim=-1
            )

        return observations

    def _get_states(self) -> dict[str, torch.Tensor] | torch.Tensor:
        if self.cfg.num_agents > 1:
            # Multi Agent
            shared_states = torch.cat(
                [
                    self.torso_lin_vel_b,           # [E, 3]
                    self.torso_ang_vel_b,           # [E, 3]
                    self.projected_gravity,         # [E, 3]
                    self.command_inputs,            # [E, 3]
                    self.joint_pos_rel,             # [E, 37]
                    self.joint_vel_rel,             # [E, 37]
                    self.actions["leg"],            # [E, 12]
                    self.actions["arm"],            # [E, 25]
                ], dim=-1) 
            
            states = {
                "leg": shared_states,
                "arm": shared_states
            }
        else:
            # Single Agent (Syncronous Actor Critic)
            states = None
            
        return states


    def _get_rewards(self) -> torch.Tensor:
        # Tracking Rewards (Torso)
        lin_vel_error = torch.sum(torch.square(self.command_inputs[:, :2] - self.vel_yaw[:, :2]), dim=1)
        ang_vel_error = torch.square(self.command_inputs[:, 2] - self.torso_ang_vel_w[:, 2])
        lin_vel_rewards = torch.exp(-lin_vel_error / 0.5**2)
        ang_vel_rewards = torch.exp(-ang_vel_error / 0.5**2)

        # Gait Rewards (Leg)
        in_contact = self.contact_time > 0.0
        in_mode_time = torch.where(in_contact, self.contact_time, self.air_time)
        single_stance = torch.sum(in_contact.int(), dim=1) == 1
        gait_reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
        gait_reward = torch.clamp(gait_reward, max=self.cfg.feet_air_time_threshold)
        # no reward for zero command
        gait_reward *= torch.norm(self.command_inputs[:, :2], dim=1) > 0.1

        # Sliding Penalty (Leg)
        slide_penalty = -torch.sum(self._robot.data.body_link_lin_vel_w[:, self.ankle_roll_link_ids, :2].norm(dim=-1) * self.is_contacts, dim=1)

        # Joint Pos Limits & Deviation Penalty
        joint_pos_penalty_ankle   = -torch.sum(self.out_of_limits_ankle, dim=1)          # Leg
        joint_pos_penalty_hip     = -torch.sum(torch.abs(self.deviation_hip), dim=1)     # Leg
        joint_pos_penalty_arms    = -torch.sum(torch.abs(self.deviation_arms), dim=1)    # Arm
        joint_pos_penalty_fingers = -torch.sum(torch.abs(self.deviation_fingers), dim=1) # Arm
        joint_pos_penalty_torso   = -torch.sum(torch.abs(self.deviation_torso), dim=1)   # Torso

        # Termination (Torso)
        terminate_penalty = -self.reset_terminated.float()

        if self.cfg.num_agents > 1:
            # Multi Agent
            common_rewards = self.cfg.w_track_lin_vel * lin_vel_rewards + \
                             self.cfg.w_track_ang_vel * ang_vel_rewards + \
                             self.cfg.w_limits_torso  * joint_pos_penalty_torso + \
                             self.cfg.w_termination   * terminate_penalty

            leg_rewards = common_rewards + \
                        self.cfg.w_feet_air_time * gait_reward + \
                        self.cfg.w_feet_slide * slide_penalty + \
                        self.cfg.w_limits_ankle * joint_pos_penalty_ankle + \
                        self.cfg.w_limits_hip * joint_pos_penalty_hip

            arm_rewards = common_rewards + \
                        self.cfg.w_limits_arm * joint_pos_penalty_arms + \
                        self.cfg.w_limits_fingers * joint_pos_penalty_fingers

            rewards = torch.stack([leg_rewards, arm_rewards], dim=-1) # [E, 2]
        else:
            # Single Agent
            rewards = self.cfg.w_track_lin_vel * lin_vel_rewards + \
                      self.cfg.w_track_ang_vel * ang_vel_rewards + \
                      self.cfg.w_feet_air_time * gait_reward + \
                      self.cfg.w_feet_slide * slide_penalty + \
                      self.cfg.w_limits_ankle * joint_pos_penalty_ankle + \
                      self.cfg.w_limits_hip * joint_pos_penalty_hip + \
                      self.cfg.w_limits_arm * joint_pos_penalty_arms + \
                      self.cfg.w_limits_fingers * joint_pos_penalty_fingers + \
                      self.cfg.w_limits_torso  * joint_pos_penalty_torso + \
                      self.cfg.w_termination   * terminate_penalty      # [E, 1]
        
        return rewards


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        torso_contact_forces = self.contact_sensors.data.net_forces_w_history[:, :, self.torso_contact_link_ids]
        died = torch.any(torch.max(torch.norm(torso_contact_forces, dim=-1), dim=1)[0] > 1.0, dim=1)
        # died = self.torso_pos_w[:, 2] < self.cfg.termination_height
        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        # Randomization by Event-based randomizer
        self._robot.reset(env_ids)
        self.commands.reset(env_ids)
        super()._reset_idx(env_ids)

        self._compute_intermediate_values()


    def _compute_intermediate_values(self):
        # Root Pose & Velocity
        self.torso_pos_w, self.torso_rot_w = self._robot.data.root_pos_w, self._robot.data.root_quat_w
        self.torso_lin_vel_w, self.torso_ang_vel_w = self._robot.data.root_lin_vel_w, self._robot.data.root_ang_vel_w
        self.torso_lin_vel_b, self.torso_ang_vel_b = self._robot.data.root_lin_vel_b, self._robot.data.root_ang_vel_b
        # Attitude
        self.projected_gravity = self._robot.data.projected_gravity_b
        # Joint Angle & Velocity
        self.joint_pos, self.joint_vel = self._robot.data.joint_pos, self._robot.data.joint_vel
        self.joint_pos_rel, self.joint_vel_rel = self.joint_pos - self._robot.data.default_joint_pos, self.joint_vel - self._robot.data.default_joint_vel
        # Height (For rough terrain)
        # self.height_scan = (self.height_scanner.data.pos_w[:, 2].unsqueeze(1) - self.height_scanner.data.ray_hits_w[..., 2] - 0.5).clip(min=-1.0, max=1.0)
        # Information related to Commands Tracking
        self.command_inputs = self.commands.command
        self.vel_yaw = quat_apply_inverse(yaw_quat(self.torso_rot_w), self.torso_lin_vel_w)
        # Information related to Gait Phase
        self.air_time = self.contact_sensors.data.current_air_time[:, self.ankle_roll_link_ids]
        self.contact_time = self.contact_sensors.data.current_contact_time[:, self.ankle_roll_link_ids]
        # Feet Slide
        self.is_contacts = self.contact_sensors.data.net_forces_w_history[:, :, self.ankle_roll_link_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        # Joint Limits
        self.out_of_limits_ankle = -(self.joint_pos[:, self.ankle_joint_ids] - self._robot.data.soft_joint_pos_limits[:, self.ankle_joint_ids, 0]).clip(max=0.0) + \
                                    (self.joint_pos[:, self.ankle_joint_ids] - self._robot.data.soft_joint_pos_limits[:, self.ankle_joint_ids, 1]).clip(min=0.0)
        self.deviation_hip = self.joint_pos[:, self.hip_joint_ids] - self._robot.data.default_joint_pos[:, self.hip_joint_ids]
        self.deviation_arms = self.joint_pos[:, self.arm_joint_ids] - self._robot.data.default_joint_pos[:, self.arm_joint_ids]
        self.deviation_fingers = self.joint_pos[:, self.finger_joint_ids] - self._robot.data.default_joint_pos[:, self.finger_joint_ids]
        self.deviation_torso = self.joint_pos[:, self.torso_joint_ids] - self._robot.data.default_joint_pos[:, self.torso_joint_ids]
        