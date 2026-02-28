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
from lib.env.G1.gait.G1_gait_env_cfg import G1GaitEnvCfg

class G1GaitEnv(G1BaseEnv):
    cfg: G1GaitEnvCfg

    def __init__(self, cfg: G1GaitEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Commands for reference signal
        self.commands = UniformVelocityCommand(self.cfg.commands, self._robot, self.device)

        # Joint Ids
        self.hip_xz_joint_ids, _ = self._robot.find_joints([".*_hip_yaw_joint", ".*_hip_roll_joint"])
        self.knee_joint_ids, _ = self._robot.find_joints([".*_knee_joint"])
        self.ankle_xy_joint_ids, _ = self._robot.find_joints([".*_ankle_pitch_joint", ".*_ankle_roll_joint"])
        self.hip_knee_all_joint_ids, _ = self._robot.find_joints([".*_hip_.*", ".*_knee_joint"])

        self.arm_all_joint_ids, _ = self._robot.find_joints([".*_shoulder_pitch_joint",
                                                             ".*_shoulder_roll_joint",
                                                             ".*_shoulder_yaw_joint",
                                                             ".*_elbow_pitch_joint",
                                                             ".*_elbow_roll_joint"])

        self.finger_all_joint_ids, _ = self._robot.find_joints([".*_five_joint",
                                                                ".*_three_joint",
                                                                ".*_six_joint",
                                                                ".*_four_joint",
                                                                ".*_zero_joint",
                                                                ".*_one_joint",
                                                                ".*_two_joint"])
        
        self.torso_joint_ids, _ = self._robot.find_joints("torso_joint")

        # Link ids
        self.torso_link_ids, _ = self._robot.find_bodies("torso_link")
        self.ankle_x_link_ids, _ = self._robot.find_bodies(".*_ankle_roll_link")

        # Contact Link ids
        self.torso_contact_link_ids, _ = self.contact_sensors.find_bodies("torso_link")
        self.ankle_contact_roll_link_ids, _ = self.contact_sensors.find_bodies(".*_ankle_roll_link")

        # Action scale factor
        self.cfg.action_scale_factor["arm"][1] = self.total_arm_joint_ids
        self.cfg.action_scale_factor["leg"][1] = self.total_leg_joint_ids

        # Robot property
        self.robot_mass = self._robot.data.default_mass.to(self.device)
        self.total_mass = self._robot.data.default_mass.sum(dim=-1).to(self.device)

        # Gait guidance
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) 
        self.update_phase_ids = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.command_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.update_command_ids = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.step_period = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.full_step_period = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Target Height
        self.z_c = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Geometry vector
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # Visualization
        debug_vis = self.num_envs <= 32
        self.set_debug_vis(debug_vis)


    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
            if not hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            if not hasattr(self, "torso_rotation_visualizer"):
                self.torso_rotation_visalizer = VisualizationMarkers(self.cfg.torso_rotation_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
            self.torso_rotation_visalizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
            if hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer.set_visibility(False)
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

        zeros = torch.zeros_like(self.command_heading)
        vel_des_arrow_quat = quat_from_euler_xyz(zeros, zeros, self.command_heading)
        vel_arrow_quat = quat_from_euler_xyz(zeros, zeros, self.root_heading)

        # =============== Root ================
        root_pos = self.root_pos_w[:, :3]
        root_rot = self.root_rot_w[:, :4]

        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)
        self.torso_rotation_visalizer.visualize(translations=root_pos,
                                                orientations=root_rot)

    def _setup_scene(self):
        super()._setup_scene()
        # sensor
        self.scene.sensors["contact_forces"] = ContactSensor(self.cfg.contact_forces)
        self.contact_sensors = self.scene.sensors["contact_forces"]
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
        

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Multi Agent
        observations = {
            "arm": torch.cat(
                [
                    self.root_pos_w[:, 2:3],
                    self.root_heading.unsqueeze(-1),
                    self.root_lin_vel_b,
                    self.root_ang_vel_b,
                    self.projected_gravity,
                    self.command_inputs_b,
                    self.phase_sin.unsqueeze(-1),
                    self.phase_cos.unsqueeze(-1),
                    self.joint_pos[:, self.total_arm_joint_ids],
                    self.joint_vel[:, self.total_arm_joint_ids]
                ],
                dim=-1
            ),
            "leg": torch.cat(
                [
                    self.root_pos_w[:, 2:3],
                    self.root_heading.unsqueeze(-1),
                    self.root_lin_vel_b,
                    self.root_ang_vel_b,
                    self.projected_gravity,
                    self.command_inputs_b,
                    self.phase_sin.unsqueeze(-1),
                    self.phase_cos.unsqueeze(-1),
                    self.joint_pos[:, self.total_leg_joint_ids],
                    self.joint_vel[:, self.total_leg_joint_ids]
                ],
                dim=-1
            )
        }

        return observations


    def _get_states(self) -> dict[str, torch.Tensor]:
        # Multi Agent
        shared_states = torch.cat(
            [
                self.root_pos_w[:, 2:3],
                self.root_heading.unsqueeze(-1),
                self.root_lin_vel_b,           # [E, 3]
                self.root_ang_vel_b,           # [E, 3]
                self.projected_gravity,         # [E, 3]
                self.command_inputs_b,          # [E, 3]
                self.phase_sin.unsqueeze(-1),   # [E, 1]
                self.phase_cos.unsqueeze(-1),   # [E, 1]
                self.joint_pos,                 # [E, 37]
                self.joint_vel,                 # [E, 37]
            ], dim=-1) 
        
        states = {
            "arm": shared_states,
            "leg": shared_states
        }

        return states


    def _get_rewards(self) -> torch.Tensor:
        # Tracking rewards
        lin_vel_error = torch.square(self.command_inputs_b[:, :2] - self.vel_yaw[:, :2])
        lin_vel_error *= 1 / torch.square(1 + torch.norm(self.command_inputs_b[:, :2], dim=-1)).unsqueeze(-1)
        lin_vel_error = torch.sum(lin_vel_error, dim=-1)
        heading_error = torch.abs(wrap_to_pi(self.command_heading - self.root_heading))
        height_error  = torch.square(self.root_pos_w[:, 2] - self.z_c)
        lin_vel_rewards = torch.exp(-lin_vel_error / 0.5**2)
        heading_rewards = torch.exp(-heading_error / 0.5**2)
        height_rewards  = torch.exp(-height_error / 0.5**2)
        # Attitute rewards 
        tilting = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        flat_rewards = torch.exp(-tilting / 0.2**2)
        # Gait rewards
        s = torch.sign(self.contact_schedule)   # right support (+), left support (-)
        diff = self.in_contact[:, 1].float() - self.in_contact[:, 0].float()  # right support (+), left support (-), double support (0)
        gait_reward = (diff * s)
        # Termination
        terminate_penalty = -self.reset_terminated.float()
        # Sliding
        slide_penalty = -torch.sum(self._robot.data.body_link_lin_vel_w[:, self.ankle_x_link_ids, :2].norm(dim=-1) * self.is_contacts, dim=1)
        # Regularization
        joint_deviation_penalty_hip_xz     = -torch.sum(torch.abs(self.deviation_hip_xz), dim=-1)
        joint_deviation_penalty_arms       = -torch.sum(torch.abs(self.deviation_arms), dim=1) 
        joint_deviation_penalty_fingers    = -torch.sum(torch.abs(self.deviation_fingers), dim=1)
        joint_deviation_penalty_torso      = -torch.sum(torch.abs(self.deviation_torso), dim=1)
        ang_vel_xy_penalty                 = -torch.sum(torch.square(self.root_ang_vel_b[:, :2]), dim=1)
        lin_vel_z_penalty                  = -torch.square(self.root_lin_vel_w[:, 2])

        joint_limit_penalty_leg         = -torch.sum(self.out_of_limits_joint[:, self.total_leg_joint_ids], dim=1)
        joint_torque_limit_penalty_leg  = -torch.sum(self.out_of_limits_torque[:, self.total_leg_joint_ids], dim=1)
        joint_torque_penalty_leg        = -torch.sum(torch.square(self._robot.data.applied_torque[:, self.total_leg_joint_ids]), dim=1)
        joint_vel_penalty_leg           = -torch.sum(torch.square(self.joint_vel[:, self.total_leg_joint_ids]), dim=1)
        action_rate_penalty_leg         = -torch.sum(torch.square(self.actions["leg"] - self.prev_actions["leg"]), dim=1)

        joint_limit_penalty_arm         = -torch.sum(self.out_of_limits_joint[:, self.total_arm_joint_ids], dim=1)
        joint_torque_limit_penalty_arm  = -torch.sum(self.out_of_limits_torque[:, self.total_arm_joint_ids], dim=1)
        joint_torque_penalty_arm        = -torch.sum(torch.square(self._robot.data.applied_torque[:, self.total_arm_joint_ids]), dim=1)
        joint_vel_penalty_arm           = -torch.sum(torch.square(self.joint_vel[:, self.total_arm_joint_ids]), dim=1)
        action_rate_penalty_arm         = -torch.sum(torch.square(self.actions["arm"] - self.prev_actions["arm"]), dim=1)
        # Multi Agent
        common_rewards = self.cfg.w_track_lin_vel * lin_vel_rewards     + \
                         self.cfg.w_track_heading * heading_rewards     + \
                         self.cfg.w_track_height  * height_rewards      + \
                         self.cfg.w_flat          * flat_rewards        + \
                         self.cfg.w_lin_vel_z     * lin_vel_z_penalty   + \
                         self.cfg.w_ang_vel_xy    * ang_vel_xy_penalty  + \
                         self.cfg.w_termination   * terminate_penalty
        
        arm_rewards = common_rewards                                                  + \
                      self.cfg.w_deviation_torso    * joint_deviation_penalty_torso   + \
                      self.cfg.w_deviation_arm      * joint_deviation_penalty_arms    + \
                      self.cfg.w_deviation_fingers  * joint_deviation_penalty_fingers + \
                      self.cfg.w_limits             * joint_limit_penalty_arm         + \
                      self.cfg.w_joint_torque_limit * joint_torque_limit_penalty_arm  + \
                      self.cfg.w_joint_torque       * joint_torque_penalty_arm        + \
                      self.cfg.w_joint_vel          * joint_vel_penalty_arm           + \
                      self.cfg.w_action_rate        * action_rate_penalty_arm
        
        leg_rewards = common_rewards                                                   + \
                      self.cfg.w_feet_slide          * slide_penalty                   + \
                      self.cfg.w_feet_gait           * gait_reward                     + \
                      self.cfg.w_deviation_hip       * joint_deviation_penalty_hip_xz  + \
                      self.cfg.w_limits              * joint_limit_penalty_leg         + \
                      self.cfg.w_joint_torque_limit  * joint_torque_limit_penalty_leg  + \
                      self.cfg.w_joint_torque        * joint_torque_penalty_leg        + \
                      self.cfg.w_joint_vel           * joint_vel_penalty_leg           + \
                      self.cfg.w_action_rate         * action_rate_penalty_leg 

        # Dictionary key order (alphabetical order in dictionary)
        rewards = torch.stack([arm_rewards, leg_rewards], dim=-1) # [E, 2]

        # Update Prev Actions (Multi Agent)
        self.prev_actions = {k: v.clone() for k, v in self.actions.items()}

        # Reward Info for logging
        self.extras["reward"] = {
            # ==========================================
            # Task Reward (+)
            # ==========================================
            "Task Reward / Common_Linear_Velocity" : self.cfg.w_track_lin_vel * lin_vel_rewards,
            "Task Reward / Common_Heading"         : self.cfg.w_track_heading * heading_rewards,
            "Task Reward / Common_Height"          : self.cfg.w_track_height  * height_rewards,
            "Task Reward / Common_Flat"            : self.cfg.w_flat          * flat_rewards,
            "Task Reward / Leg_Gait"               : self.cfg.w_feet_gait     * gait_reward,
            # ==========================================
            # Task Penalty (-)
            # ==========================================
            "Task Penalty / Common_Ang_Vel_XY"     : self.cfg.w_ang_vel_xy         * ang_vel_xy_penalty,
            "Task Penalty / Arm_Torso_Deviation"   : self.cfg.w_deviation_torso    * joint_deviation_penalty_torso,
            "Task Penalty / Arm_Deviation"         : self.cfg.w_deviation_arm      * joint_deviation_penalty_arms,
            "Task Penalty / Arm_Finger_Deviation"  : self.cfg.w_deviation_fingers  * joint_deviation_penalty_fingers,
            "Task Penalty / Arm_Joint_Limit"       : self.cfg.w_limits             * joint_limit_penalty_arm,
            "Task Penalty / Arm_Torque_Limit"      : self.cfg.w_joint_torque_limit * joint_torque_limit_penalty_arm,
            "Task Penalty / Arm_Torque"            : self.cfg.w_joint_torque       * joint_torque_penalty_arm,
            "Task Penalty / Arm_Vel"               : self.cfg.w_joint_vel          * joint_vel_penalty_arm,
            "Task Penalty / Arm_Action_Rate"       : self.cfg.w_action_rate        * action_rate_penalty_arm,
            "Task Penalty / Leg_Slide"             : self.cfg.w_feet_slide         * slide_penalty,
            "Task Penalty / Leg_Hip_XZ_Deviation"  : self.cfg.w_deviation_hip      * joint_deviation_penalty_hip_xz,
            "Task Penalty / Leg_Joint_Limit"       : self.cfg.w_limits             * joint_limit_penalty_leg,
            "Task Penalty / Leg_Torque_Limit"      : self.cfg.w_joint_torque_limit * joint_torque_limit_penalty_leg,
            "Task Penalty / Leg_Torque"            : self.cfg.w_joint_torque       * joint_torque_penalty_leg,
            "Task Penalty / Leg_Vel"               : self.cfg.w_joint_vel          * joint_vel_penalty_leg,
            "Task Penalty / Leg_Action_Rate"       : self.cfg.w_action_rate        * action_rate_penalty_leg,
        }
        
        return rewards  


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        torso_contact_forces = self.contact_sensors.data.net_forces_w_history[:, :, self.torso_contact_link_ids]
        projected_gravity_x = self.projected_gravity[:, 0]
        projected_gravity_y = self.projected_gravity[:, 1]

        died_fall   = torch.any(torch.max(torch.norm(torso_contact_forces, dim=-1), dim=1)[0] > 1.0, dim=1)
        died_fall_2 = torch.logical_or(torch.abs(projected_gravity_x) >= self.cfg.termination_gravity,
                                       torch.abs(projected_gravity_y) >= self.cfg.termination_gravity)
        died_ang = torch.norm(self.root_ang_vel_b, dim=-1) >= self.cfg.termination_ang_vel
        died = died_fall | died_fall_2 | died_ang
        return died, time_out
    

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        # Randomization by Event-based randomizer
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Gait guidance
        self.z_c[env_ids] = self.cfg.z_c_min + (self.cfg.z_c_max - self.cfg.z_c_min) * torch.rand(env_ids.shape[0], device=self.device)
        self.phase[env_ids] = 0
        self.phase_count[env_ids] = 0
        self.update_phase_ids[env_ids] = True
        self.command_count[env_ids] = 0
        self.update_command_ids[env_ids] = True

        if hasattr(self, "prev_actions"):
            self.prev_actions["leg"][env_ids] = 0.0
            self.prev_actions["arm"][env_ids] = 0.0
        else:
            self.prev_actions = {
                    "leg": torch.zeros((self.num_envs, len(self.total_leg_joint_ids)), device=self.device),
                    "arm": torch.zeros((self.num_envs, len(self.total_arm_joint_ids)), device=self.device)
            }
                


        # Command resampling
        self.commands.reset(env_ids)
        self.step_period, self.full_step_period= resample_commands(self.step_period,
                                                                   self.full_step_period,
                                                                   env_ids,
                                                                   self.step_dt,
                                                                   self.cfg.time_period_min, self.cfg.time_period_max)
        
        self._compute_intermediate_values(env_ids)


    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        # Root Pose & Velocity
        self.root_pos_w, self.root_rot_w = self._robot.data.root_pos_w, self._robot.data.root_quat_w
        self.root_lin_vel_w, self.root_ang_vel_w = self._robot.data.root_lin_vel_w, self._robot.data.root_ang_vel_w
        self.root_lin_vel_b, self.root_ang_vel_b = self._robot.data.root_lin_vel_b, self._robot.data.root_ang_vel_b
        self.vel_yaw = quat_apply_inverse(yaw_quat(self.root_rot_w), self.root_lin_vel_w[:, :3])
        # Center of Mass (CoM)
        self.CoM = (self._robot.data.body_link_pos_w * self.robot_mass.unsqueeze(-1)).sum(dim=1) / self.total_mass.unsqueeze(-1)
        # Heading
        forward_root_w = quat_apply(self._robot.data.root_quat_w, self.forward_vec)
        self.root_heading = torch.atan2(forward_root_w[:, 1], forward_root_w[:, 0])
        self.root_heading_sin = torch.sin(self.root_heading)
        self.root_heading_cos = torch.cos(self.root_heading)
        # Attitude
        self.projected_gravity = self._robot.data.projected_gravity_b
        # Joint Angle & Velocity
        self.joint_pos, self.joint_vel = self._robot.data.joint_pos, self._robot.data.joint_vel
        # Information related to Commands Tracking
        self.command_inputs_b = self.commands.command_b
        self.command_inputs_w = self.commands.command_w
        self.command_heading = self.commands.heading
        # Information related to Contact
        self.air_time = self.contact_sensors.data.current_air_time[:, self.ankle_contact_roll_link_ids] # [Left, Right]
        self.contact_time = self.contact_sensors.data.current_contact_time[:, self.ankle_contact_roll_link_ids] # [Left, Right]
        self.in_contact = self.contact_time > 0.0 # [E, 2 (Left, Right)]

        # Gait guidance
        if env_ids is not None:
            # Init env 와 progress env를 따로 처리
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_ids] = False
            if torch.any(mask):
                # Update progress env
                # Phase update signal
                self.phase[mask] += 1 / self.full_step_period[mask]
                self.phase_count[mask] += 1
                self.update_phase_ids[mask] = (self.phase_count[mask] >= self.full_step_period[mask])
                # Step command update signal
                self.command_count[mask] += 1
                self.update_command_ids[mask] = (self.command_count[mask] >= self.step_period[mask])
                # Schedule variables
                phase_update_mask = self.update_phase_ids & mask
                command_update_mask = self.update_command_ids & mask
                self.phase[phase_update_mask] = 0
                self.phase_count[phase_update_mask] = 0
                self.command_count[command_update_mask] = 0
        else:
            # Only progress env
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
        
        # Contact schedule
        self.contact_schedule = smooth_sqr_wave(self.phase)
        # Phase variable
        self.phase_sin = torch.sin(2*torch.pi*self.phase)
        self.phase_cos = torch.cos(2*torch.pi*self.phase)


        # Feet Slide
        self.is_contacts = self.contact_sensors.data.net_forces_w_history[:, :, self.ankle_contact_roll_link_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        # Regularization Parameter
        self.out_of_limits_joint = -(self.joint_pos - self._robot.data.soft_joint_pos_limits[:, :, 0]).clip(max=0.0) + \
                                    (self.joint_pos - self._robot.data.soft_joint_pos_limits[:, :, 1]).clip(min=0.0)
        self.out_of_limits_torque = torch.abs(self._robot.data.applied_torque - self._robot.data.joint_effort_limits * self.cfg.soft_torque_limit).clip(min=0.0)
        self.deviation_hip_xz = self.joint_pos[:, self.hip_xz_joint_ids] - self._robot.data.default_joint_pos[:, self.hip_xz_joint_ids]
        self.deviation_arms = self.joint_pos[:, self.arm_all_joint_ids] - self._robot.data.default_joint_pos[:, self.arm_all_joint_ids]
        self.deviation_fingers = self.joint_pos[:, self.finger_all_joint_ids] - self._robot.data.default_joint_pos[:, self.finger_all_joint_ids]
        self.deviation_torso = self.joint_pos[:, self.torso_joint_ids] - self._robot.data.default_joint_pos[:, self.torso_joint_ids]





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
def resample_commands(step_period: torch.Tensor,
                      full_step_period: torch.Tensor,
                      env_ids: torch.Tensor,
                      sim_dt: float,
                      time_period_min: float, time_period_max: float):
    """ 
    Randomly select foot step commands one/two steps ahead
    """
    period_min = int(time_period_min / sim_dt)
    period_max = int(time_period_max / sim_dt)

    if period_max == period_min:
        step_period[env_ids] = torch.tensor(period_min, dtype=torch.long,device=step_period.device).repeat(len(env_ids))
    else:
        step_period[env_ids] = torch.randint(low=period_min, 
                                            high=period_max,
                                            size=(len(env_ids),), device=step_period.device)
    
    full_step_period[env_ids] = 2 * step_period[env_ids]
    
    return step_period, full_step_period
