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

from lib.domain_randomizer.commander import UniformVelocityCommand
from lib.env.G1.base.G1_base_env import G1BaseEnv
from lib.env.G1.safe.G1_safe_env_cfg import G1SafeEnvCfg

class G1SafeEnv(G1BaseEnv):
    cfg: G1SafeEnvCfg

    def __init__(self, cfg: G1SafeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Collision link id
        self.denied_collision_link_ids, _ = self.contact_sensors.find_bodies([r"torso_link",
                                                                              r"pelvis",
                                                                              r"waist_.*_link"])
        
        self.collision_upper_leg_link_ids, _ = self.contact_sensors.find_bodies(r".*_hip_(roll|pitch|yaw)_link")
        self.collision_lower_leg_link_ids, _ = self.contact_sensors.find_bodies(r".*_knee_link")

        self.collision_upper_arm_link_ids, _ = self.contact_sensors.find_bodies([r".*_shoulder_.*_link",
                                                                                 r".*_wrist_yaw_link"])
        self.collision_lower_arm_link_ids, _ = self.contact_sensors.find_bodies([r".*_elbow_link",
                                                                                 r".*_wrist_(roll|pitch)_link"])
        
        self.collision_foot_link_ids, _ = self.contact_sensors.find_bodies([r".*_ankle_.*_link"])

        # Link id
        self.denied_link_ids, _ = self._robot.find_bodies([r"torso_link",
                                                           r"pelvis",
                                                           r"waist_.*_link"])

        self.upper_leg_link_ids, _ = self._robot.find_bodies(r".*_hip_(roll|pitch|yaw)_link")
        self.lower_leg_link_ids, _ = self._robot.find_bodies(r".*_knee_link")

        self.upper_arm_link_ids, _ = self._robot.find_bodies([r".*_shoulder_.*_link",
                                                              r".*_wrist_yaw_link"])
        self.lower_arm_link_ids, _ = self._robot.find_bodies([r".*_elbow_link",
                                                              r".*_wrist_(roll|pitch)_link"])
        
        self.foot_link_ids, _ = self._robot.find_bodies([r".*_ankle_.*_link"])

        # Joint id
        self.torso_joint_ids, _ = self._robot.find_joints([r"waist_yaw_joint",])

        # Action Mapping
        self.mapping_sort_ids = torch.argsort(torch.tensor(self.total_arm_joint_ids + self.total_leg_joint_ids, device=self.device))
                    
        # Action scale factor
        if self.cfg.num_agents > 1:
            self.cfg.action_scale_factor["arm"][1] = self.total_arm_joint_ids
            self.cfg.action_scale_factor["leg"][1] = self.total_leg_joint_ids
        else:
            self.cfg.action_scale_factor = 1.0

        # Intermediate values
        self.root_pos_w          = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.root_rot_w          = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.root_lin_vel_w      = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.root_ang_vel_w      = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.root_lin_vel_b      = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.root_ang_vel_b      = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.CoM                 = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.root_heading        = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
        self.projected_gravity   = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.joint_pos           = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
        self.joint_vel           = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
        self.contact_force       = torch.zeros((self.num_envs, self.contact_sensors.num_bodies), dtype=torch.float, device=self.device)

        # Robot property
        self.robot_mass = self._robot.data.default_mass.to(self.device)
        self.total_mass = self._robot.data.default_mass.sum(dim=-1).to(self.device)

        self.denied_link_mass = self._robot.data.default_mass[:, self.denied_link_ids].to(self.device)

        self.upper_leg_mass = self._robot.data.default_mass[:, self.upper_leg_link_ids].to(self.device)
        self.lower_leg_mass = self._robot.data.default_mass[:, self.lower_leg_link_ids].to(self.device)

        self.upper_arm_mass = self._robot.data.default_mass[:, self.upper_arm_link_ids].to(self.device)
        self.lower_arm_mass = self._robot.data.default_mass[:, self.lower_arm_link_ids].to(self.device)
        
        self.foot_mass = self._robot.data.default_mass[:, self.foot_link_ids].to(self.device)

        # Foot states
        self.illegal_force = torch.zeros((self.num_envs, len(self.denied_collision_link_ids)), dtype=torch.float, device=self.device)

        # Geometry vector
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.left_vec = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # Regularization
        self.out_of_limits_joint    = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
        self.out_of_limits_torque   = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
        self.deviation_arms         = torch.zeros((self.num_envs, len(self.total_arm_joint_ids)), dtype=torch.float, device=self.device)
        self.deviation_legs         = torch.zeros((self.num_envs, len(self.total_leg_joint_ids)), dtype=torch.float, device=self.device)
        self.deviation_torso        = torch.zeros((self.num_envs, len(self.torso_joint_ids)), dtype=torch.float, device=self.device)

        # Prev value
        self.prev_contact_force = torch.zeros((self.num_envs, self.contact_sensors.num_bodies), dtype=torch.float, device=self.device)
        if self.cfg.num_agents > 1:
            # Multi Agent
            self.prev_actions = {
                    "leg": torch.zeros((self.num_envs, len(self.total_leg_joint_ids)), device=self.device),
                    "arm": torch.zeros((self.num_envs, len(self.total_arm_joint_ids)), device=self.device)}
        else:
            # Single Agent
            self.prev_actions = torch.zeros((self.num_envs, len(self._joint_dof_ids)), device=self.device)

        # Visualization
        debug_vis = self.num_envs <= 32
        self.set_debug_vis(debug_vis)


    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "torso_rotation_visualizer"):
                self.torso_rotation_visalizer = VisualizationMarkers(self.cfg.torso_rotation_visualizer_cfg)
            self.torso_rotation_visalizer.set_visibility(True)
        else:
            if hasattr(self, "torso_rotation_visualizer"):
                self.torso_rotation_visalizer.set_visibility(False)


    def _debug_vis_callback(self, event):
        if not self._robot.is_initialized:
            return
        # ============== Arrow ================ # 
        # Arrow: get marker location
        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        # =============== Torso ================
        torso_pos = self._robot.data.body_link_pos_w[:, self.torso_link_ids].reshape(-1, 3)
        torso_rot = self._robot.data.body_link_quat_w[:, self.torso_link_ids].reshape(-1, 4)

        # display markers
        self.torso_rotation_visalizer.visualize(translations=torso_pos,
                                                orientations=torso_rot)


    def _setup_scene(self):
        super()._setup_scene()
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = TerrainImporter(self.cfg.terrain)
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
                target=self._robot.data.default_joint_pos[:, self.total_arm_joint_ids] + arm_actions,
                joint_ids=self.total_arm_joint_ids
            )

            self._robot.set_joint_position_target(
                target=self._robot.data.default_joint_pos[:, self.total_leg_joint_ids] + leg_actions,
                joint_ids=self.total_leg_joint_ids
            )
        else:
            # Single Agent
            self._robot.set_joint_position_target(
                target=self._robot.data.default_joint_pos[:, self._joint_dof_ids] + self.actions,
                joint_ids=self._joint_dof_ids
            )


    def _get_observations(self) -> dict[str, torch.Tensor]:
        if self.cfg.num_agents > 1:
            # Multi Agent
            observations = {
                "arm": torch.cat(
                    [
                        self.root_lin_vel_b,                                # [E, 3]
                        self.root_ang_vel_b,                                # [E, 3]
                        self.projected_gravity,                             # [E, 3]
                        self.joint_pos[:, self.total_arm_joint_ids],        # [E, 17]
                        self.joint_vel[:, self.total_arm_joint_ids],        # [E, 17]
                        self.prev_actions["arm"],                           # [E, 17]
                    ],
                    dim=-1
                ),
                "leg": torch.cat(
                    [
                        self.root_lin_vel_b,                                # [E, 3]
                        self.root_ang_vel_b,                                # [E, 3]
                        self.projected_gravity,                             # [E, 3]
                        self.joint_pos[:, self.total_leg_joint_ids],        # [E, 12]
                        self.joint_vel[:, self.total_leg_joint_ids],        # [E, 12]
                        self.prev_actions["leg"],                           # [E, 12]
                    ],
                    dim=-1
                )
            }
        else:
            # Single Agent
            total_joint_ids = self.total_leg_joint_ids + self.total_arm_joint_ids
            observations = torch.cat(
                [
                    self.root_lin_vel_b,                                # [E, 3]
                    self.root_ang_vel_b,                                # [E, 3]
                    self.projected_gravity,                             # [E, 3]
                    self.joint_pos[:, total_joint_ids],                 # [E, 29]
                    self.joint_vel[:, total_joint_ids],                 # [E, 29]
                    self.prev_actions,                                  # [E, 29]
                ], dim=-1) 

        return observations


    def _get_states(self) -> dict[str, torch.Tensor]:
        if self.cfg.num_agents > 1:
            # Multi Agent
            total_joint_ids = self.total_leg_joint_ids + self.total_arm_joint_ids
            shared_states = torch.cat(
                [
                    self.root_pos_w[:, 2:3],                            # [E, 1]
                    self.root_lin_vel_b,                                # [E, 3]
                    self.root_ang_vel_b,                                # [E, 3]
                    self.projected_gravity,                             # [E, 3]
                    self.joint_pos[:, total_joint_ids],                 # [E, 29]
                    self.joint_vel[:, total_joint_ids],                 # [E, 29]
                    self.prev_actions["arm"],                           # [E, 17]
                    self.prev_actions["leg"],                           # [E, 12]
                ], dim=-1) 
            
            states = {
                "arm": shared_states,
                "leg": shared_states
            }
        else:
            # Single Agent (Syncronous Actor-Critic)
            states = None

        return states
    

    def _get_rewards(self) -> torch.Tensor:
        # Collision
        illegal_collision   = (self.illegal_force - self.denied_link_mass * 9.81).clip(min=0.0)
        upper_leg_collision = (self.contact_force[:, self.collision_upper_leg_link_ids] - self.upper_leg_mass * 9.81).clip(min=0.0)
        lower_leg_collision = (self.contact_force[:, self.collision_lower_leg_link_ids] - self.lower_leg_mass * 9.81).clip(min=0.0)
        upper_arm_collision = (self.contact_force[:, self.collision_upper_arm_link_ids] - self.upper_arm_mass * 9.81).clip(min=0.0)
        lower_arm_collision = (self.contact_force[:, self.collision_lower_arm_link_ids] - self.lower_arm_mass * 9.81).clip(min=0.0)

        max_illegal_collision   = torch.max(illegal_collision,   dim=-1).values
        max_upper_leg_collision = torch.max(upper_leg_collision, dim=-1).values
        max_lower_leg_collision = torch.max(lower_leg_collision, dim=-1).values
        max_upper_arm_collision = torch.max(upper_arm_collision, dim=-1).values
        max_lower_arm_collision = torch.max(lower_arm_collision, dim=-1).values

        num_collision_illegal   = torch.sum((illegal_collision > 1e-6).float(), dim=-1).clip(min=1.0)
        num_collision_upper_leg = torch.sum((upper_leg_collision > 1e-6).float(), dim=-1).clip(min=1.0)
        num_collision_lower_leg = torch.sum((lower_leg_collision > 1e-6).float(), dim=-1).clip(min=1.0)
        num_collision_upper_arm = torch.sum((upper_arm_collision > 1e-6).float(), dim=-1).clip(min=1.0)
        num_collision_lower_arm = torch.sum((lower_arm_collision > 1e-6).float(), dim=-1).clip(min=1.0)

        illegal_collision_penalty        = -(torch.sum(illegal_collision, dim=-1) + self.cfg.w_max_collision * max_illegal_collision) / num_collision_illegal
        prefer_collision_penalty_leg     = -(torch.sum(upper_leg_collision, dim=-1) + self.cfg.w_max_collision * max_upper_leg_collision) / num_collision_upper_leg
        not_prefer_collision_penalty_leg = -(torch.sum(lower_leg_collision, dim=-1) + self.cfg.w_max_collision * max_lower_leg_collision) / num_collision_lower_leg
        prefer_collision_penalty_arm     = -(torch.sum(lower_arm_collision, dim=-1) + self.cfg.w_max_collision * max_lower_arm_collision) / num_collision_lower_arm
        not_prefer_collision_penalty_arm = -(torch.sum(upper_arm_collision, dim=-1) + self.cfg.w_max_collision * max_upper_arm_collision) / num_collision_upper_arm

        # Termination
        # terminate_penalty = -self.reset_terminated.float()

        # Regularization
        joint_deviation_leg             = -torch.sum(torch.abs(self.deviation_legs), dim=-1)
        joint_limit_penalty_leg         = -torch.sum(self.out_of_limits_joint[:, self.total_leg_joint_ids], dim=1)
        joint_vel_penalty_leg           = -torch.sum(torch.square(self.joint_vel[:, self.total_leg_joint_ids]), dim=1)
        joint_torque_limit_penalty_leg  = -torch.sum(self.out_of_limits_torque[:, self.total_leg_joint_ids], dim=1)
        joint_torque_penalty_leg        = -torch.sum(torch.square(self._robot.data.applied_torque[:, self.total_leg_joint_ids]), dim=1)
        if self.cfg.num_agents > 1:
            action_rate_penalty_leg     = -torch.sum(torch.square(self.actions["leg"] - self.prev_actions["leg"]), dim=1)
        else:
            action_rate_penalty_leg     = -torch.sum(torch.square(self.actions[:, self.total_leg_joint_ids] - self.prev_actions[:, self.total_leg_joint_ids]), dim=1)

        joint_deviation_arm             = -torch.sum(torch.abs(self.deviation_arms), dim=-1)
        joint_limit_penalty_arm         = -torch.sum(self.out_of_limits_joint[:, self.total_arm_joint_ids], dim=1)
        joint_vel_penalty_arm           = -torch.sum(torch.square(self.joint_vel[:, self.total_arm_joint_ids]), dim=1)
        joint_torque_limit_penalty_arm  = -torch.sum(self.out_of_limits_torque[:, self.total_arm_joint_ids], dim=1)
        joint_torque_penalty_arm        = -torch.sum(torch.square(self._robot.data.applied_torque[:, self.total_arm_joint_ids]), dim=1)
        if self.cfg.num_agents > 1:
            action_rate_penalty_arm     = -torch.sum(torch.square(self.actions["arm"] - self.prev_actions["arm"]), dim=1)
        else:
            action_rate_penalty_arm     = -torch.sum(torch.square(self.actions[:, self.total_arm_joint_ids] - self.prev_actions[:, self.total_arm_joint_ids]), dim=1)

        # Reward summation
        common_rewards = self.cfg.w_termination * illegal_collision_penalty
        
        arm_rewards = common_rewards                                                     + \
                      self.cfg.w_deviation_arm        * joint_deviation_arm              + \
                      self.cfg.w_prefer_collision     * prefer_collision_penalty_arm     + \
                      self.cfg.w_not_prefer_collision * not_prefer_collision_penalty_arm + \
                      self.cfg.w_limits               * joint_limit_penalty_arm          + \
                      self.cfg.w_joint_torque_limit   * joint_torque_limit_penalty_arm   + \
                      self.cfg.w_joint_torque         * joint_torque_penalty_arm         + \
                      self.cfg.w_joint_vel            * joint_vel_penalty_arm            + \
                      self.cfg.w_action_rate          * action_rate_penalty_arm
        
        leg_rewards = common_rewards                                                     + \
                      self.cfg.w_deviation_leg        * joint_deviation_leg              + \
                      self.cfg.w_prefer_collision     * prefer_collision_penalty_leg     + \
                      self.cfg.w_not_prefer_collision * not_prefer_collision_penalty_leg + \
                      self.cfg.w_limits               * joint_limit_penalty_leg          + \
                      self.cfg.w_joint_torque_limit   * joint_torque_limit_penalty_leg   + \
                      self.cfg.w_joint_torque         * joint_torque_penalty_leg         + \
                      self.cfg.w_joint_vel            * joint_vel_penalty_leg            + \
                      self.cfg.w_action_rate          * action_rate_penalty_leg 

        # ============== Update prev value =============== #
        if self.cfg.num_agents > 1:
            # Multi Agent
            # Dictionary key order (alphabetical order in dictionary)
            rewards = torch.stack([arm_rewards, leg_rewards], dim=-1) # [E, 2]
            self.prev_actions = {k: v.clone() for k, v in self.actions.items()}
        else:
            # Single Agent
            rewards = common_rewards + (arm_rewards - common_rewards) + (leg_rewards - common_rewards)
            self.prev_actions = self.actions.clone()
        
        self.prev_contact_force = self.contact_force.clone()

        # Reward Info for logging
        self.extras["reward"] = {
            # ==========================================
            # Task Reward (+)
            # ==========================================

            # ==========================================
            # Task Penalty (-)
            # ==========================================
            "Task Penalty / Arm_Prefer_Collision"     : self.cfg.w_prefer_collision     * prefer_collision_penalty_arm,
            "Task Penalty / Arm_Not_Prefer_Collision" : self.cfg.w_not_prefer_collision * not_prefer_collision_penalty_arm,
            "Task Penalty / Arm_Joint_Limit"          : self.cfg.w_limits               * joint_limit_penalty_arm,
            "Task Penalty / Arm_Torque_Limit"         : self.cfg.w_joint_torque_limit   * joint_torque_limit_penalty_arm,
            "Task Penalty / Arm_Torque"               : self.cfg.w_joint_torque         * joint_torque_penalty_arm,
            "Task Penalty / Arm_Vel"                  : self.cfg.w_joint_vel            * joint_vel_penalty_arm,
            "Task Penalty / Arm_Action_Rate"          : self.cfg.w_action_rate          * action_rate_penalty_arm,

            "Task Penalty / Leg_Prefer_Collision"     : self.cfg.w_prefer_collision     * prefer_collision_penalty_leg,
            "Task Penalty / Leg_Not_Prefer_Collision" : self.cfg.w_not_prefer_collision * not_prefer_collision_penalty_leg,
            "Task Penalty / Leg_Joint_Limit"          : self.cfg.w_limits               * joint_limit_penalty_leg,
            "Task Penalty / Leg_Torque_Limit"         : self.cfg.w_joint_torque_limit   * joint_torque_limit_penalty_leg,
            "Task Penalty / Leg_Torque"               : self.cfg.w_joint_torque         * joint_torque_penalty_leg,
            "Task Penalty / Leg_Vel"                  : self.cfg.w_joint_vel            * joint_vel_penalty_leg,
            "Task Penalty / Leg_Action_Rate"          : self.cfg.w_action_rate          * action_rate_penalty_leg,
        }
        
        return rewards   


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # critical_contact_forces = self.illegal_force
        # died_collision   = torch.any(torch.norm(critical_contact_forces, dim=-1) > 1.0, dim=1)
        
        died = time_out
        time_out = time_out

        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        # Randomization by Event-based randomizer
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # term.last_prev_action and env._dataset_reset_prev_action share memory;
        # the dataset reset event has just written the sampled rows for env_ids.
        staged = getattr(self, "_dataset_reset_prev_action", None)
        if staged is not None:
            self.prev_actions["arm"][env_ids] = staged[env_ids][:, self.total_arm_joint_ids]
            self.prev_actions["leg"][env_ids] = staged[env_ids][:, self.total_leg_joint_ids]
        else:
            # Fallback: dataset reset event not registered for this env.
            self.prev_actions["arm"][env_ids] = 0.0
            self.prev_actions["leg"][env_ids] = 0.0

        self.prev_contact_force[env_ids] = 0.0

        self._compute_intermediate_values(env_ids)


    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        i = env_ids if env_ids is not None else self._robot._ALL_INDICES
        # Root Pose & Velocity
        self.root_pos_w[i], self.root_rot_w[i] = self._robot.data.root_pos_w[i], self._robot.data.root_quat_w[i]
        self.root_lin_vel_w[i], self.root_ang_vel_w[i] = self._robot.data.root_lin_vel_w[i], self._robot.data.root_ang_vel_w[i]
        self.root_lin_vel_b[i], self.root_ang_vel_b[i] = self._robot.data.root_lin_vel_b[i], self._robot.data.root_ang_vel_b[i]
        # Center of Mass (CoM)
        self.CoM[i] = (self._robot.data.body_link_pos_w[i] * self.robot_mass[i].unsqueeze(-1)).sum(dim=1) / self.total_mass[i].unsqueeze(-1)
        # Heading
        forward_root_w = quat_apply(self._robot.data.root_quat_w[i], self.forward_vec[i])
        self.root_heading[i] = torch.atan2(forward_root_w[:, 1], forward_root_w[:, 0]).unsqueeze(-1)
        # Attitude
        self.projected_gravity[i] = self._robot.data.projected_gravity_b[i]
        # Joint Angle & Velocity
        self.joint_pos[i], self.joint_vel[i] = self._robot.data.joint_pos[i], self._robot.data.joint_vel[i]
        # Ilegal force
        self.illegal_force[i] = torch.norm(self.contact_sensors.data.net_forces_w[i][:, self.denied_collision_link_ids], dim=-1)
        # Component-wise contact force
        self.contact_force[i] = torch.norm(self.contact_sensors.data.net_forces_w[i], dim=-1)
        # Regularization Parameter
        self.out_of_limits_joint[i]  = -(self.joint_pos[i] - self._robot.data.soft_joint_pos_limits[i, :, 0]).clip(max=0.0) + \
                                        (self.joint_pos[i] - self._robot.data.soft_joint_pos_limits[i, :, 1]).clip(min=0.0)
        self.out_of_limits_torque[i] = (torch.abs(self._robot.data.applied_torque[i]) - self._robot.data.joint_effort_limits[i] * self.cfg.soft_torque_limit).clip(min=0.0)
        self.deviation_arms[i]       = self.joint_pos[i][:, self.total_arm_joint_ids] - self._robot.data.default_joint_pos[i][:, self.total_arm_joint_ids]
        self.deviation_legs[i]       = self.joint_pos[i][:, self.total_leg_joint_ids] - self._robot.data.default_joint_pos[i][:, self.total_leg_joint_ids]
        self.deviation_torso[i]      = self.joint_pos[i][:, self.torso_joint_ids] - self._robot.data.default_joint_pos[i][:, self.torso_joint_ids]
    
    def _update_viz_data(self):
        mean_joint_deviation = torch.mean(torch.cat([self.deviation_arms, self.deviation_legs], dim=-1), dim=-1) # [E,]
        max_torque = torch.max(torch.abs(self._robot.data.applied_torque), dim=-1).values # [E,]
        max_contact_force = torch.max(self.contact_force, dim=-1).values # [E,]
        max_contact_impulse = max_contact_force * self.cfg.sim_dt # [E,]
        torso_collision = self.contact_force[:, self.denied_collision_link_ids[0]] # [E,]
        
        extras = copy.deepcopy(self.extras)
        extras["viz_data"]["max_torque"] = max_torque
        extras["viz_data"]["max_contact_impulse"] = max_contact_impulse
        extras["viz_data"]["max_contact_force"] = max_contact_force
        extras["viz_data"]["torso_contact_force"] = torso_collision
        extras["viz_data"]["mean_joint_deviation"] = mean_joint_deviation

        # print(f"{max_torque}")
        # print(f"{max_contact_force}")
        # print(f"{mean_joint_deviation}")

        return extras