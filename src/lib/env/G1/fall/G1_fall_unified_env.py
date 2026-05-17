# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import gymnasium as gym
import copy

from lib.utils.space_utils import spec_to_gym_space

from lib.env.G1.fall.G1_fall_env_cfg import G1FallUnifiedPlayEnvCfg
from lib.env.G1.fall.G1_fall_env import G1FallEnv

class G1FallUnifiedEnv(G1FallEnv):
    cfg: G1FallUnifiedPlayEnvCfg

    def __init__(self, cfg: G1FallUnifiedPlayEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # link id
        self.denied_link_ids, _ = self._robot.find_bodies([r"torso_link",
                                                           r"pelvis",
                                                           r"waist_.*_link"])

        # Collision link
        self.denied_collision_link_ids, _ = self.contact_sensors.find_bodies([r"torso_link",
                                                                              r"pelvis",
                                                                              r"waist_.*_link"])
        
        # Foot collision link
        self.foot_collision_link_ids, _ = self.contact_sensors.find_bodies([r".*_ankle_(pitch|roll)_link"])


        # Illegal collision
        self.illegal_force = torch.zeros((self.num_envs, len(self.denied_collision_link_ids), 3), dtype=torch.float, device=self.device)
        self.contact_force = torch.zeros((self.num_envs, self.contact_sensors.num_bodies), dtype=torch.float, device=self.device)
        self.denied_link_mass = self._robot.data.default_mass[:, self.denied_link_ids].to(self.device)

        # Deviation
        self.deviation_legs = torch.zeros((self.num_envs, len(self.total_leg_joint_ids)), dtype=torch.float, device=self.device)

        # Safe Policy space config (single space only for action processing)
        self.safe_observation_space= spec_to_gym_space(self.cfg.safe_observation_space)
        self.safe_action_space = spec_to_gym_space(self.cfg.safe_action_space)
        if self.cfg.safe_state_space:
            self.safe_state_space = spec_to_gym_space(self.cfg.safe_state_space)
        else:
            self.safe_state_space = spec_to_gym_space(self.cfg.safe_observation_space)

        # Action buffer for handling two policies
        self.action_buffer = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)

    # Overriding to handle actions of multi-agent nominal policy and single-agent safety policy
    def _apply_action(self):
        if isinstance(self.actions, dict):
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
        elif isinstance(self.actions, torch.Tensor):
            # Single Agent
            self._robot.set_joint_position_target(
                target=self._robot.data.default_joint_pos[:, self._joint_dof_ids] + self.actions,
                joint_ids=self._joint_dof_ids
            )
        else:
            raise ValueError

    # Overriding to handle previous actions of multi-agent nominal policy and single-agent safety policy
    def _get_observations(self):
        # Action Processing
        if isinstance(self.prev_actions, dict):
            arm_actions = self.prev_actions["arm"]
            leg_actions = self.prev_actions["leg"]
        else:
            arm_actions = self.prev_actions[:, self.total_arm_joint_ids]
            leg_actions = self.prev_actions[:, self.total_leg_joint_ids]

        if self.cfg.num_agents > 1:
            # Multi Agent
            observations = {
                "arm": torch.cat(
                    [
                        self.root_lin_vel_b,                                # [E, 3]
                        self.root_ang_vel_b,                                # [E, 3]
                        self.projected_gravity,                             # [E, 3]
                        self.command_inputs_b,                              # [E, 3]
                        self.phase_sin.unsqueeze(-1),                       # [E, 1]
                        self.phase_cos.unsqueeze(-1),                       # [E, 1]
                        self.joint_pos[:, self.total_arm_joint_ids],        # [E, 17]
                        self.joint_vel[:, self.total_arm_joint_ids],        # [E, 17]
                        arm_actions
                    ],
                    dim=-1
                ),
                "leg": torch.cat(
                    [
                        self.root_lin_vel_b,                                # [E, 3]
                        self.root_ang_vel_b,                                # [E, 3]
                        self.projected_gravity,                             # [E, 3]
                        self.command_inputs_b,                              # [E, 3]
                        self.phase_sin.unsqueeze(-1),                       # [E, 1]
                        self.phase_cos.unsqueeze(-1),                       # [E, 1]
                        self.joint_pos[:, self.total_leg_joint_ids],        # [E, 12]
                        self.joint_vel[:, self.total_leg_joint_ids],        # [E, 12]
                        leg_actions
                    ],
                    dim=-1
                )
            }
        else:
            # Single Agent
            action_buffer = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
            action_buffer[:, self.total_arm_joint_ids] = arm_actions
            action_buffer[:, self.total_leg_joint_ids] = leg_actions
            total_joint_ids = self.total_leg_joint_ids + self.total_arm_joint_ids
            observations = torch.cat(
                [
                    self.root_lin_vel_b,                                # [E, 3]
                    self.root_ang_vel_b,                                # [E, 3]
                    self.projected_gravity,                             # [E, 3]
                    self.command_inputs_b,                              # [E, 3]
                    self.phase_sin.unsqueeze(-1),                       # [E, 1]
                    self.phase_cos.unsqueeze(-1),                       # [E, 1]
                    self.joint_pos[:, total_joint_ids],                 # [E, 29]
                    self.joint_vel[:, total_joint_ids],                 # [E, 29]
                    action_buffer                                       # [E, 29]
                ], dim=-1) 

        return observations


    # Overriding to add Safety policy information
    def _get_states(self) -> dict[str, torch.Tensor]:
        # Action Processing
        if isinstance(self.prev_actions, dict):
            arm_actions = self.prev_actions["arm"]
            leg_actions = self.prev_actions["leg"]
        else:
            arm_actions = self.prev_actions[:, self.total_arm_joint_ids]
            leg_actions = self.prev_actions[:, self.total_leg_joint_ids]

        # Nominal Policy
        if self.cfg.num_agents > 1:
           # Multi Agent
            total_joint_ids = self.total_leg_joint_ids + self.total_arm_joint_ids
            shared_states = torch.cat(
                [
                    self.root_pos_w[:, 2:3],                            # [E, 1]
                    self.root_lin_vel_b,                                # [E, 3]
                    self.root_ang_vel_b,                                # [E, 3]
                    self.projected_gravity,                             # [E, 3]
                    self.command_inputs_b,                              # [E, 3]
                    self.phase_sin.unsqueeze(-1),                       # [E, 1]
                    self.phase_cos.unsqueeze(-1),                       # [E, 1]
                    self.joint_pos[:, total_joint_ids],                 # [E, 29]
                    self.joint_vel[:, total_joint_ids],                 # [E, 29]
                    leg_actions,
                    arm_actions,
                ], dim=-1) 
            
            states = {
                "arm": shared_states,
                "leg": shared_states
            }
        else:
            # Single Agent
            states = None

        # Reach-Avoid information
        self.extras["ra_states"] = torch.cat([self.root_ang_vel_b,                                # [E, 3]
                                              self.projected_gravity,                             # [E, 3]
                                              self.dist_from_icp_to_stance,                       # [E, 1]
                                              self.phase.unsqueeze(-1),                           # [E, 1]
                                              self.root_state_buffer.reshape(self.num_envs, -1)   # [E, body_hist_length*8]
                                            ], dim=-1)

        base_tilt = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        # is_fall = ((self.capturable_boundary - self.dist_from_icp_to_stance) <= 0).float().squeeze(-1)

        self.extras["l_values"] = torch.tanh(torch.log(base_tilt / self.cfg.target_set_threshold**2))
        self.extras["g_values"] = 2 * self.reset_terminated.float() - 1
        # self.extras["g_values"] = 2 * is_fall - 1

        # SafeFall baseline observation (gravity_xy, root_ang_vel, joint_pos, joint_vel)
        total_joint_ids = self.total_leg_joint_ids + self.total_arm_joint_ids
        self.extras["safe_fall_obs"] = torch.cat([
            self.projected_gravity[:, :2],          # [E, 2]
            self.root_ang_vel_b,                    # [E, 3]
            self.joint_pos[:, total_joint_ids],     # [E, 29]
            self.joint_vel[:, total_joint_ids],     # [E, 29]
        ], dim=-1)                                  # [E, 63]

        # Safe Policy
        if self.cfg.num_safe_agents > 1:
            # Multi Agent
            safe_obs_dict = {
                "arm": torch.cat(
                    [
                        self.root_lin_vel_b,                                # [E, 3]
                        self.root_ang_vel_b,                                # [E, 3]
                        self.projected_gravity,                             # [E, 3]
                        self.joint_pos[:, self.total_arm_joint_ids],        # [E, 17]
                        self.joint_vel[:, self.total_arm_joint_ids],        # [E, 17]
                        arm_actions                                         # [E, 17]
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
                        leg_actions                                         # [E, 12]
                    ],
                    dim=-1
                )
            }
            self.extras["safe_observations"] = torch.cat([safe_obs_dict["arm"], safe_obs_dict["leg"]], dim=-1)

        else:
            # Single Agent
            action_buffer = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float, device=self.device)
            action_buffer[:, self.total_arm_joint_ids] = arm_actions
            action_buffer[:, self.total_leg_joint_ids] = leg_actions
            total_joint_ids = self.total_leg_joint_ids + self.total_arm_joint_ids
            self.extras["safe_observations"] = torch.cat(
                [
                    self.root_lin_vel_b,                                # [E, 3]
                    self.root_ang_vel_b,                                # [E, 3]
                    self.projected_gravity,                             # [E, 3]
                    self.joint_pos[:, total_joint_ids],                 # [E, 29]
                    self.joint_vel[:, total_joint_ids],                 # [E, 29]
                    action_buffer,                                      # [E, 29]
                ], dim=-1
            )

        return states 
    
    # Overriding to handle previous actions of multi-agent nominal policy and single-agent safety policy
    def _get_rewards(self):
        if isinstance(self.actions, dict):
            self.prev_actions = {k: v.clone() for k, v in self.actions.items()}
        elif isinstance(self.actions, torch.Tensor):
            self.prev_actions = self.actions.clone()
        else:
            raise ValueError

        return torch.zeros((self.num_envs, self.cfg.num_agents), dtype=torch.float, device=self.device)

    # Overriding to add Safety policy information
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Falling and Collision
        z_c = self.CoM[:, 2]
        died_fall   = z_c <= self.cfg.termination_height
        critical_contact_forces = (torch.norm(self.illegal_force, dim=-1) - self.denied_link_mass).clip(min=0.0)
        # Only Dynamic Collision & Delete Additional Push Effect 
        died_collision   = torch.any(critical_contact_forces > 10.0, dim=1)
        
        died = died_collision & died_fall
        time_out = time_out

        return died, time_out

    def _compute_intermediate_values(self, env_ids = None):
        super()._compute_intermediate_values(env_ids)
        i = env_ids if env_ids is not None else self._robot._ALL_INDICES
        # Component-wise contact force
        self.contact_force[i]  = torch.norm(self.contact_sensors.data.net_forces_w[i], dim=-1)
        self.deviation_legs[i] = self.joint_pos[i][:, self.total_leg_joint_ids] - self._robot.data.default_joint_pos[i][:, self.total_leg_joint_ids]
    
    def _update_viz_data(self):
        valid_mask = torch.ones((self.num_envs, self.contact_sensors.num_bodies), dtype=torch.bool, device=self.device)
        valid_mask[:, self.foot_collision_link_ids] = False
        mean_joint_deviation = torch.mean(torch.cat([self.deviation_arms, self.deviation_legs], dim=-1), dim=-1) # [E,]
        max_torque = torch.max(torch.abs(self._robot.data.applied_torque), dim=-1).values # [E,]
        max_valid_contact_force = torch.max(self.contact_force[valid_mask], dim=-1).values # [E,]
        max_contact_impulse = max_valid_contact_force * self.cfg.sim_dt # [E,]
        torso_collision = (self.contact_force[:, self.denied_collision_link_ids[0]] - 9.81 * self.denied_link_mass[0]).clip(min=0.0) # [E,]
        
        extras = copy.deepcopy(self.extras)
        extras["viz_data"]["max_torque"] = max_torque
        extras["viz_data"]["max_contact_impulse"] = max_contact_impulse
        extras["viz_data"]["max_contact_force"] = max_valid_contact_force
        extras["viz_data"]["torso_contact_force"] = torso_collision
        extras["viz_data"]["mean_joint_deviation"] = mean_joint_deviation

        return extras