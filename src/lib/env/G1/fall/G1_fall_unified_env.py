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

        # Collision link
        self.denied_collision_link_ids, _ = self.contact_sensors.find_bodies([r"torso_link",
                                                                              r"pelvis",
                                                                              r"waist_.*_link"])
        # Illegal collision
        self.illegal_force = torch.zeros((self.num_envs, len(self.denied_collision_link_ids), 3), dtype=torch.float, device=self.device)
        self.contact_force = torch.zeros((self.num_envs, self.contact_sensors.num_bodies), dtype=torch.float, device=self.device)

        # Deviation
        self.deviation_legs = torch.zeros((self.num_envs, len(self.total_leg_joint_ids)), dtype=torch.float, device=self.device)

        # Safe Policy space config (single space only for action processing)
        self.safe_observation_space= spec_to_gym_space(self.cfg.safe_observation_space)
        self.safe_action_space = spec_to_gym_space(self.cfg.safe_action_space)
        if self.cfg.safe_state_space:
            self.safe_state_space = spec_to_gym_space(self.cfg.safe_state_space)
        else:
            self.safe_state_space = spec_to_gym_space(self.cfg.safe_observation_space)

    # Overriding to add Safety policy information
    def _get_states(self) -> dict[str, torch.Tensor]:
        states = super()._get_states()

        # Safe Policy information
        if self.cfg.num_agents > 1:
            # Multi Agent
            safe_obs_dict = {
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
            self.extras["safe_observations"] = torch.cat([safe_obs_dict["arm"], safe_obs_dict["leg"]], dim=-1)

        else:
            # Single Agent
            total_joint_ids = self.total_leg_joint_ids + self.total_arm_joint_ids
            self.extras["safe_observations"] = torch.cat(
                [
                    self.root_lin_vel_b,                                # [E, 3]
                    self.root_ang_vel_b,                                # [E, 3]
                    self.projected_gravity,                             # [E, 3]
                    self.joint_pos[:, total_joint_ids],                 # [E, 29]
                    self.joint_vel[:, total_joint_ids],                 # [E, 29]
                    self.prev_actions,                                  # [E, 29]
                ]
            )

        return states 

    # Overriding to add Safety policy information
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        critical_contact_forces = self.illegal_force
        died_collision   = torch.any(torch.norm(critical_contact_forces, dim=-1) > 1.0, dim=1)
        
        died = died_collision
        time_out = time_out

        return died, time_out

    def _compute_intermediate_values(self, env_ids = None):
        super()._compute_intermediate_values(env_ids)
        i = env_ids if env_ids is not None else self._robot._ALL_INDICES
        # Component-wise contact force
        self.contact_force[i]  = torch.norm(self.contact_sensors.data.net_forces_w[i], dim=-1)
        self.deviation_legs[i] = self.joint_pos[i][:, self.total_leg_joint_ids] - self._robot.data.default_joint_pos[i][:, self.total_leg_joint_ids]
    
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

        return extras