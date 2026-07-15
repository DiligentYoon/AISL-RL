# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import copy

from lib.env.G1.fall.G1_fall_env_cfg import G1FallEnvCfg
from lib.env.G1.recovery.G1_recovery_env import G1RecoveryEnv

class G1FallEnv(G1RecoveryEnv):
    cfg: G1FallEnvCfg

    def __init__(self, cfg: G1FallEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # History buffer
        self.hist_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_state_buffer = torch.zeros((self.num_envs, 8), dtype=torch.float32, device=self.device)
        self.root_state_buffer = torch.zeros((self.num_envs, self.cfg.body_hist_length, 8), dtype=torch.float32, device=self.device)

        # Capturability information
        self.ICP_pos_w           = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self.capturable_boundary = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
        self.dist_from_icp_to_stance = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)

    # Overriding to add RA states
    def _get_states(self) -> dict[str, torch.Tensor]:
        states = super()._get_states()

        # Reach-Avoid information
        self.extras["ra_states"] = torch.cat([self.root_lin_vel_b,                                  # [E, 3]
                                              self.root_ang_vel_b,                                  # [E, 3]
                                              self.projected_gravity,                               # [E, 3]
                                              self.phase.unsqueeze(-1),                             # [E, 1]
                                            #   self.dist_from_icp_to_stance,                       # [E, 1]
                                            #   self.root_state_buffer.reshape(self.num_envs, -1)   # [E, body_hist_length*8]
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

        return states

    # Overriding to add new fall termination condition
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        base_fall = self.CoM[:, 2] <= self.cfg.termination_height
        critical_contact_forces = self.contact_sensors.data.net_forces_w[:, self.denied_collision_link_ids]
        
        died_collision   = torch.any(torch.norm(critical_contact_forces, dim=-1) > 1.0, dim=1)
        died = died_collision & base_fall

        return died, time_out

    # Overriding to add history buffer reset
    def _reset_idx(self, env_ids):
        # History buffer reset
        self.hist_count[env_ids] = 0
        self.root_state_buffer[env_ids] = 0.0
        self.prev_state_buffer[env_ids] = 0.0
        super()._reset_idx(env_ids)

    # Overriding to compute capturability information and update history buffer
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        super()._compute_intermediate_values(env_ids)
        i = env_ids if env_ids is not None else self._robot._ALL_INDICES
        
        # Capturable
        icp_x, icp_y, radius = self.compute_2_step_capturability(env_ids=i)
        self.ICP_pos_w[i] = torch.stack([icp_x, icp_y], dim=-1)
        self.capturable_boundary[i] = radius.unsqueeze(-1)
        self.dist_from_icp_to_stance[i] = torch.norm(self.ICP_pos_w[i, :2] - self.support_foot_pos[i, :2], dim=-1).unsqueeze(-1)

        if env_ids is None:
            # History buffer update
            self.root_state_buffer[i, :-1] = self.root_state_buffer[i, 1:].clone()
            self.root_state_buffer[i, -1]  = self.prev_state_buffer[i].clone()
        
        # Prev state for history buffer
        self.prev_state_buffer[i] = torch.cat([self.root_ang_vel_b[i], self.projected_gravity[i], self.dist_from_icp_to_stance[i], self.phase[i].unsqueeze(-1)], dim=-1)


    # ======================= Auxillary functions ======================= #
    
    def compute_2_step_capturability(self, env_ids):
        # Data setting
        step_period = self.step_period[env_ids]
        command_count = self.command_count[env_ids]
        tau = self.step_dt * command_count
        T = self.step_dt * step_period
        z_c = self.CoM[env_ids, 2]
        w0 = torch.sqrt(9.81 / (z_c + 1e-6))

        # ICP at tau
        xi_t_x = self.CoM[env_ids, 0] + self.root_lin_vel_w[env_ids, 0] / w0
        xi_t_y = self.CoM[env_ids, 1] + self.root_lin_vel_w[env_ids, 1] / w0

        # 2-step capturable boundary
        phase_1 = w0 * T
        phase_2 = w0 * (T - tau)
        radius = self.cfg.l_max * torch.exp(-phase_2) * (1 + torch.exp(-phase_1))

        return xi_t_x, xi_t_y, radius
        
    def _update_viz_data(self):
        extras = copy.deepcopy(self.extras)

        dist_from_icp_to_ankle = self.dist_from_icp_to_stance[0, 0]

        # extras["viz_data"]["com_pos"]               = self.CoM[0]
        # extras["viz_data"]["left_foot_pos"]         = self.foot_pos_w[0, 0, :]
        # extras["viz_data"]["right_foot_pos"]        = self.foot_pos_w[0, 1, :]
        # extras["viz_data"]["icp_pos"]               = self.ICP_pos_w[0]
        # extras["viz_data"]["capture_region_center"] = self.support_foot_pos[0, :2]
        # extras["viz_data"]["capture_region_radius"] = self.capturable_boundary[0, 0]
        # extras["viz_data"]["time_hist"]             = self.episode_length_buf[0] * self.step_dt
        extras["viz_data"]["icp_ankle_dist_hist"]   = dist_from_icp_to_ankle

        return extras