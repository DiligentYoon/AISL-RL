# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import copy

from lib.env.G1.fall.G1_fall_env_cfg import G1FallEnvCfg
from lib.env.G1.fall.G1_fall_env import G1FallEnv

class G1FallCollectEnv(G1FallEnv):
    cfg: G1FallEnvCfg

    def __init__(self, cfg:G1FallEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # State collection
        self.extras["collection"] = {}
        self.extras["collection"]["root_pos_offset_w"]      = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.extras["collection"]["root_quat_w"]            = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.extras["collection"]["root_lin_vel_w"]         = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.extras["collection"]["root_ang_vel_w"]         = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.extras["collection"]["joint_pos"]              = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.extras["collection"]["joint_vel"]              = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.extras["collection"]["prev_action"]            = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)

        # Disturbance record, written by the push_and_log event term.
        # Layout matches root_vel_w: (x, y, z, roll, pitch, yaw).
        self.extras["disturbance"]         = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self.extras["disturbance_applied"] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # RA state captured before auto-reset, so the terminal sample is not lost.
        self.extras["terminal_ra_state"] = torch.zeros((self.num_envs, self.cfg.ra_state_space), dtype=torch.float32, device=self.device)

    # Overriding to capture the terminal RA state before Env.step auto-resets.
    # _get_dones runs after _compute_intermediate_values and before _reset_idx,
    # so the cached fields still describe the state that ended the episode.
    def _get_dones(self):
        died, time_out = super()._get_dones()
        self.extras["terminal_ra_state"] = self._build_ra_state()
        return died, time_out

    # Overriding to reopen the one-push-per-episode guard at the episode boundary.
    def _reset_idx(self, env_ids):
        self.extras["disturbance"][env_ids]         = 0.0
        self.extras["disturbance_applied"][env_ids] = False
        super()._reset_idx(env_ids)

    def _get_states(self):
        states = super()._get_states()

        # Update collection
        self.extras["collection"]["root_pos_offset_w"]     = self.root_pos_w - self.scene.env_origins
        self.extras["collection"]["root_quat_w"]           = self.root_rot_w
        self.extras["collection"]["root_lin_vel_w"]        = self.root_lin_vel_w
        self.extras["collection"]["root_ang_vel_w"]        = self.root_ang_vel_w
        self.extras["collection"]["joint_pos"]             = self.joint_pos
        self.extras["collection"]["joint_vel"]             = self.joint_vel

        # Compose joint-natural-ordered prev_action from arm/leg dict.
        # Called before _get_rewards updates prev_actions, so this holds a_{t-1}.
        self.extras["collection"]["prev_action"][:, self.total_arm_joint_ids] = self.prev_actions["arm"]
        self.extras["collection"]["prev_action"][:, self.total_leg_joint_ids] = self.prev_actions["leg"]

        return states