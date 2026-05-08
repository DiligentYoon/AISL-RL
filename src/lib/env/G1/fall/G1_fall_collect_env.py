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