# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid locomotion environment.
"""

import gymnasium as gym


##
# Register Gym environments.
##

gym.register(
    id="My-Humanoid-Test",
    entry_point=f"{__name__}.humanoid_env:HumanoidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_env_cfg:HumanoidEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml"
    },
)