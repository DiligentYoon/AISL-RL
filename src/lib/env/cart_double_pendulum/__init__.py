# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Inverted Double Pendulum on a Cart balancing environment.
"""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="My-Cart-Double-Pendulum-Direct-Test",
    entry_point=f"{__name__}.cart_double_pendulum_env:CartDoublePendulumEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cart_double_pendulum_env_cfg:CartDoublePendulumEnvCfg",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
    },
)
