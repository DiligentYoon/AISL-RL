
"""
Ant locomotion environment.
"""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Ant-GNN",
    entry_point=f"{__name__}.ant_gnn_env:AntGNNEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_gnn_env_cfg:AntGNNEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    },
)