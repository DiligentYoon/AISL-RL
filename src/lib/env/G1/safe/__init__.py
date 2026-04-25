import gymnasium as gym

gym.register(
    id="G1-safe", 
    entry_point=f"{__name__}.G1_safe_env:G1SafeEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.G1_safe_env_cfg:G1SafeEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
        "ra_cfg_entry_point": f"{__name__}.cfg:ra_cfg.yaml"
    }
)

gym.register(
    id="G1-safe-play", 
    entry_point=f"{__name__}.G1_safe_env:G1SafeEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.G1_safe_env_cfg:G1SafePlayEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
        "ra_cfg_entry_point": f"{__name__}.cfg:ra_cfg.yaml",
        "safe_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml"
    }
)