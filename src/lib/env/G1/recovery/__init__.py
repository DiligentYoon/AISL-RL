import gymnasium as gym

gym.register(
    id="G1-recovery", 
    entry_point=f"{__name__}.G1_recovery_env:G1RecoveryEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.G1_recovery_env_cfg:G1RecoveryEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
    }
)

gym.register(
    id="G1-recovery-play", 
    entry_point=f"{__name__}.G1_recovery_env:G1RecoveryEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.G1_recovery_env_cfg:G1RecoveryPlayEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
    }
)