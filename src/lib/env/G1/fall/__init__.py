import gymnasium as gym

gym.register(
    id="G1-fall", 
    entry_point=f"{__name__}.G1_fall_env:G1FallEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.G1_fall_env_cfg:G1FallEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
        "ra_cfg_entry_point": f"{__name__}.cfg:ra_cfg.yaml",
    }
)

gym.register(
    id="G1-fall-play",
    entry_point=f"{__name__}.G1_fall_env:G1FallEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.G1_fall_env_cfg:G1FallPlayEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
        "ra_cfg_entry_point": f"{__name__}.cfg:ra_cfg.yaml",
    }
)

gym.register(
    id="G1-fall-collect",
    entry_point=f"{__name__}.G1_fall_collect_env:G1FallCollectEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.G1_fall_env_cfg:G1FallCollectEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
        "ra_cfg_entry_point": f"{__name__}.cfg:ra_cfg.yaml",
    }
)

gym.register(
    id="G1-fall-region-collect",
    entry_point=f"{__name__}.G1_fall_collect_env:G1FallCollectEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.G1_fall_env_cfg:G1FallRegionCollectEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
        "ra_cfg_entry_point": f"{__name__}.cfg:ra_cfg.yaml",
    }
)

gym.register(
    id="G1-fall-unified-play",
    entry_point=f"{__name__}.G1_fall_unified_env:G1FallUnifiedEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.G1_fall_env_cfg:G1FallUnifiedPlayEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
        "ra_cfg_entry_point": f"{__name__}.cfg:ra_cfg.yaml",
        "safe_rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
        "safe_rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml"
    }
)