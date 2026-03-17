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
    }
)