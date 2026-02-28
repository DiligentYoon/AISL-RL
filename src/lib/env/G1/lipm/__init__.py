import gymnasium as gym

gym.register(
    id="G1-lipm", 
    entry_point=f"{__name__}.G1_lipm_env:G1LIPMLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.G1_lipm_env_cfg:G1LIPMEnvCfg",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
    }
)