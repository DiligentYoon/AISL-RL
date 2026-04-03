import gymnasium as gym

gym.register(
    id="G1-pusher", 
    entry_point=f"{__name__}.G1_pusher_env:G1PusherEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.G1_pusher_env_cfg:G1PusherEnvCfg",
        "rl_ppo_nominal_cfg_entry_point": f"{__name__}.cfg:ppo_nominal_cfg.yaml",
        "rl_ppo_pusher_cfg_entry_point": f"{__name__}.cfg:ppo_pusher_cfg.yaml",
        "rl_mappo_nominal_cfg_entry_point": f"{__name__}.cfg:mappo_nominal_cfg.yaml",
    }
)