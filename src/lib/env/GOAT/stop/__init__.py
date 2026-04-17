import gymnasium as gym

gym.register(
    id="GOAT-stop", 
    entry_point=f"{__name__}.GOAT_stop_env:GOATStopEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.GOAT_stop_env_cfg:GOATStopEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

gym.register(
    id="GOAT-stop-play", 
    entry_point=f"{__name__}.GOAT_stop_env:GOATStopEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.GOAT_stop_env_cfg:GOATStopPlayEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

print(f"Registration is Complete.")