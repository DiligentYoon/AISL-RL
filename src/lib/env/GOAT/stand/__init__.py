import gymnasium as gym

gym.register(
    id="GOAT-stand", 
    entry_point=f"{__name__}.GOAT_stand_env:GOATStandEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.GOAT_stand_env_cfg:GOATStandEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

gym.register(
    id="GOAT-stand-play", 
    entry_point=f"{__name__}.GOAT_stand_env:GOATStandEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.GOAT_stand_env_cfg:GOATStandPlayEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

print(f"Registration is Complete.")