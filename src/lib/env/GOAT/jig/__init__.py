import gymnasium as gym

gym.register(
    id="GOAT-jig", 
    entry_point=f"{__name__}.GOAT_jig_env:GOATJigEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.GOAT_jig_env_cfg:GOATJigEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

gym.register(
    id="GOAT-jig-play", 
    entry_point=f"{__name__}.GOAT_jig_env:GOATJigEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.GOAT_jig_env_cfg:GOATJigPlayEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

print(f"Registration is Complete.")