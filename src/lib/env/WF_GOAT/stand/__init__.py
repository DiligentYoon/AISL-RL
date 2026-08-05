import gymnasium as gym

gym.register(
    id="WF-GOAT-stand", 
    entry_point=f"{__name__}.WF_GOAT_stand_env:WFGOATStandEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.WF_GOAT_stand_env_cfg:WFGOATStandEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

gym.register(
    id="WF-GOAT-stand-play", 
    entry_point=f"{__name__}.WF_GOAT_stand_env:WFGOATStandEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.WF_GOAT_stand_env_cfg:WFGOATStandPlayEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

print(f"Registration is Complete.")