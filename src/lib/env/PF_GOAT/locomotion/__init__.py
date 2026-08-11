import gymnasium as gym

gym.register(
    id="PF-GOAT-locomotion", 
    entry_point=f"{__name__}.PF_GOAT_locomotion_env:PFGOATLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.PF_GOAT_locomotion_env_cfg:PFGOATLocomotionEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

gym.register(
    id="PF-GOAT-locomotion-play", 
    entry_point=f"{__name__}.PF_GOAT_locomotion_env:PFGOATLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.PF_GOAT_locomotion_env_cfg:PFGOATLocomotionPlayEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

print(f"Registration is Complete.")