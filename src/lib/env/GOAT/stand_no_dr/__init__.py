import gymnasium as gym

gym.register(
    id="GOAT-stand-no-dr", 
    entry_point=f"{__name__}.GOAT_stand_no_dr_env:GOATStandNoDREnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.GOAT_stand_no_dr_env_cfg:GOATStandNoDREnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

print(f"Registration is Complete.")