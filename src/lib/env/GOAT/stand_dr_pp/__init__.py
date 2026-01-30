import gymnasium as gym

gym.register(
    id="GOAT-stand-dr-pp", 
    entry_point=f"{__name__}.GOAT_stand_dr_pp_env:GOATStandDRPPEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.GOAT_stand_dr_pp_env_cfg:GOATStandDRPPEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

print(f"Registration is Complete.")