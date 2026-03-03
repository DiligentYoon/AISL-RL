import gymnasium as gym

gym.register(
    id="G1-gait", 
    entry_point=f"{__name__}.G1_gait_env:G1GaitEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.G1_gait_env_cfg:G1GaitEnvCfg",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
    }
)

print(f"Registration is Complete.")