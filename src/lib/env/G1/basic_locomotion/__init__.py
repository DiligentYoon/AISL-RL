import gymnasium as gym

gym.register(
    id="G1-basic-locomotion", 
    entry_point=f"{__name__}.G1_basic_locomotion_env:G1BasicLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.G1_basic_locomotion_env_cfg:G1BasicLocomotionEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

print(f"Registration is Complete.")