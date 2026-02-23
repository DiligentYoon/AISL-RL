import gymnasium as gym

gym.register(
    id="G1-balancing-locomotion", 
    entry_point=f"{__name__}.G1_balancing_locomotion_env:G1BalancingLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.G1_balancing_locomotion_env_cfg:G1BalancingLocomotionEnvCfg",
        "rl_mappo_cfg_entry_point": f"{__name__}.cfg:mappo_cfg.yaml",
    }
)

print(f"Registration is Complete.")