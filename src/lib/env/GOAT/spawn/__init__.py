import gymnasium as gym

gym.register(
    id="GOAT-spawn-v0", 
    entry_point=f"{__name__}.GOAT_spawn_env:GOATSpawnEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.GOAT_spawn_env_cfg:GOATSpawnEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

print(f"Registration is Complete.")