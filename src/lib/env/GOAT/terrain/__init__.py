import gymnasium as gym

gym.register(
    id="GOAT-stand-terrain", 
    entry_point=f"{__name__}.GOAT_stand_terrain_env:GOATStandTerrainEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.GOAT_stand_terrain_env_cfg:GOATStandTerrainEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

print(f"Registration is Complete.")