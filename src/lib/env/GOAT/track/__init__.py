import gymnasium as gym

gym.register(
    id="GOAT-track", 
    entry_point=f"{__name__}.GOAT_track_env:GOATTrackEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.GOAT_track_env_cfg:GOATTrackEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

gym.register(
    id="GOAT-track-play", 
    entry_point=f"{__name__}.GOAT_track_env:GOATTrackEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.GOAT_track_env_cfg:GOATTrackPlayEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

print(f"Registration is Complete.")