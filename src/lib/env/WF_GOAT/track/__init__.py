import gymnasium as gym

gym.register(
    id="WF-GOAT-track", 
    entry_point=f"{__name__}.WF_GOAT_track_env:WFGOATTrackEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.WF_GOAT_track_env_cfg:WFGOATTrackEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

gym.register(
    id="WF-GOAT-track-play", 
    entry_point=f"{__name__}.WF_GOAT_track_env:WFGOATTrackEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.WF_GOAT_track_env_cfg:WFGOATTrackPlayEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

gym.register(
    id="WF-GOAT-track-fixed", 
    entry_point=f"{__name__}.WF_GOAT_track_fixed_env:WFGOATTrackFixedEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.WF_GOAT_track_fixed_env_cfg:WFGOATTrackFixedEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

gym.register(
    id="WF-GOAT-track-fixed-play", 
    entry_point=f"{__name__}.WF_GOAT_track_fixed_env:WFGOATTrackFixedEnv",
    disable_env_checker=True,
    kwargs={
        # Environment-Specific Entry Point for Env Cfg Class
        "env_cfg_entry_point": f"{__name__}.WF_GOAT_track_fixed_env_cfg:WFGOATTrackFixedPlayEnvCfg",
        "rl_ppo_cfg_entry_point": f"{__name__}.cfg:ppo_cfg.yaml",
    }
)

print(f"Registration is Complete.")