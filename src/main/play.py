"""
Script to play a checkpoint of an RL agent.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="G1-basic-locomotion", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default="logs/g1_locomotion/2026-02-21_00-27-37_mappo/agent_3840.pt", help="Path to model checkpoint.")

parser.add_argument("--algorithm",
                    type=str,
                    default="MAPPO",
                    choices=["PPO", "SAC", "TD3", "MAPPO"],
                    help="The RL algorithm used for training the agent.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import copy
import numpy as np

from datetime import datetime

import lib

from lib.utils.parse_utils import parse_env_cfg, load_cfg_from_registry
from wrapper.isaaclab_wrapper import IsaacLabWrapper

# config shortcuts
algorithm = args_cli.algorithm.lower()

def main():

    # ================================================================================================================
    # =========================================== Parsing Test =======================================================
    # ================================================================================================================
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    try:
        cfg = load_cfg_from_registry(args_cli.task, f"rl_{algorithm}_cfg_entry_point")
    except ValueError as e:
        print(e)
        return

    # ============================================================================================================================
    # =========================================== Env Spawn & Wrapper Test =======================================================
    # ============================================================================================================================

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap around environment
    env = IsaacLabWrapper(env)  

    # configure and instantiate the skrl runner
    cfg["agent"]["experiment"]["write_interval"] = 0  
    cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    # ==================================================================================================================
    # ======================================== Buffer Spawn Test =======================================================
    # ==================================================================================================================
    from lib.buffer.rolloutbuffer import RolloutBuffer

    multi_agent = algorithm == "mappo"
    cfg["models"]["multi_agent"] = multi_agent
    # Initialization
    if cfg["buffer"]["buffer_size"] == -1:
        cfg["buffer"]["buffer_size"] = cfg["agent"]["rollouts"]
    else:
        raise RuntimeError("Replaybuffer for Off-policy algorithm is not implemented yet.")
    
    possible_agents = None
    if multi_agent:
        obs_size = {}
        state_size = {}
        act_size = {}
        buffers = {}
        possible_agents = env._unwrapped.cfg.possible_agents
        for uid in possible_agents:
            observation_space = env.observation_space[uid]
            action_space = env.action_space[uid]
            if env.state_space:
                state_space = env.state_space[uid]
                cfg["agent"]["async_actor_critic"] = True
            else:
                state_space = None
                cfg["agent"]["async_actor_critic"] = False
            
            buffer = RolloutBuffer(cfg["buffer"]["buffer_size"], env.num_envs, device=env.device)
            buffer.init_buffer(observation_space, state_space, action_space)
            buffers[uid] = buffer
            obs_size[uid] = buffer.tensors["observations"].shape[-1]
            state_size[uid] = buffer.tensors["states"].shape[-1] if env.state_space else obs_size[uid]
            act_size[uid] = buffer.tensors["actions"].shape[-1]

    else:
        observation_space = env.observation_space
        action_space = env.action_space
        if env.state_space:
            state_space = env.state_space
            cfg["agent"]["async_actor_critic"] = True
        else:
            state_space = None
            cfg["agent"]["async_actor_critic"] = False
        
        buffer = RolloutBuffer(cfg["buffer"]["buffer_size"], env.num_envs, device=env.device)
        buffer.init_buffer(observation_space, state_space, action_space)
        obs_size = buffer.tensors["observations"].shape[-1]
        state_size = buffer.tensors["states"].shape[-1] if env.state_space else obs_size
        act_size = buffer.tensors["actions"].shape[-1]


    # ==========================================================================================================================
    # ======================================== Model & Agent Spawn Test ========================================================
    # ==========================================================================================================================
    from lib.model.model_factory import ModelFactory
    # ====================== Model Spawn  ==========================
    model_manager = ModelFactory(cfg=cfg["models"], device=env.device)
    if model_manager.model_type is None:
        models = model_manager.generate_mlp_models(observation_size=obs_size,
                                                   state_size=state_size,
                                                   action_size=act_size,
                                                   possible_agents=possible_agents)
    else:
        node_cfg = None
        mapping_cfg = None
        if model_manager.model_type.lower() == "nervenet":
            node_cfg = {'node_info': env._unwrapped.cfg.node_info,
                        'num_nodes': env._unwrapped.cfg.num_nodes,
                        'num_actuated_nodes': env._unwrapped.cfg.num_actuated_nodes}
            
        elif model_manager.model_type.lower() == "bodytransformer":
            mapping_cfg = env._unwrapped.cfg.map_info

        else:
            raise RuntimeError("Not supported type")
        
        models = model_manager.generate_gnn_models(observation_space=observation_space,
                                                   state_space=state_space,
                                                   action_space=action_space,
                                                   node_cfg=node_cfg,
                                                   mapping_cfg=mapping_cfg)

    # ====================== Agent Spawn  ==========================
    # Scale Factor
    cfg["agent"]["action_scale_factor"] = env._unwrapped.cfg.action_scale_factor
    if multi_agent:
        if model_manager.is_shared:
            from lib.agent.cooperative_mappo import CooperativeMAPPO
            agent = CooperativeMAPPO(observation_space=env.observation_space,
                                     state_space=env.state_space,
                                     action_space=env.action_space,
                                     possible_agents=possible_agents,
                                     model=models,
                                     buffer=buffers,
                                     device=env.device,
                                     cfg=cfg["agent"])
        else:
            from lib.agent.mappo import MAPPO
            agent = MAPPO(observation_space=env.observation_space,
                          state_space=env.state_space,
                          action_space=env.action_space,
                          possible_agents=possible_agents,
                          model=models,
                          buffer=buffers,
                          device=env.device,
                          cfg=cfg["agent"])
    else:
        from lib.agent.ppo_scaled import PPO
        # Agent initialization
        agent = PPO(model=models,
                    buffer=buffer, 
                    device=env.device,
                    cfg=cfg["agent"],
                    shared=model_manager.is_shared)
    

    # 2. Checkpoint
    if args_cli.checkpoint is not None:
        resume_path = os.path.abspath(args_cli.checkpoint)
        agent.load(resume_path)
        print(f"[INFO] Get checkpoint from {resume_path}")
    else:
        print("[INFO] Unfortunately a pre-trained checkpoint is not found for this task.")
        resume_path = None


    # ======================================================================================================================
    # ======================================== Env Interaction Test ========================================================
    # ======================================================================================================================

    # reset environment
    obs, states, infos = env.reset()
    rollout = 0
    timestep = 0
    test_checkpoint_step = 100
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions, _,  _, _ = agent.act(obs, infos, timestep=timestep, deterministic=True)
            # env stepping
            next_obs, next_states, rewards, terminated, truncated, next_infos = env.step(actions)
            # update rollout number
            rollout += 1

        # Video update
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        
        # state update
        # simulation_app.update()
        obs = next_obs
        states = next_states
        infos = next_infos


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()