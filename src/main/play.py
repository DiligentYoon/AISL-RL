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
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")

parser.add_argument("--algorithm",
                    type=str,
                    default="PPO",
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
    # Initialization
    if cfg["buffer"]["buffer_size"] == -1:
        cfg["buffer"]["buffer_size"] = cfg["agent"]["rollouts"]
    else:
        raise RuntimeError("Replaybuffer for Off-policy algorithm is not implemented yet.")
    
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
    from lib.model.MLP import Actor, Critic
    from lib.model.NerveNet import NerveNetPolicy
    from lib.utils.graph_utils import Mapping
    from lib.model.BodyTransformer.body_transformer import BodyLevelActor, BodyLevelCritic
    from lib.model.BodyTransformer.linear_components import ObsTokenizer, ValueDetokenizer, ActionDetokenizer
    from lib.model.BodyTransformer.transformer_components import BodyTransformer
    # from lib.agent.ppo import PPO
    from lib.agent.ppo_new import PPO
    from lib.agent.mappo import MAPPO
    from lib.agent.cooperative_mappo import CooperativeMAPPO
    
    # 1. Initialization
    # Model initialization
    is_shared = cfg["models"].get("shared", False)
    is_squashed = cfg["models"].get("squashed", False)
    is_cooperative = cfg["models"].get("cooperative", None)
    model_type = cfg["models"]["policy"].get("type", None)
    model = {}
    if multi_agent:
        if model_type is not None:
            raise RuntimeError("MARL With CTDE structure only supports a MLP network.")

        if is_cooperative is not None:
            cfg["agent"]["proactive"] = env._unwrapped.cfg.proactive_id
            cfg["agent"]["reactive"] = env._unwrapped.cfg.reactive_id
            # For proactive action processing 
            obs_size[cfg["agent"]["reactive"]] += act_size[cfg["agent"]["proactive"]] 
            state_size[cfg["agent"]["reactive"]] += act_size[cfg["agent"]["proactive"]]

        for uid in possible_agents:
            # Per-Agent Network
            actor = Actor(num_observations=obs_size[uid],
                          num_actions=act_size[uid],
                          min_log_std=cfg["models"]["policy"]["min_log_std"],
                          max_log_std=cfg["models"]["policy"]["max_log_std"],
                          squash=is_squashed,
                          device=env.device)
            critic = Critic(num_states=state_size[uid],
                            device=env.device)
            
            model[uid] = {
                'actor': actor,
                'critic': critic
            }

        if is_cooperative is not None:
            agent = CooperativeMAPPO(observation_space=env.observation_space,
                                     state_space=env.state_space,
                                     action_space=env.action_space,
                                     possible_agents=possible_agents,
                                     model=model,
                                     buffer=buffers,
                                     device=env.device,
                                     cfg=cfg["agent"])
        
        else:
            agent = MAPPO(observation_space=env.observation_space,
                        state_space=env.state_space,
                        action_space=env.action_space,
                        possible_agents=possible_agents,
                        model=model,
                        buffer=buffers,
                        device=env.device,
                        cfg=cfg["agent"])

    else:
        if model_type is None:
                actor = Actor(num_observations=obs_size,
                              num_actions=act_size,
                              min_log_std=cfg["models"]["policy"]["min_log_std"],
                              max_log_std=cfg["models"]["policy"]["max_log_std"],
                              squash=is_squashed,
                              device=env.device)
                
                critic = Critic(num_states=state_size,
                                device=env.device)
            
        else:
            model_type_lower = model_type.lower()
            if model_type_lower == "gnn":
                actor = NerveNetPolicy(
                    observation_space=observation_space,
                    action_space=action_space,
                    node_info=env._unwrapped.cfg.node_info,
                    device=env.device,
                    num_nodes=env._unwrapped.cfg.num_nodes,
                    num_actuated_nodes=env._unwrapped.cfg.num_actuated_nodes,
                    min_log_std=cfg['models']['policy']['min_log_std'],
                    max_log_std=cfg['models']['policy']['max_log_std'],
                )

                critic = Critic(num_states=state_size,
                                device=env.device)
                
            elif model_type_lower == "bodytransformer":
                mapping = Mapping(env._unwrapped.cfg.map_info)
                use_mlp = cfg["models"].get("use_mlp", False)
                action_detokenizer = ActionDetokenizer(mapping=mapping,
                                                    action_dim=action_space.shape[0], 
                                                    device=env.device)
                value_detokenizer = ValueDetokenizer(mapping=mapping,
                                                    use_mlp=use_mlp, 
                                                    device=env.device)
                if is_shared:
                    if state_space is not None:
                        raise RuntimeError("Shared structure should not use state space different from observation sapce.")
                    
                    tokenizer = ObsTokenizer(mapping=mapping,
                                            device=env.device)
                    trunk = BodyTransformer(mapping=mapping,
                                            device=env.device)

                    actor = BodyLevelActor(
                        observation_space=observation_space,
                        action_space=action_space,
                        mapping=mapping,
                        tokenizer=tokenizer,
                        trunk=trunk,
                        detokenizer=action_detokenizer,
                        device=env.device,
                        min_log_std=cfg['models']['policy']['min_log_std'],
                        max_log_std=cfg['models']['policy']['max_log_std'],)
                    
                    critic = BodyLevelCritic(
                        state_space=observation_space,
                        mapping=mapping,
                        tokenizer=tokenizer,
                        trunk=trunk,
                        detokenizer=value_detokenizer,
                        device=env.device)
                    
                else:
                    actor = BodyLevelActor(
                        observation_space=observation_space,
                        action_space=action_space,
                        mapping=mapping,
                        tokenizer=ObsTokenizer(mapping=mapping,
                                            device=env.device),
                        trunk=BodyTransformer(mapping=mapping,
                                            device=env.device),
                        detokenizer=action_detokenizer,
                        device=env.device,
                        min_log_std=cfg['models']['policy']['min_log_std'],
                        max_log_std=cfg['models']['policy']['max_log_std'],)
                    
                    critic = BodyLevelCritic(
                        state_space=observation_space if state_space is None else state_space,
                        mapping=mapping,
                        tokenizer=ObsTokenizer(mapping=mapping,
                                            device=env.device),
                        trunk=BodyTransformer(mapping=mapping,
                                            device=env.device),
                        detokenizer=value_detokenizer,
                        device=env.device)

            else:
                raise ValueError(f"Unknown model type specified in cfg: {model_type}")

        model = {"actor": actor, "critic": critic}

        # Agent initialization
        agent = PPO(model=model,
                    buffer=buffer, 
                    device=env.device,
                    cfg=cfg["agent"],
                    shared=is_shared)
    

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
    obs, states, _ = env.reset()
    rollout = 0
    timestep = 0
    test_checkpoint_step = 100
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions, action_log_probs, _, _ = agent.act(obs, timestep=timestep, deterministic=True)
            # env stepping
            next_obs, next_states, rewards, terminated, truncated, infos = env.step(actions)
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


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()