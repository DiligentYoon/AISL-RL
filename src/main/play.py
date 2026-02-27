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
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="G1-balancing-locomotion", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default="logs/g1_balancing_locomotion/2026-02-27_17-59-10_mappo/agent_28800.pt", help="Path to model checkpoint.")

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

from lib.utils.plot_utils import PyQtLivePlotter
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
    env_cfg.seed = cfg.get("seed", None)
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
    if env._unwrapped.cfg.viz_data is not None:
        plot_cfg = env._unwrapped.cfg.viz_data
        plot = PyQtLivePlotter(env, plot_cfg)
    else:
        plot = None

    agent.set_running_mode("eval")
    obs, states, infos = env.reset()
    timestep = 0
    per_step_task_rewards = None
    cumulative_task_rewards = None
    cumulative_total_rewards = 0
    cumulative_steps = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.no_grad():
            # agent stepping
            actions, nonscaled_actions, action_log_probs, _ = agent.act(obs, infos, timestep=timestep, deterministic=True)
            # env stepping
            next_obs, next_states, rewards, terminated, truncated, next_infos = env.step(actions)

        # Task reward
        task_reward = next_infos.get("reward", None)
        if task_reward is not None:
            if cumulative_task_rewards is None:
                # Initialization
                cumulative_task_rewards = {}
                for k in task_reward.keys():
                    cumulative_task_rewards[k] = 0
            
            if per_step_task_rewards is None:
                # Initialization
                per_step_task_rewards = {}
                for k in task_reward.keys():
                    per_step_task_rewards[k] = 0
            
            for k, v in task_reward.items():
                cumulative_task_rewards[k] += v.mean().item()

        # Total reward
        cumulative_total_rewards += rewards[0].mean().item()
        cumulative_steps += 1

        # logging and reset metric
        if terminated[0] | truncated[0]:
            # Per-step calculation
            per_step_total_rewards = cumulative_total_rewards / cumulative_steps
            if cumulative_task_rewards:
                for k, v in cumulative_task_rewards.items():
                    per_step_task_rewards[k] = cumulative_task_rewards[k] / cumulative_steps

            content_width = 80
            line_header = "Evaluation Metric Table"
            print(f" {'_' * content_width}")
            print(f"|{' ' * content_width}|")
            print(f"|{line_header.center(content_width)}|")
            print(f"|{'_' * content_width}|")
            print(f"|{' ' * content_width}|")
            if cumulative_task_rewards:
                for k, v in cumulative_task_rewards.items():
                    print(f"| {k:<50}: {v:<26.3f} |")
            if per_step_task_rewards:
                for k, v in per_step_task_rewards.items():
                    print(f"| {f'Per step {k}':<50}: {v:<26.3f} |")
            print(f"| {'Total Reward':<50}: {cumulative_total_rewards:<26.3f} |")
            print(f"| {'Total Step':<50}: {cumulative_steps:<26d} |")
            print(f"|{'_' * content_width}|")

            for k in task_reward.keys():
                if cumulative_task_rewards:
                    cumulative_task_rewards[k] = 0
                if per_step_task_rewards:
                    per_step_task_rewards[k] = 0

            cumulative_total_rewards = 0
            cumulative_steps = 0

        # Plot Phase
        if plot is not None:
            # Plotter Update
            if "viz_data" in next_infos:
                plot.update(next_infos["viz_data"])

            # Plotter should be resetted in accordance with env reset (Assume single env (index = 0))
            if terminated[0] | truncated[0]:
                plot.reset()

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