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
parser.add_argument("--disable_fabric", type=bool, default=True, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="My-Ant-Test", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")

parser.add_argument("--algorithm",
                    type=str,
                    default="PPO",
                    choices=["PPO", "SAC", "TD3"],
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
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"rl_{algorithm}_cfg_entry_point")
    except ValueError as e:
        print(e)
        return

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}"
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Exact experiment name requested from command line: {log_dir}")
    if experiment_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{experiment_cfg['agent']['experiment']['experiment_name']}"
    log_dir = os.path.join(log_root_path, log_dir)

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
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    # ==================================================================================================================
    # ======================================== Buffer Spawn Test =======================================================
    # ==================================================================================================================
    from lib.buffer.rolloutbuffer import RolloutBuffer

    # 1. initialization
    if experiment_cfg["buffer"]["buffer_size"] == -1:
        experiment_cfg["buffer"]["buffer_size"] = experiment_cfg["agent"]["rollouts"]
    buffer = RolloutBuffer(experiment_cfg["buffer"]["buffer_size"], env.num_envs, device=env.device)
    
    observation_space = env.observation_space
    action_space = env.action_space
    if env.state_space:
        state_space = env.state_space
        experiment_cfg["agent"]["async_actor_critic"] = True
    else:
        state_space = None
        experiment_cfg["agent"]["async_actor_critic"] = False
    buffer.init_buffer(observation_space, state_space, action_space)
    obs_size = buffer.tensors["observations"].shape[-1]
    state_size = buffer.tensors["states"].shape[-1] if env.state_space else obs_size
    act_size = buffer.tensors["actions"].shape[-1]
    # for _ in range(3):
    #     # 2. Storing
    #     for i in range(experiment_cfg["agent"]["rollouts"]):
            # obs_size = buffer.tensors["states"].shape[-1]
            # act_size = buffer.tensors["actions"].shape[-1]
    #         buffer.add_samples(
    #             states=torch.randn((env.num_envs, obs_size), dtype=torch.float32, device=env.device),
    #             next_states = torch.randn((env.num_envs, obs_size), dtype=torch.float32, device=env.device),
    #             actions=torch.randn((env.num_envs, act_size), dtype=torch.float32, device=env.device),
    #             action_log_probs = torch.randn((env.num_envs, 1), dtype=torch.float32, device=env.device),
    #             rewards=torch.randn((env.num_envs, 1), dtype=torch.float32, device=env.device),
    #             truncated=torch.zeros((env.num_envs, 1), dtype=torch.bool, device=env.device),
    #             terminated=torch.zeros((env.num_envs, 1), dtype=torch.bool, device=env.device),
    #             value_preds=torch.randn((env.num_envs, 1), dtype=torch.float32, device=env.device))
    #     # 3. Sampling
    #     sampled_data = buffer.sample(('states', 'actions', 'rewards'), experiment_cfg["agent"]["rollouts"], experiment_cfg["agent"]["mini_batches"])
    #     sampled_states = buffer.get_tensor_by_name("states", keepdim=True)
    #     sampled_states_2d = buffer.get_tensor_by_name("states", keepdim=False)
        # 4. GAE calculation (omit for parameter update test of agent class)
        # buffer.compute_gae(torch.randn((env.num_envs, 1), dtype=torch.float32, device=env.device), gamma=0.99, lamb=0.95)


    # ==========================================================================================================================
    # ======================================== Model & Agent Spawn Test ========================================================
    # ==========================================================================================================================
    from lib.model.MLP import Actor, Critic
    from lib.agent.ppo import PPO
    
    # 1. Initialization
    actor = Actor(num_observations=obs_size,
                  num_actions=act_size,
                  device=env.device)

    critic = Critic(num_states=state_size,
                    device=env.device)

    model = {"actor": actor, "critic": critic}
    agent = PPO(model=model,
                buffer=buffer, 
                device=env.device,
                cfg=experiment_cfg["agent"])
    

    # 2. Checkpoint
    if args_cli.checkpoint is not None:
        resume_path = os.path.abspath(args_cli.checkpoint)
        agent.load(resume_path)
        print(f"[INFO] Get checkpoint from {resume_path}")
    else:
        print("[INFO] Unfortunately a pre-trained checkpoint is not found for this task.")
        resume_path = None
    
    # # 2. Forward propagation (Actor, Critic)
    # actions, log_probs, entropy = agent.act(sampled_states, timestep=1, deterministic=True)
    # values, _ = agent.critic(sampled_states, deterministic=True)
    
    # # 3. Parameter Update
    # policy_loss, value_loss, entropy_loss, approx_kl = agent.update()


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
            actions, action_log_probs, _ = agent.act(obs, timestep=timestep, deterministic=False)
            # env stepping
            next_obs, next_states, rewards, terminated, truncated, infos = env.step(actions)
            # update rollout number
            rollout += 1
        
            # Insert data to the buffer
            agent.insert_data(observations=obs,
                              states=states,
                              actions=actions,
                              action_log_probs=action_log_probs.reshape(-1, 1),
                              rewards=rewards,
                              next_observations=next_obs,
                              next_states=next_states,
                              truncated=truncated,
                              terminated=terminated,
                              infos=infos)
        
        # Parameter update
        if rollout % buffer.buffer_size == 0:
            if buffer.memory_index == 0:
                policy_loss, value_loss, entropy_loss, approx_kl = agent.update()
            else:
                raise RuntimeError("Discrepency appears between Buffer Logic and Rollout Policy.")

        # Checkpoint save
        if rollout % test_checkpoint_step == 0:
            checkpoint_path = os.path.join(log_dir, f"agent_{rollout}.pt")
            agent.save(checkpoint_path)

        # Video update
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        
        # state update
        obs = next_obs
        states = next_states


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()