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
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    # get checkpoint path
    if args_cli.checkpoint is not None:
        resume_path = os.path.abspath(args_cli.checkpoint)
        log_dir = os.path.dirname(os.path.dirname(resume_path))
    else:
        print("[INFO] Unfortunately a pre-trained checkpoint is not found for this task.")
        resume_path = None

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
    buffer.init_buffer(observation_space, action_space)
    for _ in range(3):
        # 2. Storing
        for i in range(experiment_cfg["agent"]["rollouts"]):
            obs_size = buffer.tensors["states"].shape[-1]
            act_size = buffer.tensors["actions"].shape[-1]
            buffer.add_samples(
                states=torch.randn((env.num_envs, obs_size), dtype=torch.float32, device=env.device),
                actions=torch.randn((env.num_envs, act_size), dtype=torch.float32, device=env.device),
                rewards=torch.randn((env.num_envs, 1), dtype=torch.float32, device=env.device),
                dones=torch.zeros((env.num_envs, 1), dtype=torch.bool, device=env.device),
                value_preds=torch.randn((env.num_envs, 1), dtype=torch.float32, device=env.device))
        # 3. Sampling
        sampled_data = buffer.sample(('states', 'actions', 'rewards'), experiment_cfg["agent"]["rollouts"], experiment_cfg["agent"]["mini_batches"])
        sampled_states = buffer.get_tensor_by_name("states", keepdim=True)
        sampled_states_2d = buffer.get_tensor_by_name("states", keepdim=False)
        # 4. GAE calculation
        buffer.compute_gae(torch.randn((env.num_envs, 1), dtype=torch.float32, device=env.device), gamma=0.99, lamb=0.95)



    # runner = Runner(env, experiment_cfg)

    # if resume_path is not None:
    #     print(f"[INFO] Loading model checkpoint from: {resume_path}")
    #     runner.agent.load(resume_path)
    # runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # values, _, _ = runner.agent.value.act({"states": runner.agent._state_preprocessor(obs)}, role="value")
            # actions = outputs[-1].get("mean_actions", outputs[0])
            actions = torch.zeros((env.num_envs, env._unwrapped.cfg.action_space))
            # env stepping
            obs, _, _, _, info = env.step(actions)
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()