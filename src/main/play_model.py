# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
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
from lib.model.MLP import Actor,Critic

# config shortcuts
algorithm = args_cli.algorithm.lower()

def main():
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
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    # runner = Runner(env, experiment_cfg)

    actor = Actor(num_observations=10,
                  num_actions=10,
                  device="cpu")
    
    critic = Critic(num_states=10,
                    device="cpu")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():

            actions = torch.rand((env.num_envs, 10))
            observations = torch.rand((env.num_envs, 10))
            states = torch.rand((env.num_envs, 10))

            x,_ = actor.forward(inputs=observations, eval=False)
            y,_ = critic.forward(inputs=states, eval=False)
            
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