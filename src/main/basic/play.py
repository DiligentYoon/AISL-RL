"""
Script to play a checkpoint of an RL agent.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent.")
parser.add_argument("--seed", type=int, default=None, help="Seed of RL environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="G1-recovery-play", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")

parser.add_argument("--algorithm",
                    type=str,
                    default="PPO",
                    choices=["PPO", "SAC", "TD3", "MAPPO"],
                    help="The RL algorithm used for training the agent.")

parser.add_argument("--model",
                    type=str,
                    default="MLP",
                    choices=["MLP", "Shared", "Communet"],
                    help="The NN model used for training the agent.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
args_cli.headless = True                    # Headless mode
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import copy
import numpy as np
import collections

from datetime import datetime

import lib

from lib.utils.parse_utils import parse_env_cfg, load_cfg_from_registry
from lib.utils.plot_utils import GIFSavePlotter
from wrapper.isaaclab_wrapper import IsaacLabWrapper
from wrapper.record_wrapper import RecordVideo

# config shortcuts
algorithm = args_cli.algorithm.lower()
model = args_cli.model.lower() if args_cli.model is not None else None

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
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        cfg["agent"]["seed"] = args_cli.seed
    else:
        env_cfg.seed = cfg.get("seed", None)
        cfg["agent"]["seed"] = cfg.get("seed", 42) # 42 is a default seed (equal to env)
    env_cfg.total_timesteps = cfg["train"]["timesteps"]
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Logging directory
    if args_cli.checkpoint is not None:
        log_dir = os.path.dirname(args_cli.checkpoint)
    else:
        log_root_path = os.path.join("logs", cfg["agent"]["experiment"]["directory"])
        log_root_path = os.path.abspath(log_root_path)
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}"
        log_dir += f"_{cfg['agent']['experiment']['experiment_name']}"
        log_dir = os.path.join(log_root_path, log_dir)

    # wrap for video recording
    if args_cli.video:
        args_cli.video_interval = int(cfg["train"]["timesteps"] / 5)
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = RecordVideo(env, **video_kwargs)

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
        num_agent = len(possible_agents)
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
    if model is not None:
        cfg["models"]["model_type"] = model
    
    model_manager = ModelFactory(cfg=cfg["models"], device=env.device)
    if model_manager.model_class == "mlp":
        models = model_manager.generate_mlp_models(observation_size=obs_size,
                                                   state_size=state_size,
                                                   action_size=act_size,
                                                   possible_agents=possible_agents)
    else:
        raise RuntimeError("Not supported class")

    # ====================== Agent Spawn  ==========================
    # Scale Factor
    cfg["agent"]["action_scale_factor"] = env._unwrapped.cfg.action_scale_factor
    if multi_agent:
        if model_manager.model_type == "mlp":
            from lib.agent.mappo import MAPPO
            agent = MAPPO(observation_space=env.observation_space,
                          state_space=env.state_space,
                          action_space=env.action_space,
                          possible_agents=possible_agents,
                          model=models,
                          buffer=buffers,
                          device=env.device,
                          cfg=cfg["agent"])
            
        elif model_manager.model_type == "communet":
            from lib.agent.communet_mappo import CommunetMAPPO
            agent = CommunetMAPPO(observation_space=env.observation_space,
                                    state_space=env.state_space,
                                    action_space=env.action_space,
                                    possible_agents=possible_agents,
                                    model=models,
                                    buffer=buffers,
                                    device=env.device,
                                    cfg=cfg["agent"])
        
        elif model_manager.model_type == "shared":
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
            raise RuntimeError("Unvalid model type.")

    else:
        from lib.agent.ppo import PPO
        agent = PPO(model=models,
                    buffer=buffer, 
                    device=env.device,
                    cfg=cfg["agent"])
    
    # Checkpoint
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
    plotter_cls = getattr(env._unwrapped.cfg, "plotter", None)
    if plotter_cls is not None:
        plot_cfg = env._unwrapped.cfg.viz_data
        plot_dir = os.path.join(log_dir, "plot") if args_cli.checkpoint else None
        plot: GIFSavePlotter = plotter_cls(env, plot_cfg, plot_dir)
    else:
        plot = None

    agent.set_running_mode("eval")
    obs, states, infos = env.reset()
    write_interval = int(env._unwrapped.cfg.episode_length_s / (env._unwrapped.cfg.sim_dt * env._unwrapped.cfg.decimation))
    timestep = 0
    tracking_data = collections.defaultdict(list)
    track_cumulative_rewards   = collections.deque(maxlen=500)
    track_cumulative_timesteps = collections.deque(maxlen=500)
    cumulative_rewards = None
    cumulative_timesteps = None
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.no_grad():
            # agent stepping
            actions, _, _, _ = agent.act(obs, infos, timestep=timestep, deterministic=True)
            # env stepping
            next_obs, next_states, rewards, terminated, truncated, next_infos = env.step(actions)
            # update rollout number
            timestep += 1

        # ============ Logging phase =============

        # Data setting for logging
        if multi_agent:
            logged_reward = rewards.view(-1, num_agent)
            logged_reward = torch.mean(logged_reward, dim=-1).unsqueeze(-1) # Mean value of agent axis
        else:
            logged_reward = rewards

        if cumulative_rewards is None:
            cumulative_rewards = torch.zeros_like(logged_reward, dtype=torch.float32)
            cumulative_timesteps = torch.zeros_like(logged_reward, dtype=torch.int32)

        # Record data
        tracking_data["Per step Reward / Maximum Value"].append(torch.max(logged_reward).item())
        tracking_data["Per step Reward / Minimum Value"].append(torch.min(logged_reward).item())
        tracking_data["Per step Reward / Mean Value"].append(torch.mean(logged_reward).item())

        # Accumulates per-step rewards
        cumulative_rewards.add_(logged_reward)
        cumulative_timesteps.add_(1)

        done = (terminated | truncated).squeeze(-1)
        finished_episodes = done.nonzero(as_tuple=False).squeeze(-1)
        if finished_episodes.numel():
            # Storage cumulative rewards and timesteps
            track_cumulative_rewards.extend(cumulative_rewards[finished_episodes][:, 0].reshape(-1).tolist())
            track_cumulative_timesteps.extend(cumulative_timesteps[finished_episodes][:, 0].reshape(-1).tolist())
            # Reset the cumulative rewards and timesteps
            cumulative_rewards[finished_episodes] = 0
            cumulative_timesteps[finished_episodes] = 0

        # Record cumulative data
        if len(track_cumulative_rewards):
            track_reward_np = np.array(track_cumulative_rewards)
            track_timestep_np = np.array(track_cumulative_timesteps)

            tracking_data["Total Reward / Maximum Value"].append(np.max(track_reward_np))
            tracking_data["Total Reward / Minimum Value"].append(np.min(track_reward_np))
            tracking_data["Total Reward / Mean Value"].append(np.mean(track_reward_np))

            tracking_data["Total Episode / Maximum Value"].append(np.max(track_timestep_np))
            tracking_data["Total Episode / Minimum Value"].append(np.min(track_timestep_np))
            tracking_data["Total Episode / Mean Value"].append(np.mean(track_timestep_np))

            # reset data containers for next iteration
            track_cumulative_rewards.clear()
            track_cumulative_timesteps.clear()

        # Record Task Reward data (Only per step)
        task_reward = next_infos.get("reward", None)
        if task_reward is not None:
            for k, v in task_reward.items():
                # Mean value of env axis
                key = f"Per step {k}"
                tracking_data[key].append(torch.mean(v).item())

        # CLI Logging with specific interval
        if timestep % write_interval == 0:
            last_prefix = None
            # sorted_keys = sorted(tracking_data.keys())
            content_width = 90
            line_header = f"Metric Table of {args_cli.task} Task"
            print(f" {'_' * content_width}")
            print(f"|{' ' * content_width}|")
            print(f"|{line_header.center(content_width)}|")
            print(f"|{'_' * content_width}|")
            print(f"|{' ' * content_width}|")

            for k, v in tracking_data.items():
                current_prefix = k.split('/')[0].strip() if '/' in k else "Other"

                if last_prefix is not None and last_prefix != current_prefix:
                    print(f"|{'-' * (content_width)}|")
                last_prefix = current_prefix

                if k.endswith("(min)"):
                    print(f"| {k:<50}: {np.min(v):<36.3f} |")
                elif k.endswith("(max)"):
                    print(f"| {k:<50}: {np.max(v):<36.3f} |")
                else:
                    print(f"| {k:<50}: {np.mean(v):<36.3f} |")
            print(f"|{'_' * content_width}|")

            # reset data containers for next iteration
            tracking_data.clear()

        # Plot Phase
        if plot is not None:
            done = terminated[0] | truncated[0]
            plot.append(viz_data=infos["viz_data"], episode_end=done)


        # Video update
        if args_cli.video and timestep == args_cli.video_length:
            # exit the play loop after recording one video
            break
        
        # state update
        obs = next_obs
        states = next_states
        infos = next_infos


    # close the simulator
    env.close()

    # close and save GIF plotter
    if plot is not None:
        plot.save()
        plot.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()