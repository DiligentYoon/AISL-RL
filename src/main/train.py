"""
Script to train a RL agent.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="My-Cart-Double-Pendulum-Direct-Test", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")

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
args_cli.headless = True
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

from torch.utils.tensorboard import SummaryWriter


from datetime import datetime

import lib

from wrapper.isaaclab_wrapper import IsaacLabWrapper
from lib.utils.parse_utils import parse_env_cfg, load_cfg_from_registry
from lib.buffer.rolloutbuffer import RolloutBuffer
from lib.agent.ppo import PPO
from lib.agent.mappo import MAPPO

from lib.model.MLP import Actor, Critic

from lib.model.NerveNet import NerveNetPolicy

from lib.utils.graph_utils import Mapping
from lib.model.BodyTransformer.body_transformer import BodyLevelActor, BodyLevelCritic
from lib.model.BodyTransformer.linear_components import ObsTokenizer, ValueDetokenizer, ActionDetokenizer
from lib.model.BodyTransformer.transformer_components import BodyTransformer

# config shortcuts
algorithm = args_cli.algorithm.lower()

def main():
    """
    main training method
    """

    # ============================= Config Parsing ===============================
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    try:
        cfg = load_cfg_from_registry(args_cli.task, f"rl_{algorithm}_cfg_entry_point")
    except ValueError as e:
        print(e)
        return

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}"
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Exact experiment name requested from command line: {log_dir}")
    if cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{cfg['agent']['experiment']['experiment_name']}"
    log_dir = os.path.join(log_root_path, log_dir)

    # ============================ Env & Wrapper Spawn ================================

    # Create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # Wrap around environment
    env = IsaacLabWrapper(env)  

    if cfg["agent"]["experiment"]["write_interval"] == "auto":
        write_interval = int(cfg["train"]["timesteps"] / 100)
    if cfg["agent"]["experiment"]["checkpoint_interval"] == "auto":
        checkpoint_interval = int(cfg["train"]["timesteps"] / 5)


    # ======================= Buffer =========================

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

    # =================== Model & Agent Spawn Test ==========================
    
    # Model initialization
    is_shared = False
    model_type = cfg["models"]["policy"].get("type", None)
    if multi_agent:
        model = {}
        if model_type is not None:
            raise RuntimeError("MARL With CTDE structure only supports a MLP network.")

        for uid in possible_agents:
            # Per-Agent Network
            actor = Actor(num_observations=obs_size[uid],
                          num_actions=act_size[uid],
                          min_log_std=cfg["models"]["policy"]["min_log_std"],
                          max_log_std=cfg["models"]["policy"]["max_log_std"],
                          device=env.device)
            
            critic = Critic(num_states=state_size[uid],
                            device=env.device)
            
            model[uid] = {
                'actor': actor,
                'critic': critic
            }
        
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
                is_shared = cfg["models"].get("shared", False)
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
    
    # Checkpoint
    if args_cli.checkpoint is not None:
        resume_path = os.path.abspath(args_cli.checkpoint)
        agent.load(resume_path)
        print(f"[INFO] Get checkpoint from {resume_path}")
    else:
        print("[INFO] Unfortunately a pre-trained checkpoint is not found for this task.")
        resume_path = None

    # ======================= Training ============================

    # Tensorboard Wrtier
    writer = SummaryWriter(log_dir=log_dir)
    cumulative_rewards = None
    cumulative_timesteps = None
    tracking_data = collections.defaultdict(list)
    track_rewards = collections.deque(maxlen=100)
    track_timesteps = collections.deque(maxlen=100)
    t1_rollout = time.time()
    t2_rollout = 0
    t1_update = 0
    t2_update = 0
    CLI_track_rewards = []
    CLI_track_timesteps = []
    CLI_step_reward_means = []

    # Reset environment
    obs, states, _ = env.reset()
    timestep = 0
    rollout = 0
    elapsed_time = 0
    
    # Simulate environment
    while simulation_app.is_running() and timestep <= cfg["train"]["timesteps"]:

        # ================== Interaction Phase =====================
        with torch.no_grad():
            # agent stepping
            actions, action_log_probs, _ = agent.act(obs, timestep=timestep, deterministic=False)
            # env stepping
            next_obs, next_states, rewards, terminated, truncated, infos = env.step(actions)
            # update rollout number
            timestep += 1
        
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
        if timestep % buffer.buffer_size == 0:
            if buffer.memory_index == 0:
                t2_rollout = time.time()

                t1_update = time.time()
                policy_loss, value_loss, entropy_loss, approx_kl = agent.update()
                t2_update = time.time()

                rollout += 1
            else:
                raise RuntimeError("Discrepency appears between Buffer Logic and Rollout Policy.")
            
        # =============== Logging Phase ================

        # Data setting for logging
        if cumulative_rewards is None:
            cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
            cumulative_timesteps = torch.zeros_like(rewards, dtype=torch.int32)
        
        cumulative_rewards.add_(rewards)
        cumulative_timesteps.add_(1)
        CLI_step_reward_means.append(torch.mean(rewards).item())

        finished_episodes = (terminated + truncated).nonzero(as_tuple=False)
        if finished_episodes.numel():
            # storage cumulative rewards and timesteps
            track_rewards.extend(cumulative_rewards[finished_episodes][:, 0].reshape(-1).tolist())
            track_timesteps.extend(cumulative_timesteps[finished_episodes][:, 0].reshape(-1).tolist())
            CLI_track_rewards.extend(cumulative_rewards[finished_episodes][:, 0].detach().cpu().tolist())
            CLI_track_timesteps.extend(cumulative_timesteps[finished_episodes][:, 0].detach().cpu().tolist())
            # reset the cumulative rewards and timesteps
            cumulative_rewards[finished_episodes] = 0
            cumulative_timesteps[finished_episodes] = 0

        # record data
        tracking_data["Reward / Instantaneous reward (max)"].append(torch.max(rewards).item())
        tracking_data["Reward / Instantaneous reward (min)"].append(torch.min(rewards).item())
        tracking_data["Reward / Instantaneous reward (mean)"].append(torch.mean(rewards).item())

        if len(track_rewards):
            track_reward_np = np.array(track_rewards)
            track_timestep_np = np.array(track_timesteps)

            tracking_data["Reward / Total reward (max)"].append(np.max(track_reward_np))
            tracking_data["Reward / Total reward (min)"].append(np.min(track_reward_np))
            tracking_data["Reward / Total reward (mean)"].append(np.mean(track_reward_np))

            tracking_data["Episode / Total timesteps (max)"].append(np.max(track_timestep_np))
            tracking_data["Episode / Total timesteps (min)"].append(np.min(track_timestep_np))
            tracking_data["Episode / Total timesteps (mean)"].append(np.mean(track_timestep_np))
        
        # Tensorboard logging
        if timestep % write_interval == 0: 
            for k, v in tracking_data.items():
                if k.endswith("(min)"):
                    writer.add_scalar(k, np.min(v), timestep)
                elif k.endswith("(max)"):
                    writer.add_scalar(k, np.max(v), timestep)
                else:
                    writer.add_scalar(k, np.mean(v), timestep)
            # reset data containers for next iteration
            track_rewards.clear()
            track_timesteps.clear()
            tracking_data.clear()

        # CLI Logging about the training process at each parameter update
        if timestep % buffer.buffer_size == 0 and buffer.memory_index == 0:
            per_step_reward = float(np.mean(CLI_step_reward_means)) if len(CLI_step_reward_means) else float("nan")
            avg_ep_step = float(np.mean(CLI_track_timesteps)) if len(CLI_track_timesteps) else float("nan")
            avg_ep_reward = float(np.mean(CLI_track_rewards)) if len(CLI_track_rewards) else float("nan")

            ep_step = "-" if np.isnan(avg_ep_step) else f"{avg_ep_step:6.2f} steps"
            per_r = "-" if np.isnan(per_step_reward) else f"{per_step_reward:6.2f}"
            ep_r = "-" if np.isnan(avg_ep_reward) else f"{avg_ep_reward:6.2f}"

            elapsed_time += (t2_rollout + t2_update - t1_rollout - t1_update)
            e_h = int(elapsed_time // 3600)
            e_m = int((elapsed_time % 3600) // 60)
            e_s = int(elapsed_time % 60)
            total_rollout = int(cfg["train"]["timesteps"] // buffer.buffer_size)
            complete_time = (t2_rollout + t2_update - t1_rollout - t1_update) * (total_rollout - rollout)
            c_h = int(complete_time // 3600)
            c_m = int((complete_time % 3600) // 60)
            c_s = int(complete_time % 60)

            content_width = 64
            line_header = f"Step Progress {timestep} / {cfg['train']['timesteps']}"
            line_time_header = f"Time Progress  {e_h:02d}:{e_m:02d}:{e_s:02d}/{c_h:02d}:{c_m:02d}:{c_s:02d}"
            line_rollout_time = f"Rollout Time      : {t2_rollout - t1_rollout:6.2f} sec"
            line_train_time = f"Training Time     : {t2_update - t1_update:6.2f} sec"
            line_value_loss = f"Value Loss        : {value_loss:6.2f}"
            line_policy_loss = f"Policy Loss       : {policy_loss:6.2f}"
            line_entropy_loss = f"Entropy Loss      : {entropy_loss:6.2f}"
            line_episode_step = f"Avg Episode Step  : {ep_step}"
            line_per_step_reward = f"Per-Step Rewards  : {per_r}"
            line_episode_reward = f"Epiode Rewards    : {ep_r}"
            # line_episode_success = f"Avg Success Rate  : {100 * global_success_rate:6.2f} %"

            print(f" ________________________________________________________________")
            print(f"|                                                                |")
            print(f"|{line_header.center(content_width)}|")
            print(f"|{line_time_header.center(content_width)}|")
            print(f"|________________________________________________________________|")
            print(f"|                                                                |")
            print(f"| {line_rollout_time:<{content_width-1}}|")
            print(f"| {line_train_time:<{content_width-1}}|")
            print(f"| {line_value_loss:<{content_width-1}}|")
            print(f"| {line_policy_loss:<{content_width-1}}|")
            print(f"| {line_entropy_loss:<{content_width-1}}|")
            print(f"| {line_episode_step:<{content_width-1}}|")
            print(f"| {line_per_step_reward:<{content_width-1}}|")
            print(f"| {line_episode_reward:<{content_width-1}}|")
            # print(f"| {line_episode_success:<{content_width-1}}|")
            print(f"|________________________________________________________________|")

            # update rollout time
            t1_rollout = time.time()

        # Checkpoint save
        if timestep % checkpoint_interval == 0:
            checkpoint_path = os.path.join(log_dir, f"agent_{timestep}.pt")
            agent.save(checkpoint_path)

        # update
        obs = next_obs
        states = next_states

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()