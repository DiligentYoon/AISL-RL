"""
Script to train a Reach-avoid value function.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent.")
parser.add_argument("--seed", type=int, default=None, help="Seed of RL environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="G1-fall-play", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint.")
parser.add_argument("--predictor_checkpoint", type=str, default=None, help="Path to fall predictor model checkpoint.")

parser.add_argument("--predictor",
                    type=str,
                    default="ra",
                    choices=["ra", "safefall"],
                    help="Fall predictor type to train.")

parser.add_argument("--algorithm",
                    type=str,
                    default="MAPPO",
                    choices=["PPO", "SAC", "TD3", "MAPPO"],
                    help="The RL algorithm used for training the agent.")

parser.add_argument("--model",
                    type=str,
                    default="Shared",
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
from isaaclab.envs.common import ViewerCfg
import lib

from wrapper.isaaclab_wrapper import IsaacLabWrapper
from wrapper.record_wrapper import RecordVideo
from lib.utils.parse_utils import parse_env_cfg, load_cfg_from_registry
from lib.buffer.rolloutbuffer import RolloutBuffer
from lib.model.model_factory import ModelFactory
from lib.utils.plot_utils import GIFSavePlotter

# config shortcuts
algorithm = args_cli.algorithm.lower()
model = args_cli.model.lower() if args_cli.model is not None else None

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
        ra_cfg = load_cfg_from_registry(args_cli.task, f"ra_cfg_entry_point")
    except ValueError as e:
        print(e)
        return

    # specify directory for logging experiments (load checkpoint)
    if args_cli.predictor_checkpoint is not None:
        log_dir = os.path.dirname(args_cli.predictor_checkpoint)
    else:
        raise ValueError("Predictor checkpoint path must be assigned for policy-conditioned RA value function.")

    # ============================ Env & Wrapper Spawn ================================

    # Predictor selection
    predictor = args_cli.predictor
    pred_cfg = ra_cfg[predictor if predictor == "ra" else "safe_fall"]

    # Create isaac environment
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        cfg["agent"]["seed"] = args_cli.seed
        pred_cfg["agent"]["seed"] = args_cli.seed
    else:
        env_cfg.seed = cfg.get("seed", None)
        cfg["agent"]["seed"] = cfg.get("seed", 42) # 42 is a default seed (equal to env)
        pred_cfg["agent"]["seed"] = cfg.get("seed", 42) # 42 is a default seed (equal to env)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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

    # Get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # Wrap around environment
    env = IsaacLabWrapper(env)  
    
    # ======================= Buffer =========================
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

    # ====================== Model Spawn  ==========================
    # Overwrite cfg by cli argument
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
        

    # ============= Fall Predictor Buffer/Model/Agent Spawn ===============
    if predictor == "ra":
        from lib.buffer.reach_avoid.replaybuffer import HindSightReplayBuffer
        from lib.model.MLP import RA_Critic
        from lib.agent.reach_avoid import ReachAvoid

        if not hasattr(env._unwrapped.cfg, "ra_state_space"):
            raise RuntimeError("Explicit state space is not defined.")

        pred_buffer = HindSightReplayBuffer(pred_cfg["buffer"]["buffer_size"],
                                            env.num_envs, device=env.device)
        pred_buffer.init_buffer(env._unwrapped.cfg.ra_state_space)
        pred_model = {"critic": RA_Critic(env._unwrapped.cfg.ra_state_space, env.device)}
        pred_agent = ReachAvoid(pred_model, pred_buffer,
                                device=env.device, cfg=pred_cfg["agent"])

    elif predictor == "safefall":
        from lib.model.Baselines.SafeFall.safe_fall import GRU
        from lib.agent.Baselines.safe_fall import SafeFall

        if not hasattr(env._unwrapped.cfg, "safe_fall_obs_dim"):
            raise RuntimeError("safe_fall_obs_dim is not defined in env cfg.")

        obs_dim = env._unwrapped.cfg.safe_fall_obs_dim
        # Inference-only model holder; trained offline via collect_offline.py + train_offline.py.
        pred_model = {"critic": GRU(obs_dim=obs_dim,
                                    hidden_dim=pred_cfg["model"]["hidden_dim"],
                                    num_layers=pred_cfg["model"].get("num_layers", 1),
                                    dropout=pred_cfg["model"].get("dropout", 0.0))}
        pred_agent = SafeFall(pred_model, device=env.device, cfg=pred_cfg["agent"])

    else:
        raise ValueError(f"Unknown predictor: {predictor}")
    
    # Checkpoint (Policy)
    resume_path = os.path.abspath(args_cli.checkpoint)
    agent.load(resume_path)
    print(f"[INFO] Get checkpoint of policy from {resume_path}")

    # Checkpoint (Predictor)
    if args_cli.predictor_checkpoint is not None:
        resume_path_pred = os.path.abspath(args_cli.predictor_checkpoint)
        pred_agent.load(resume_path_pred)
        print(f"[INFO] Get predictor checkpoint ({predictor}) from {resume_path_pred}")
    else:
        resume_path_pred = None
        print(f"[INFO] No pre-trained predictor ({predictor}) checkpoint found.")

    # ======================= Evaluation ============================
    
    # Reset environment
    plotter_cls = getattr(env._unwrapped.cfg, "plotter", None)
    if plotter_cls is not None:
        plot_cfg = env._unwrapped.cfg.viz_data
        plot_dir = os.path.join(log_dir, "plot") if args_cli.checkpoint else None
        plot: GIFSavePlotter = plotter_cls(env, plot_cfg, plot_dir)
    else:
        plot = None

    agent.set_running_mode("eval")
    obs, states, infos = env.reset()
    timestep = 0

    # SafeFall predictor recurrent state; carried across steps within an episode,
    # zeroed per-env on done. None for RA (stateless MLP).
    pred_h = None
    if predictor == "safefall":
        n_layers = int(pred_cfg["model"].get("num_layers", 1))
        hidden_dim = int(pred_cfg["model"]["hidden_dim"])
        pred_h = torch.zeros(n_layers, env.num_envs, hidden_dim, device=env.device)

    # Simulate environment
    while simulation_app.is_running() and timestep <= cfg["train"]["timesteps"]:

        # ================== Interaction Phase =====================
        with torch.no_grad():
            # agent stepping
            actions, _, _ = agent.act(obs, infos, timestep=timestep, deterministic=True)

            # Predictor forward
            if predictor == "ra":
                risk_value, _, _ = pred_agent.critic(infos["ra_states"])
            else:  # safefall
                obs_in = infos["safe_fall_obs"].unsqueeze(1)            # (N, 1, obs_dim)
                logits, pred_h, _ = pred_agent.critic(obs_in, pred_h)
                risk_value = logits.softmax(dim=-1)[..., 1].reshape(-1, 1)

            # env stepping
            next_obs, next_states, rewards, terminated, truncated, next_infos = env.step(actions)
            # update rollout number
            timestep += 1

        # Per-env hidden state reset for SafeFall on episode boundary
        if pred_h is not None:
            done_mask = (terminated | truncated).reshape(-1)
            if done_mask.any():
                pred_h[:, done_mask, :] = 0.0

        # Plot Phase
        if plot is not None:
            done = terminated[0] | truncated[0]
            risk_value = risk_value.float().squeeze(-1)
            infos["viz_data"]["risk_value"] = risk_value
            plot.append(viz_data=infos["viz_data"], episode_end=done)

        # Video update
        if timestep == args_cli.video_length:
            # exit the play loop after recording one video
            break

        # update
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