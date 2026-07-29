"""
Script to evaluate a Unified Policy Framework.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent.")
parser.add_argument("--seed", type=int, default=None, help="Seed of RL environment")
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="G1-fall-unified-play", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default="/home/oksusu/Downloads/agent_32000.pt", help="Path to model checkpoint.")
parser.add_argument("--predictor_checkpoint", type=str, default="/home/oksusu/Downloads/ra_agent_16000.pt", help="Path to fall predictor model checkpoint.")
parser.add_argument("--safe_checkpoint", type=str, default="/home/oksusu/Downloads/agent_56000.pt", help="Path to safe model checkpoint.")
# parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing {low,mid,high}_risk.pt produced by collect.py.")

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
                    choices=["MLP", "Shared", "Superconnected", "Communet"],
                    help="The NN model used for training the agent.")

parser.add_argument("--safe_algorithm",
                    type=str,
                    default="MAPPO",
                    choices=["PPO", "SAC", "TD3", "MAPPO"],
                    help="The RL algorithm used for training the agent.")

parser.add_argument("--safe_model",
                    type=str,
                    default="Shared",
                    choices=["MLP", "Shared", "Superconnected", "Communet"],
                    help="The NN model used for training the agent.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
args_cli.headless = False                   # Show Isaac Sim UI
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
from lib.utils.parse_utils import parse_env_cfg, load_cfg_from_registry
from lib.buffer.rolloutbuffer import RolloutBuffer
from lib.model.model_factory import ModelFactory

# config shortcuts
algorithm = args_cli.algorithm.lower()
model = args_cli.model.lower() if args_cli.model is not None else None

safe_algorithm = args_cli.safe_algorithm.lower()
safe_model = args_cli.safe_model.lower() if args_cli.safe_model is not None else None

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
        safe_cfg = load_cfg_from_registry(args_cli.task, f"safe_rl_{safe_algorithm}_cfg_entry_point")
    except ValueError as e:
        print(e)
        return

    if args_cli.safe_checkpoint is not None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(args_cli.safe_checkpoint)), "Total")
    elif args_cli.checkpoint is not None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(args_cli.checkpoint)))
    else:
        # specify directory for logging experiments
        log_root_path = os.path.join("logs", cfg["agent"]["experiment"]["directory"])
        log_root_path = os.path.abspath(log_root_path)
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}"
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        print(f"[INFO] Exact experiment name requested from command line: {log_dir}")
        if cfg["agent"]["experiment"]["experiment_name"]:
            log_dir += f"_{cfg['agent']['experiment']['experiment_name']}"
        log_dir = os.path.join(log_root_path, log_dir)

    # ============================ Env & Wrapper Spawn ================================

    # Predictor selection
    predictor = args_cli.predictor
    pred_cfg = ra_cfg[predictor if predictor == "ra" else "safe_fall"]

    # Create isaac environment
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        cfg["agent"]["seed"] = args_cli.seed
        pred_cfg["agent"]["seed"] = args_cli.seed
        safe_cfg["agent"]["seed"] = args_cli.seed
    else:
        env_cfg.seed = cfg.get("seed", 42)
        cfg["agent"]["seed"] = cfg.get("seed", 42) # 42 is a default seed (equal to env)
        pred_cfg["agent"]["seed"] = cfg.get("seed", 42) # 42 is a default seed (equal to env)
        safe_cfg["agent"]["seed"] = cfg.get("seed", 42) # 42 is a default seed (equal to env)
    env_cfg.total_timesteps = cfg["train"]["timesteps"]
    env = gym.make(args_cli.task, cfg=env_cfg)

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
        from lib.buffer.replaybuffer import HindSightReplayBuffer
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

    # =============== Safe Policy Buffer and Model Spawn ===============
    safe_multi_agent = safe_algorithm == "mappo"
    safe_cfg["models"]["multi_agent"] = safe_multi_agent
    # Initialization
    if safe_cfg["buffer"]["buffer_size"] == -1:
        safe_cfg["buffer"]["buffer_size"] = safe_cfg["agent"]["rollouts"]
    else:
        raise RuntimeError("Replaybuffer for Off-policy algorithm is not implemented yet.")
    
    possible_agents = None
    if safe_multi_agent:
        safe_obs_size = {}
        safe_state_size = {}
        safe_act_size = {}
        safe_buffers = {}
        possible_agents = env._unwrapped.cfg.possible_agents
        for uid in possible_agents:
            observation_space = env._unwrapped.cfg.safe_observation_space[uid]
            action_space = env._unwrapped.cfg.safe_action_space[uid]
            state_space = env._unwrapped.cfg.safe_state_space[uid]
            safe_cfg["agent"]["async_actor_critic"] = True

            safe_buffer = RolloutBuffer(safe_cfg["buffer"]["buffer_size"], env.num_envs, device=env.device)
            safe_buffer.init_buffer(observation_space, state_space, action_space)
            safe_buffers[uid] = safe_buffer
            safe_obs_size[uid] = safe_buffer.tensors["observations"].shape[-1]
            safe_state_size[uid] = safe_buffer.tensors["states"].shape[-1]
            safe_act_size[uid] = safe_buffer.tensors["actions"].shape[-1]
    else:
        observation_space = env._unwrapped.safe_observation_space
        action_space = env._unwrapped.safe_action_space
        state_space = env._unwrapped.safe_state_space

        safe_buffer = RolloutBuffer(safe_cfg["buffer"]["buffer_size"], env.num_envs, device=env.device)
        safe_buffer.init_buffer(observation_space, state_space, action_space)
        safe_buffer = safe_buffer
        safe_obs_size = safe_buffer.tensors["observations"].shape[-1]
        safe_state_size = safe_buffer.tensors["states"].shape[-1]
        safe_act_size = safe_buffer.tensors["actions"].shape[-1]   

    # Overwrite cfg by cli argument
    if safe_model is not None:
        safe_cfg["models"]["model_type"] = safe_model

    safe_model_manager = ModelFactory(cfg=safe_cfg["models"], device=env.device)
    if safe_model_manager.model_class == "mlp":
        safe_models = safe_model_manager.generate_mlp_models(observation_size=safe_obs_size,
                                                             state_size=safe_state_size,
                                                             action_size=safe_act_size,
                                                             possible_agents=possible_agents)
    else:
        raise RuntimeError("Not supported class")

    # ======================= Safe Agent ============================
    safe_cfg["agent"]["action_scale_factor"] = env._unwrapped.cfg.action_scale_factor
    if safe_multi_agent:
        if safe_model_manager.model_type == "mlp":
            from lib.agent.mappo import MAPPO
            safe_agent = MAPPO(observation_space=env._unwrapped.safe_observation_space,
                               state_space=env._unwrapped.safe_state_space,
                               action_space=env._unwrapped.safe_action_space,
                               possible_agents=possible_agents,
                               model=safe_models,
                               buffer=safe_buffers,
                               device=env.device,
                               cfg=safe_cfg["agent"])
        elif safe_model_manager.model_type == "shared" or safe_model_manager.model_type == "superconnected":
            from lib.agent.cooperative_mappo import CooperativeMAPPO
            safe_agent = CooperativeMAPPO(observation_space=env._unwrapped.safe_observation_space,
                                          state_space=env._unwrapped.safe_state_space,
                                          action_space=env._unwrapped.safe_action_space,
                                          possible_agents=possible_agents,
                                          model=safe_models,
                                          buffer=safe_buffers,
                                          device=env.device,
                                          cfg=safe_cfg["agent"])
        else:
            raise RuntimeError("Unvalid model type.")
    else:
        from lib.agent.ppo import PPO
        safe_agent = PPO(model=safe_models,
                    buffer=safe_buffer, 
                    device=env.device,
                    cfg=safe_cfg["agent"])


    # ======================= Checkpoint Load ========================
    # Checkpoint (Policy)
    if args_cli.checkpoint is not None:
        resume_path = os.path.abspath(args_cli.checkpoint)
        agent.load(resume_path)
        print(f"[INFO] Get checkpoint of policy from {resume_path}.")
    else:
        print(f"[INFO] Unfortunately a pre-trained Policy is not found for this task.")
    # Checkpoint (Predictor)
    if args_cli.predictor_checkpoint is not None:
        resume_path_pred = os.path.abspath(args_cli.predictor_checkpoint)
        pred_agent.load(resume_path_pred)
        print(f"[INFO] Get checkpoint RA Value from {resume_path_pred}.")
    else:
        resume_path_pred = None
        print("[INFO] Unfortunately a pre-trained RA Value is not found for this task.")
    # Checkpoint (Safe Policy)
    if args_cli.safe_checkpoint is not None:
        resume_path_safe = os.path.abspath(args_cli.safe_checkpoint)
        safe_agent.load(resume_path_safe)
        print(f"[INFO] Get checkpoint Safety Policy from {resume_path_safe}.")
    else:
        resume_path_safe = None
        print("[INFO] Unfortunately a pre-trained Safety Policy is not found for this task.")


    # ======================= Evaluation ============================
    agent.set_running_mode("eval")
    obs, states, infos = env.reset()
    safe_obs = infos["safe_observations"]
    timestep = 0
    total_ep = 0
    success_ep = 0

    prev_switch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    switch_threshold = ra_cfg["collection"]["thresholds"]["mid_high"]

    # SafeFall predictor recurrent state; carried across steps within an episode,
    # zeroed per-env on done. None for RA (stateless MLP).
    pred_h = None
    safefall_threshold = 0.5
    if predictor == "safefall":
        n_layers = int(pred_cfg["model"].get("num_layers", 1))
        hidden_dim = int(pred_cfg["model"]["hidden_dim"])
        pred_h = torch.zeros(n_layers, env.num_envs, hidden_dim, device=env.device)
        safefall_threshold = float(pred_cfg.get("eval", {}).get("threshold", 0.5))

    # Simulate environment
    while simulation_app.is_running() and timestep <= cfg["train"]["timesteps"]:

        # ================== Interaction Phase =====================
        t1_loop = time.time()
        with torch.no_grad():
            # agent stepping
            nominal_actions, _, _, _ = agent.act(obs, infos, timestep=timestep, deterministic=True)
            safe_actions, _, _, _ = safe_agent.act(safe_obs, infos, timestep=timestep, deterministic=True)

            # Predictor forward
            if predictor == "ra":
                risk_value, _, _ = pred_agent.critic(infos["ra_states"])
                cur_switch = risk_value.float().reshape(-1) > switch_threshold
            else:  # safefall
                obs_in = infos["safe_fall_obs"].unsqueeze(1)            # (N, 1, obs_dim)
                logits, pred_h, _ = pred_agent.critic(obs_in, pred_h)   # h carried over
                prob = logits.softmax(dim=-1)[..., 1].reshape(-1)
                risk_value = prob.unsqueeze(-1)
                cur_switch = prob > safefall_threshold

            # safe action post-processing
            if not safe_multi_agent:
                # NOTE: Exceptionally, use env variables for assigning action with dictionary convention
                safe_actions_buffer = torch.zeros_like(safe_actions)
                safe_actions_buffer[:, :len(env._unwrapped.total_arm_joint_ids)] = safe_actions[:, env._unwrapped.total_arm_joint_ids]
                safe_actions_buffer[:, len(env._unwrapped.total_arm_joint_ids):] = safe_actions[:, env._unwrapped.total_leg_joint_ids]
                safe_actions = safe_actions_buffer

            switch = torch.logical_or(cur_switch, prev_switch)
            actions = torch.where(switch.unsqueeze(-1), safe_actions, nominal_actions)
            # env stepping
            next_obs, next_states, rewards, terminated, truncated, next_infos = env.step(actions)
            # update rollout number
            timestep += 1

        # Check terminated
        done = terminated[0] | truncated[0]

        # Per-env hidden state reset for SafeFall on episode boundary
        if pred_h is not None:
            done_mask = (terminated | truncated).reshape(-1)
            if done_mask.any():
                pred_h[:, done_mask, :] = 0.0

        # update
        if done:
            prev_switch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            total_ep += 1
            if not terminated[0]:
                success_ep += 1
        else:
            prev_switch = switch
        obs = next_obs
        states = next_states
        infos = next_infos
        safe_obs = infos["safe_observations"]

    # close the simulator
    env.close()

    # Print success rate
    print(f"Total Success Rate : {success_ep} / {total_ep}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()