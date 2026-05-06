"""
Script to train a Robust Adversarial RL(RARL) agent.
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
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="G1-pusher", help="Name of the task.")
parser.add_argument("--checkpoint_nominal", type=str, default="/home/oksusu/Downloads/agent_32000.pt", help="Path to nominal agent model checkpoint.")
parser.add_argument("--checkpoint_adversarial", type=str, default=None, help="Path to adversarial agent model checkpoint.")

parser.add_argument("--algorithm_nominal",
                    type=str,
                    default="MAPPO",
                    choices=["PPO", "SAC", "TD3", "MAPPO"],
                    help="The RL algorithm used for training the nominal agent.")

parser.add_argument("--algorithm_adversarial",
                    type=str,
                    default="MAPPO",
                    choices=["PPO", "SAC", "TD3", "MAPPO"], 
                    help="The RL algorithm used for training the adversarial agent.")

parser.add_argument("--model_nominal",
                    type=str,
                    default="Shared",
                    choices=["MLP", "Shared", "Superconnected", "Communet"],
                    help="The NN model used for training the nominal agent.")

parser.add_argument("--model_adversarial",
                    type=str,
                    default="MLP",
                    choices=["MLP"],
                    help="The NN model used for training the adversarial agent.")

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

from torch.utils.tensorboard import SummaryWriter


from datetime import datetime

import lib

from wrapper.isaaclab_wrapper import IsaacLabWrapper
from wrapper.record_wrapper import RecordVideo
from lib.utils.parse_utils import parse_env_cfg, load_cfg_from_registry
from lib.buffer.rolloutbuffer import RolloutBuffer
from lib.model.model_factory import ModelFactory

# config shortcuts
algorithm_nominal = args_cli.algorithm_nominal.lower()
algorithm_adversarial = args_cli.algorithm_adversarial.lower()

model_nominal = args_cli.model_nominal.lower() if args_cli.model_nominal is not None else None
model_adversarial = args_cli.model_adversarial.lower() if args_cli.model_adversarial is not None else None

def main():
    """
    main training method
    """

    # ============================= Config Parsing ===============================
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    try:
        cfg_nominal = load_cfg_from_registry(args_cli.task, f"rl_{algorithm_nominal}_nominal_cfg_entry_point")                      # NOTE: Nominal cfg is main
        cfg_adversarial = load_cfg_from_registry(args_cli.task, f"rl_{algorithm_adversarial}_adv_cfg_entry_point")
    except ValueError as e:
        print(e)
        return

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", cfg_nominal["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "RARL"
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Exact experiment name requested from command line: {log_dir}")
    if cfg_nominal["agent"]["experiment"]["experiment_name"]:
        log_dir_nominal = log_dir + f"_{cfg_nominal['agent']['experiment']['experiment_name']}"
    else:
        log_dir_nominal = log_dir
    log_dir_nominal = os.path.join(log_root_path, log_dir_nominal)

    # ============================ Env & Wrapper Spawn ================================

    # Create isaac environment
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        cfg_nominal["agent"]["seed"] = args_cli.seed
    else:
        env_cfg.seed = cfg_nominal.get("seed", None)
        cfg_nominal["agent"]["seed"] = cfg_nominal.get("seed", 42)                                              # 42 is a default seed (equal to env)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        args_cli.video_interval = int(cfg_nominal["train"]["timesteps"] / 5)
        video_kwargs = {
            "video_folder": os.path.join(log_dir_nominal, "videos", "train"),
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

    if cfg_nominal["agent"]["experiment"]["write_interval"] == "auto":
        write_interval_nominal = int(cfg_nominal["train"]["timesteps"] / 100)
    if cfg_nominal["agent"]["experiment"]["checkpoint_interval"] == "auto":
        checkpoint_interval_nominal = int(cfg_nominal["train"]["timesteps"] / 10)



    # ======================= MDP =========================
    multi_agent = algorithm_nominal == "mappo"
    cfg_nominal["models"]["multi_agent"] = multi_agent
    # Initialization
    if cfg_nominal["buffer"]["buffer_size"] == -1:
        cfg_nominal["buffer"]["buffer_size"] = cfg_nominal["agent"]["rollouts"]
    else:
        raise RuntimeError("Replaybuffer for Off-policy algorithm is not implemented yet.")
    
    possible_agents = None
    state_space = None
    nominal_update_turn = True                                                  # Update flag
    obs_size = {}
    state_size = {}
    act_size = {}
    buffers = {}
    possible_agents = env._unwrapped.cfg.possible_agents
    adversarial_agents = env._unwrapped.cfg.adversarial_agents                  # Extract adversarial agent
    nominal_agents = []
    num_agent = len(possible_agents)
    for uid in possible_agents:
        observation_space = env.observation_space[uid]
        action_space = env.action_space[uid]
        if env.state_space:
            if uid in env.state_space.keys():                                   # Adversarial agent may not be async AC
                state_space = env.state_space[uid]
            cfg_nominal["agent"]["async_actor_critic"] = True
        else:
            state_space = None
            cfg_nominal["agent"]["async_actor_critic"] = False
        
        if not uid in adversarial_agents:                                       # Extract nominal agent
            nominal_agents.append(uid)
        
        buffer = RolloutBuffer(cfg_nominal["buffer"]["buffer_size"], env.num_envs, device=env.device)
        buffer.init_buffer(observation_space, state_space, action_space)
        buffers[uid] = buffer
        obs_size[uid] = buffer.tensors["observations"].shape[-1]
        state_size[uid] = buffer.tensors["states"].shape[-1] if env.state_space else obs_size[uid]
        act_size[uid] = buffer.tensors["actions"].shape[-1]

### ========================================= Nominal Agent ========================================= ###
    # ====================== Model Spawn  ==========================
    # Overwrite cfg by cli argument
    if model_nominal is not None:
        cfg_nominal["models"]["model_type"] = model_nominal
    
    model_manager = ModelFactory(cfg=cfg_nominal["models"], device=env.device)
    if model_manager.model_class == "mlp":
        nominal_models = model_manager.generate_mlp_models(observation_size=obs_size,
                                                           state_size=state_size,
                                                           action_size=act_size,
                                                           possible_agents=nominal_agents)
    elif model_manager.model_class == "gnn":
        node_cfg = None
        mapping_cfg = None
        if model_manager.model_type == "nervenet":
            node_cfg = {'node_info': env._unwrapped.cfg.node_info,
                        'num_nodes': env._unwrapped.cfg.num_nodes,
                        'num_actuated_nodes': env._unwrapped.cfg.num_actuated_nodes}
            
        elif model_manager.model_type == "bodytransformer":
            mapping_cfg = env._unwrapped.cfg.map_info

        else:
            raise RuntimeError("Not supported type")
        
        nominal_models = model_manager.generate_gnn_models(observation_space=observation_space,
                                                           state_space=state_space,
                                                           action_space=action_space,
                                                           node_cfg=node_cfg,
                                                           mapping_cfg=mapping_cfg)
    else:
        raise RuntimeError("Not supported class")

    # ====================== Agent Spawn  ==========================
    # Scale Factor
    cfg_nominal["agent"]["action_scale_factor"] = env._unwrapped.cfg.action_scale_factor
    if multi_agent:
        if model_manager.model_type == "mlp":
            from lib.agent.mappo import MAPPO
            nominal_agent = MAPPO(observation_space=env.observation_space,
                                  state_space=env.state_space,
                                  action_space=env.action_space,
                                  possible_agents=nominal_agents,
                                  model=nominal_models,
                                  buffer=buffers,
                                  device=env.device,
                                  cfg=cfg_nominal["agent"])
            
        elif model_manager.model_type == "communet":
            from lib.agent.communet_mappo import CommunetMAPPO
            nominal_agent = CommunetMAPPO(observation_space=env.observation_space,
                                          state_space=env.state_space,
                                          action_space=env.action_space,
                                          possible_agents=nominal_agents,
                                          model=nominal_models,
                                          buffer=buffers,
                                          device=env.device,
                                          cfg=cfg_nominal["agent"])
        
        elif model_manager.model_type == "shared" or model_manager.model_type == "superconnected":
            from lib.agent.cooperative_mappo import CooperativeMAPPO
            nominal_agent = CooperativeMAPPO(observation_space=env.observation_space,
                                             state_space=env.state_space,
                                             action_space=env.action_space,
                                             possible_agents=nominal_agents,
                                             model=nominal_models,
                                             buffer=buffers,
                                             device=env.device,
                                             cfg=cfg_nominal["agent"])
        
        else:
            raise RuntimeError("Unvalid model type.")

    else:
        # from lib.agent.ppo import PPO
        nominal_agent = MAPPO(observation_space=env.observation_space,
                              state_space=env.state_space,
                              action_space=env.action_space,
                              possible_agents=nominal_agents,
                              model=nominal_models,
                              buffer=buffers,
                              device=env.device,
                              cfg=cfg_nominal["agent"])
    
    # Checkpoint
    if args_cli.checkpoint_nominal is not None:
        resume_path = os.path.abspath(args_cli.checkpoint_nominal)
        nominal_agent.load(resume_path)
        print(f"[INFO] Get checkpoint from {resume_path}")
    else:
        print("[INFO] Unfortunately a pre-trained checkpoint is not found for this task.")
        resume_path = None
    
    # Verify save logic
    verify_save_logic = True
    if verify_save_logic:
        test_nominal_agent = copy.deepcopy(nominal_agent)

### ========================================= Adversarial Agent ========================================= ###
    
    adversarial_agents = cfg_adversarial["agent"]["agents_name"]
    # Specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", cfg_adversarial["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading adversarial experiment from directory: {log_root_path}")
    print(f"[INFO] Exact adversarial experiment name requested from command line: {log_dir}")
    if cfg_adversarial["agent"]["experiment"]["experiment_name"]:
        log_dir_adversarial = log_dir + f"_{cfg_adversarial['agent']['experiment']['experiment_name']}"
    log_dir_adversarial = os.path.join(log_root_path, log_dir_adversarial)

    if cfg_adversarial["agent"]["experiment"]["write_interval"] == "auto":
        write_interval_adv = int(cfg_adversarial["train"]["timesteps"] / 100)
    if cfg_adversarial["agent"]["experiment"]["checkpoint_interval"] == "auto":
        checkpoint_interval_adv = int(cfg_adversarial["train"]["timesteps"] / 10)
    
    # ====================== Model Spawn  ==========================
    # Overwrite cfg by cli argument
    if model_adversarial is not None:
        cfg_adversarial["models"]["model_type"] = model_adversarial
    
    model_manager = ModelFactory(cfg=cfg_adversarial["models"], device=env.device)
    if model_manager.model_class == "mlp":
        adversarial_models = model_manager.generate_mlp_models(observation_size=obs_size,
                                                               state_size=state_size,
                                                               action_size=act_size,
                                                               possible_agents=adversarial_agents)
    elif model_manager.model_class == "gnn":
        node_cfg = None
        mapping_cfg = None
        if model_manager.model_type == "nervenet":
            node_cfg = {'node_info': env._unwrapped.cfg.node_info,
                        'num_nodes': env._unwrapped.cfg.num_nodes,
                        'num_actuated_nodes': env._unwrapped.cfg.num_actuated_nodes}
            
        elif model_manager.model_type == "bodytransformer":
            mapping_cfg = env._unwrapped.cfg.map_info

        else:
            raise RuntimeError("Not supported type")
        
        adversarial_models = model_manager.generate_gnn_models(observation_space=observation_space,
                                                               state_space=state_space,
                                                               action_space=action_space,
                                                               node_cfg=node_cfg,
                                                               mapping_cfg=mapping_cfg)
    else:
        raise RuntimeError("Not supported class")

    # ====================== Agent Spawn  ==========================
    # Scale Factor
    cfg_adversarial["agent"]["action_scale_factor"] = env._unwrapped.cfg.action_scale_factor
    if multi_agent:
        if model_manager.model_type == "mlp":
            from lib.agent.mappo import MAPPO
            adversarial_agent = MAPPO(observation_space=env.observation_space,
                                      state_space=env.state_space,
                                      action_space=env.action_space,
                                      possible_agents=adversarial_agents,
                                      model=adversarial_models,
                                      buffer=buffers,
                                      device=env.device,
                                      cfg=cfg_adversarial["agent"])
            
        elif model_manager.model_type == "communet":
            from lib.agent.communet_mappo import CommunetMAPPO
            adversarial_agent = CommunetMAPPO(observation_space=env.observation_space,
                                              state_space=env.state_space,
                                              action_space=env.action_space,
                                              possible_agents=adversarial_agents,
                                              model=adversarial_models,
                                              buffer=buffers,
                                              device=env.device,
                                              cfg=cfg_adversarial["agent"])
        
        elif model_manager.model_type == "shared" or model_manager.model_type == "superconnected":
            from lib.agent.cooperative_mappo import CooperativeMAPPO
            adversarial_agent = CooperativeMAPPO(observation_space=env.observation_space,
                                                 state_space=env.state_space,
                                                 action_space=env.action_space,
                                                 possible_agents=adversarial_agents,
                                                 model=adversarial_models,
                                                 buffer=buffers,
                                                 device=env.device,
                                                 cfg=cfg_adversarial["agent"])
        
        else:
            raise RuntimeError("Unvalid model type.")

    else:
        # from lib.agent.ppo_RARL import PPO_RARL
        adversarial_agent = MAPPO(observation_space=env.observation_space,
                                  state_space=env.state_space,
                                  action_space=env.action_space,
                                  possible_agents=adversarial_agents,
                                  model=adversarial_models,
                                  buffer=buffers,
                                  device=env.device,
                                  cfg=cfg_adversarial["agent"])
    
    # Checkpoint
    if args_cli.checkpoint_adversarial is not None:
        resume_path = os.path.abspath(args_cli.checkpoint_adversarial)
        adversarial_agent.load(resume_path)
        print(f"[INFO] Get checkpoint from {resume_path}")
    else:
        print("[INFO] Unfortunately a pre-trained checkpoint is not found for this task.")
        resume_path = None
    
    # Verify save logic
    verify_save_logic = True
    if verify_save_logic:
        test_adversarial_agent = copy.deepcopy(adversarial_agent)

    # ======================= Training ============================

    # Tensorboard Wrtier
    writer = SummaryWriter(log_dir=log_dir)
    nominal_cumulative_rewards = None
    adv_cumulative_rewards = None
    cumulative_timesteps = None
    tracking_data = collections.defaultdict(list)

    nominal_track_rewards = collections.deque(maxlen=env.num_envs)
    nominal_CLI_track_rewards = collections.deque(maxlen=env.num_envs)
    nominal_CLI_step_reward_means = collections.deque(maxlen=env.num_envs)

    adv_track_rewards = collections.deque(maxlen=env.num_envs)
    adv_CLI_track_rewards = collections.deque(maxlen=env.num_envs)
    adv_CLI_step_reward_means = collections.deque(maxlen=env.num_envs)

    track_timesteps = collections.deque(maxlen=env.num_envs)
    CLI_track_timesteps = collections.deque(maxlen=env.num_envs)
    
    t1_rollout = time.time()
    t2_rollout = 0
    t1_update = 0
    t2_update = 0

    # Reset environment
    obs, states, infos = env.reset()
    timestep = 0
    rollout = 0
    elapsed_time = 0
    
    nominal_obs = {k: v for k, v in obs.items() if k not in adversarial_agents}
    adv_obs = {k: v for k, v in obs.items() if k in adversarial_agents}

    nominal_states = {k: v for k, v in states.items() if k not in adversarial_agents}
    adv_states = {k: v for k, v in states.items() if k in adversarial_agents}

    nominal_update_turn = False                                                             # NOTE: for test

    # Simulate environment
    while simulation_app.is_running() and timestep <= cfg_nominal["train"]["timesteps"]:

        # ================== Interaction Phase =====================
        with torch.no_grad():
            # agent stepping0120012
            nominal_actions, nominal_nonscaled_actions, nominal_action_log_probs, _ = nominal_agent.act(nominal_obs, infos, timestep=timestep, deterministic=False)
            adv_actions, adv_nonscaled_actions, adv_action_log_probs, _ = adversarial_agent.act(adv_obs, infos, timestep=timestep, deterministic=False)
            
            # Action combination
            actions = {**nominal_actions, **adv_actions}
            # env stepping
            # NOTE: action을 dictionary로 묶어서 RA agent도 같이 dict로 env에 넘겨줘야한다
            next_obs, next_states, rewards, terminated, truncated, next_infos = env.step(actions)
            
            # Data slicing
            nominal_next_obs = {k: v for k, v in next_obs.items() if k not in adversarial_agents}
            adv_next_obs = {k: v for k, v in next_obs.items() if k in adversarial_agents}
            
            nominal_next_states = {k: v for k, v in next_states.items() if k not in adversarial_agents}
            adv_next_states = {k: v for k, v in next_states.items() if k in adversarial_agents}
            
            nominal_rewards = {k: v for k, v in rewards.items() if k not in adversarial_agents}
            adv_rewards = {k: v for k, v in rewards.items() if k in adversarial_agents}
            
            # update rollout number
            timestep += 1

            # NOTE: curriculum에 따라 두 agent를 번갈아 insert_data, update 진행
            # NOTE: observation, state를 agent별로 dict를 분해하여 넣어줘야한다.
            # NOTE: reward[:2]는 arm, leg reward[2:]는 adv
            
            # Insert data to the buffer
            if nominal_update_turn:
                nominal_agent.insert_data(observations=nominal_obs,
                                        states=nominal_states,
                                        actions=nominal_nonscaled_actions,
                                        action_log_probs=nominal_action_log_probs.reshape(-1, 1),
                                        rewards=nominal_rewards,
                                        next_observations=nominal_next_obs,
                                        next_states=nominal_next_states,
                                        truncated=truncated,
                                        terminated=terminated,
                                        infos=infos)
            else:
                adversarial_agent.insert_data(observations=adv_obs,
                                            states=adv_states,
                                            actions=adv_nonscaled_actions,
                                            action_log_probs=adv_action_log_probs.reshape(-1, 1),
                                            rewards=adv_rewards,
                                            next_observations=adv_next_obs,
                                            next_states=adv_next_states,
                                            truncated=truncated,
                                            terminated=terminated,
                                            infos=infos)
        
        # Parameter update
        if timestep % buffer.buffer_size == 0:
            if buffer.memory_index == 0:
                t2_rollout = time.time()

                t1_update = time.time()
                if nominal_update_turn: 
                    nominal_policy_loss, nominal_value_loss, nominal_entropy_loss, nominal_approx_kl = nominal_agent.update()
                else: 
                    adv_policy_loss, adv_value_loss, adv_entropy_loss, adv_approx_kl = adversarial_agent.update()
                t2_update = time.time()

                rollout += 1
            else:
                raise RuntimeError("Discrepency appears between Buffer Logic and Rollout Policy.")
            
        # =============== Logging Phase ================

        # Data setting for logging
        nominal_logged_reward = nominal_rewards.view(-1, num_agent)
        nominal_logged_reward = torch.mean(nominal_logged_reward, dim=-1).unsqueeze(-1) # Mean value of agent axis
        
        adv_logged_reward = adv_rewards.view(-1, num_agent)
        adv_logged_reward = torch.mean(adv_logged_reward, dim=-1).unsqueeze(-1) # Mean value of agent axis
        
        if nominal_cumulative_rewards is None:
            nominal_cumulative_rewards = torch.zeros_like(nominal_logged_reward, dtype=torch.float32)
            adv_cumulative_rewards = torch.zeros_like(adv_logged_reward, dtype=torch.float32)
            cumulative_timesteps = torch.zeros_like(nominal_logged_reward, dtype=torch.int32)
        
        # Accumulates per-step rewards
        nominal_cumulative_rewards.add_(nominal_logged_reward)
        adv_cumulative_rewards.add_(adv_logged_reward)
        cumulative_timesteps.add_(1)

        # CLI용 평균치 저장
        nominal_CLI_step_reward_means.append(torch.mean(nominal_logged_reward, dim=0).item())
        adv_CLI_step_reward_means.append(torch.mean(adv_logged_reward, dim=0).item())

        done = (terminated | truncated).squeeze(-1)
        finished_episodes = done.nonzero(as_tuple=False).squeeze(-1)

        if finished_episodes.numel():
            # Nominal 에피소드 보상 저장
            nominal_track_rewards.extend(nominal_cumulative_rewards[finished_episodes][:, 0].reshape(-1).tolist())
            nominal_CLI_track_rewards.extend(nominal_cumulative_rewards[finished_episodes][:, 0].detach().cpu().tolist())
            nominal_cumulative_rewards[finished_episodes] = 0

            # Adversarial 에피소드 보상 저장
            adv_track_rewards.extend(adv_cumulative_rewards[finished_episodes][:, 0].reshape(-1).tolist())
            adv_CLI_track_rewards.extend(adv_cumulative_rewards[finished_episodes][:, 0].detach().cpu().tolist())
            adv_cumulative_rewards[finished_episodes] = 0

            # 공통 에피소드 길이 저장
            track_timesteps.extend(cumulative_timesteps[finished_episodes][:, 0].reshape(-1).tolist())
            CLI_track_timesteps.extend(cumulative_timesteps[finished_episodes][:, 0].detach().cpu().tolist())
            cumulative_timesteps[finished_episodes] = 0

        # 기록을 위한 데이터 세팅 (접두사로 구분)
        tracking_data["Nominal/Instantaneous_reward_mean"].append(torch.mean(nominal_logged_reward).item())
        tracking_data["Adversarial/Instantaneous_reward_mean"].append(torch.mean(adv_logged_reward).item())

        if len(nominal_track_rewards):
            tracking_data["Nominal/Episode_reward_mean"].append(np.mean(nominal_track_rewards))
            tracking_data["Adversarial/Episode_reward_mean"].append(np.mean(adv_track_rewards))
            tracking_data["Episode/Total_timesteps_mean"].append(np.mean(track_timesteps))

            nominal_track_rewards.clear()
            adv_track_rewards.clear()
            track_timesteps.clear()
        
        # Tensorboard Logging
        if timestep % write_interval_nominal == 0: 
            for k, v in tracking_data.items():
                writer.add_scalar(k, np.mean(v), timestep)
            tracking_data.clear()

        # # Accumulates per-step rewards
        # cumulative_rewards.add_(logged_reward)
        # cumulative_timesteps.add_(1)
        # # Mean of per-step rewards (Mean value of env axis)
        # CLI_step_reward_means.append(torch.mean(logged_reward, dim=0).item())

        # done = (terminated | truncated).squeeze(-1)
        # finished_episodes = done.nonzero(as_tuple=False).squeeze(-1)
        # if finished_episodes.numel():
        #     # Storage cumulative rewards and timesteps
        #     track_rewards.extend(cumulative_rewards[finished_episodes][:, 0].reshape(-1).tolist())
        #     track_timesteps.extend(cumulative_timesteps[finished_episodes][:, 0].reshape(-1).tolist())
        #     CLI_track_rewards.extend(cumulative_rewards[finished_episodes][:, 0].detach().cpu().tolist())
        #     CLI_track_timesteps.extend(cumulative_timesteps[finished_episodes][:, 0].detach().cpu().tolist())
        #     # reset the cumulative rewards and timesteps
        #     cumulative_rewards[finished_episodes] = 0
        #     cumulative_timesteps[finished_episodes] = 0

        # # record data
        # tracking_data["Reward / Instantaneous reward (max)"].append(torch.max(logged_reward).item())
        # tracking_data["Reward / Instantaneous reward (min)"].append(torch.min(logged_reward).item())
        # tracking_data["Reward / Instantaneous reward (mean)"].append(torch.mean(logged_reward).item())

        # task_reward = next_infos.get("reward", None)
        # if task_reward is not None:
        #     for k, v in task_reward.items():
        #         # Mean value of env axis
        #         tracking_data[k].append(torch.mean(v).item())

        # if len(track_rewards):
        #     track_reward_np = np.array(track_rewards)
        #     track_timestep_np = np.array(track_timesteps)

        #     tracking_data["Reward / Total reward (max)"].append(np.max(track_reward_np))
        #     tracking_data["Reward / Total reward (min)"].append(np.min(track_reward_np))
        #     tracking_data["Reward / Total reward (mean)"].append(np.mean(track_reward_np))

        #     tracking_data["Episode / Total timesteps (max)"].append(np.max(track_timestep_np))
        #     tracking_data["Episode / Total timesteps (min)"].append(np.min(track_timestep_np))
        #     tracking_data["Episode / Total timesteps (mean)"].append(np.mean(track_timestep_np))

        #     # reset data containers for next iteration
        #     track_rewards.clear()
        #     track_timesteps.clear()
        
        # # Tensorboard logging
        # if timestep % write_interval == 0: 
        #     for k, v in tracking_data.items():
        #         if k.endswith("(min)"):
        #             writer.add_scalar(k, np.min(v), timestep)
        #         elif k.endswith("(max)"):
        #             writer.add_scalar(k, np.max(v), timestep)
        #         else:
        #             writer.add_scalar(k, np.mean(v), timestep)
        #     # reset data containers for next iteration
        #     tracking_data.clear()

        # CLI Logging about the training process at each parameter update
        if timestep % buffer.buffer_size == 0 and buffer.memory_index == 0:
            # per_step_reward = float(np.mean(CLI_step_reward_means)) if len(CLI_step_reward_means) else float("nan")
            # avg_ep_reward = float(np.mean(CLI_track_rewards)) if len(CLI_track_rewards) else float("nan")


            avg_ep_step = float(np.mean(CLI_track_timesteps)) if len(CLI_track_timesteps) else float("nan")
            
            nominal_per_reward = float(np.mean(nominal_CLI_step_reward_means)) if len(nominal_CLI_step_reward_means) else float("nan")
            nominal_ep_reward = float(np.mean(nominal_CLI_track_rewards)) if len(nominal_CLI_track_rewards) else float("nan")
            
            adv_per_reward = float(np.mean(adv_CLI_step_reward_means)) if len(adv_CLI_step_reward_means) else float("nan")
            adv_ep_reward = float(np.mean(adv_CLI_track_rewards)) if len(adv_CLI_track_rewards) else float("nan")

            # ep_step = "-" if np.isnan(avg_ep_step) else f"{avg_ep_step:6.3f} steps"
            # per_r = "-" if np.isnan(per_step_reward) else f"{per_step_reward:6.3f}"
            # ep_r = "-" if np.isnan(avg_ep_reward) else f"{avg_ep_reward:6.3f}"

            elapsed_time += (t2_rollout + t2_update - t1_rollout - t1_update)


            print(f"| Step Progress {timestep} / {cfg_nominal['train']['timesteps']}")
            print(f"| Time Progress {e_h:02d}:{e_m:02d}:{e_s:02d} / {c_h:02d}:{c_m:02d}:{c_s:02d}")
            print(f"| Avg Episode Step: {avg_ep_step:.1f}")
            print(f" -" * 40)
            print(f"| [Nominal Agent (Robot)]")
            print(f"| Value Loss : {nominal_value_loss:6.3f} | Policy Loss: {nominal_policy_loss:6.3f}")
            print(f"| Per-Step R : {nominal_per_reward:6.3f} | Episode R  : {nominal_ep_reward:6.3f}")
            print(f" -" * 40)
            print(f"| [Adversarial Agent (Pusher)]")
            print(f"| Value Loss : {adv_value_loss:6.3f} | Policy Loss: {adv_policy_loss:6.3f}")
            print(f"| Per-Step R : {adv_per_reward:6.3f} | Episode R  : {adv_ep_reward:6.3f}")



            e_h = int(elapsed_time // 3600)
            e_m = int((elapsed_time % 3600) // 60)
            e_s = int(elapsed_time % 60)
            total_rollout = int(cfg_nominal["train"]["timesteps"] // buffer.buffer_size)
            complete_time = (t2_rollout + t2_update - t1_rollout - t1_update) * (total_rollout - rollout)
            c_h = int(complete_time // 3600)
            c_m = int((complete_time % 3600) // 60)
            c_s = int(complete_time % 60)

            content_width = 64
            line_header = f"Step Progress {timestep} / {cfg_nominal['train']['timesteps']}"
            line_time_header = f"Time Progress  {e_h:02d}:{e_m:02d}:{e_s:02d}/{c_h:02d}:{c_m:02d}:{c_s:02d}"
            line_rollout_time = f"Rollout Time      : {t2_rollout - t1_rollout:6.3f} sec"
            line_train_time = f"Training Time     : {t2_update - t1_update:6.3f} sec"
            nominal_loss = f"| Value Loss : {nominal_value_loss:6.3f} | Policy Loss: {nominal_policy_loss:6.3f}"
            nominal_reward = f"| Per-Step R : {nominal_per_reward:6.3f} | Episode R  : {nominal_ep_reward:6.3f}"
            adv_loss = f"| Value Loss : {adv_value_loss:6.3f} | Policy Loss: {adv_policy_loss:6.3f}"
            adv_reward = f"| Per-Step R : {adv_per_reward:6.3f} | Episode R  : {adv_ep_reward:6.3f}"

            # line_episode_step = f"Avg Episode Step  : {ep_step}"
            # line_per_step_reward = f"Per-Step Rewards  : {per_r}"
            # line_episode_reward = f"Epiode Rewards    : {ep_r}"

            if nominal_update_turn:
                print(f"===================== Nominal Agent Update =====================")
            else:
                print(f"=================== Adversarial Agent Update ===================")
            print(f" ________________________________________________________________")
            print(f"|                                                                |")
            print(f"|{line_header.center(content_width)}|")
            print(f"|{line_time_header.center(content_width)}|")
            print(f"|________________________________________________________________|")
            print(f"|                                                                |")
            print(f"| {line_rollout_time:<{content_width-1}}|")
            print(f"| {line_train_time:<{content_width-1}}|")
            print(f"------------------------------------------------------------------")
            print(f"| [Nominal Agent (Robot)]")
            print(f"| {nominal_loss:<{content_width-1}}|")
            print(f"| {nominal_reward:<{content_width-1}}|")
            print(f"------------------------------------------------------------------")
            print(f"| [Adversarial Agent (Pusher)]")
            print(f"| {adv_loss:<{content_width-1}}|")
            print(f"| {adv_reward:<{content_width-1}}|")
            # print(f"------------------------------------------------------------------")
            # print(f"| {line_episode_step:<{content_width-1}}|")
            # print(f"| {line_per_step_reward:<{content_width-1}}|")
            # print(f"| {line_episode_reward:<{content_width-1}}|")
            print(f"|________________________________________________________________|")

            # update rollout time
            t1_rollout = time.time()

        # Checkpoint save
        if timestep % checkpoint_interval_nominal == 0:
            nominal_checkpoint_path = os.path.join(log_dir, f"agent_{timestep}.pt")
            nominal_checkpoint_path_jit = os.path.join(log_dir, f"agent_jit_{timestep}.pt") if not multi_agent else None
            nominal_agent.save(nominal_checkpoint_path, nominal_checkpoint_path_jit)
        
        if timestep % checkpoint_interval_adv == 0:
            adv_checkpoint_path = os.path.join(log_dir, f"agent_{timestep}.pt")
            adv_checkpoint_path_jit = os.path.join(log_dir, f"agent_jit_{timestep}.pt") if not multi_agent else None
            adversarial_agent.save(adv_checkpoint_path, adv_checkpoint_path_jit)

            # if verify_save_logic:
            #     test_agent.load(checkpoint_path)
            #     test_agent.set_running_mode("eval")
            #     with torch.no_grad():
            #         agent.set_running_mode("eval")
            #         actions, nonscaled_actions, action_log_probs, _ = agent.act(obs, infos, timestep=timestep, deterministic=True)
            #         test_actions, test_nonscaled_actions, test_action_log_probs, _ = test_agent.act(obs, infos, timestep=timestep, deterministic=True)
            #         agent.set_running_mode("train")
                
            #     if not torch.allclose(actions, test_actions):
            #         max_err = (actions - test_actions).abs().max().item()
            #         raise RuntimeError(f"Model mismatch. Please check the save logic. [Max Error : {max_err}]")
                
            #     if not torch.allclose(nonscaled_actions, test_nonscaled_actions):
            #         max_err = (nonscaled_actions - test_nonscaled_actions).abs().max().item()
            #         raise RuntimeError(f"Model mistmatch. Please check the save logic. [Max Error : {max_err}]")


        # update
        nominal_obs = nominal_next_obs
        adv_obs = adv_next_obs
        nominal_states = nominal_next_states
        adv_states = adv_next_states
        infos = next_infos

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()