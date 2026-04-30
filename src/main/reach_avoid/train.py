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
parser.add_argument("--num_envs", type=int, default=5, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="G1-fall", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--ra_checkpoint", type=str, default=None, help="Path to Reach-Avoid model checkpoint.")

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

from torch.utils.tensorboard import SummaryWriter


from datetime import datetime

import lib

from wrapper.isaaclab_wrapper import IsaacLabWrapper
from wrapper.record_wrapper import RecordVideo
from lib.utils.parse_utils import parse_env_cfg, load_cfg_from_registry
from lib.buffer.rolloutbuffer import RolloutBuffer
from lib.model.model_factory import ModelFactory

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
    if args_cli.checkpoint is not None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(args_cli.checkpoint)), "Reach_Avoid")
    else:
        log_dir = os.getcwd()

    # ============================ Env & Wrapper Spawn ================================

    # Create isaac environment
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        cfg["agent"]["seed"] = args_cli.seed
        ra_cfg["agent"]["seed"] = args_cli.seed
    else:
        env_cfg.seed = cfg.get("seed", None)
        cfg["agent"]["seed"] = cfg.get("seed", 42) # 42 is a default seed (equal to env)
        ra_cfg["agent"]["seed"] = cfg.get("seed", 42) # 42 is a default seed (equal to env)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        args_cli.video_interval = int(cfg["train"]["timesteps"] / 5)
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
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

    if cfg["agent"]["experiment"]["write_interval"] == "auto":
        write_interval = int(cfg["train"]["timesteps"] / 100)
    if cfg["agent"]["experiment"]["checkpoint_interval"] == "auto":
        checkpoint_interval = int(cfg["train"]["timesteps"] / 10)

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
        

    # ============= RA Buffer and Model Spawn ===============
    from lib.buffer.replaybuffer import HindSightReplayBuffer
    from lib.model.MLP import RA_Critic
    if not hasattr(env._unwrapped.cfg, "ra_state_space"):
        raise RuntimeError("Explicit state space is not defined.")
    ra_buffer =  HindSightReplayBuffer(ra_cfg["buffer"]["buffer_size"], env.num_envs, device=env.device)
    ra_buffer.init_buffer(env._unwrapped.cfg.ra_state_space)
    ra_model = {"critic": RA_Critic(env._unwrapped.cfg.ra_state_space, env.device)}

    # ==================== RA Agent Spawn ===================
    from lib.agent.reach_avoid import ReachAvoid
    ra_agent = ReachAvoid(ra_model, ra_buffer, device=env.device, cfg=ra_cfg["agent"])
    
    # Checkpoint (Policy)
    if args_cli.checkpoint is not None:
        resume_path = os.path.abspath(args_cli.checkpoint)
        agent.load(resume_path)
        print(f"[INFO] Get checkpoint of policy from {resume_path}")
    else:
        resume_path = None
        print("[INFO] Unfortunately a pre-trained policy is not found for this task.")

    # Checkpoint (RA value)
    if args_cli.ra_checkpoint is not None:
        resume_path_ra = os.path.abspath(args_cli.ra_checkpoint)
        ra_agent.load(resume_path_ra)
        print(f"[INFO] Get checkpoint RA Value from {resume_path_ra}")
    else:
        resume_path_ra = None
        print("[INFO] Unfortunately a pre-trained RA Value is not found for this task.")

    # ======================= Training ============================

    # Tensorboard Wrtier
    writer = SummaryWriter(log_dir=log_dir)
    tracking_data = collections.defaultdict(list)
    CLI_step_reward_means = collections.deque(maxlen=env.num_envs)
    CLI_value_loss = collections.deque(maxlen=env.num_envs)

    # Reset environment
    obs, states, infos = env.reset()
    timestep = 0
    elapsed_time = 0
    
    # Simulate environment
    while simulation_app.is_running() and timestep <= ra_cfg["train"]["timesteps"]:

        # ================== Interaction Phase =====================
        t1_loop = time.time()
        with torch.no_grad():
            # agent stepping
            actions, nonscaled_actions, action_log_probs, _ = agent.act(obs, infos, timestep=timestep, deterministic=False)
            # env stepping
            next_obs, next_states, rewards, terminated, truncated, next_infos = env.step(actions)
            # update rollout number
            timestep += 1
        
            # Insert data to the buffer
            ra_agent.insert_data(states=infos["ra_states"],
                                 next_states=next_infos["ra_states"],
                                 l_values=infos["l_values"],
                                 g_values=infos["g_values"],
                                 truncated=truncated,
                                 terminated=terminated)
        
        # Parameter update
        if timestep >= ra_cfg["agent"]["learning_starts"]:
            value_loss = ra_agent.update()
            CLI_value_loss.append(value_loss)
            tracking_data["Loss / RA Value Loss"].append(value_loss)
        
        t2_loop = time.time()

        # =============== Logging Phase ================
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
            tracking_data.clear()

        # CLI Logging about the training process at each parameter update
        if timestep % ra_buffer.buffer_size == 0:
            per_update_value_loss = float(np.mean(CLI_value_loss)) if len(CLI_value_loss) else float("nan")
            per_value_loss =  "-" if np.isnan(per_update_value_loss) else f"{per_update_value_loss:6.3f}"

            elapsed_time += (t2_loop - t1_loop)
            e_h = int(elapsed_time // 3600)
            e_m = int((elapsed_time % 3600) // 60)
            e_s = int(elapsed_time % 60)
            total_timesteps = int(ra_cfg["train"]["timesteps"])
            complete_time = (t2_loop - t1_loop) * (total_timesteps - timestep)
            c_h = int(complete_time // 3600)
            c_m = int((complete_time % 3600) // 60)
            c_s = int(complete_time % 60)

            content_width = 64
            line_header = f"Step Progress {timestep} / {ra_cfg['train']['timesteps']}"
            line_time_header = f"Time Progress  {e_h:02d}:{e_m:02d}:{e_s:02d}/{c_h:02d}:{c_m:02d}:{c_s:02d}"
            line_value_loss = f"Value Loss        : {per_value_loss}"

            print(f" ________________________________________________________________")
            print(f"|                                                                |")
            print(f"|{line_header.center(content_width)}|")
            print(f"|{line_time_header.center(content_width)}|")
            print(f"|________________________________________________________________|")
            print(f"|                                                                |")
            print(f"| {line_value_loss:<{content_width-1}}|")
            print(f"|________________________________________________________________|")

        # Checkpoint save
        if timestep % checkpoint_interval == 0:
            checkpoint_ra_path = os.path.join(log_dir, f"ra_agent_{timestep}.pt")
            ra_agent.save(checkpoint_ra_path)

        # update
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