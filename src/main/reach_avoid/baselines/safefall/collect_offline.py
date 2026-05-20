"""
Stage 1 of the SafeFall baseline pipeline.

Roll out a nominal locomotion policy and dump per-episode `safe_fall_obs`
sequences to .pt shards under ``Safe_Fall/dataset_<timestamp>/``. Labels are
not persisted; ``train_offline.py`` regenerates them from `fall_lead_seconds`
at load time using the 3-segment rule shared with online collection.
"""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="SafeFall offline data collection.")
parser.add_argument("--seed", type=int, default=None, help="Override seed from cfg.")
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel envs.")
parser.add_argument("--task", type=str, default="G1-fall", help="Task name.")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to the nominal locomotion policy checkpoint (.pt).")
parser.add_argument("--algorithm", type=str, default="MAPPO",
                    choices=["PPO", "SAC", "TD3", "MAPPO"],
                    help="Algorithm used to train the nominal policy.")
parser.add_argument("--model", type=str, default="Shared",
                    choices=["MLP", "Shared", "Communet"],
                    help="Model class of the nominal policy.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import json
import math
import os
from datetime import datetime

import gymnasium as gym
import torch

import lib  # noqa: F401  (registers tasks)

from wrapper.isaaclab_wrapper import IsaacLabWrapper
from lib.utils.parse_utils import parse_env_cfg, load_cfg_from_registry
from lib.buffer.rolloutbuffer import RolloutBuffer
from lib.buffer.baselines.recurrent_replay import RecurrentReplayBuffer
from lib.model.model_factory import ModelFactory


algorithm = args_cli.algorithm.lower()
model_cli = args_cli.model.lower() if args_cli.model is not None else None


def _build_policy_agent(env, cfg, multi_agent, possible_agents):
    """Reproduce train.py's policy agent construction so the checkpoint loads cleanly."""
    if multi_agent:
        observation_space = {uid: env.observation_space[uid] for uid in possible_agents}
        action_space = {uid: env.action_space[uid] for uid in possible_agents}
        state_space = env.state_space if env.state_space else None
        buffers = {
            uid: RolloutBuffer(cfg["buffer"]["buffer_size"], env.num_envs, device=env.device)
            for uid in possible_agents
        }
        for uid in possible_agents:
            sspace = env.state_space[uid] if env.state_space else None
            buffers[uid].init_buffer(observation_space[uid], sspace, action_space[uid])

        obs_size = {uid: buffers[uid].tensors["observations"].shape[-1] for uid in possible_agents}
        state_size = {
            uid: (buffers[uid].tensors["states"].shape[-1] if env.state_space else obs_size[uid])
            for uid in possible_agents
        }
        act_size = {uid: buffers[uid].tensors["actions"].shape[-1] for uid in possible_agents}
    else:
        observation_space = env.observation_space
        action_space = env.action_space
        state_space = env.state_space if env.state_space else None
        buffer = RolloutBuffer(cfg["buffer"]["buffer_size"], env.num_envs, device=env.device)
        buffer.init_buffer(observation_space, state_space, action_space)
        obs_size = buffer.tensors["observations"].shape[-1]
        state_size = buffer.tensors["states"].shape[-1] if env.state_space else obs_size
        act_size = buffer.tensors["actions"].shape[-1]

    if model_cli is not None:
        cfg["models"]["model_type"] = model_cli
    cfg["models"]["multi_agent"] = multi_agent
    model_manager = ModelFactory(cfg=cfg["models"], device=env.device)
    if model_manager.model_class != "mlp":
        raise RuntimeError("Only MLP model class is supported for nominal policy.")
    models = model_manager.generate_mlp_models(
        observation_size=obs_size,
        state_size=state_size,
        action_size=act_size,
        possible_agents=possible_agents,
    )

    cfg["agent"]["action_scale_factor"] = env._unwrapped.cfg.action_scale_factor
    if multi_agent:
        if model_manager.model_type == "mlp":
            from lib.agent.mappo import MAPPO
            agent = MAPPO(observation_space=env.observation_space,
                          state_space=env.state_space,
                          action_space=env.action_space,
                          possible_agents=possible_agents,
                          model=models, buffer=buffers,
                          device=env.device, cfg=cfg["agent"])
        elif model_manager.model_type == "communet":
            from lib.agent.communet_mappo import CommunetMAPPO
            agent = CommunetMAPPO(observation_space=env.observation_space,
                                  state_space=env.state_space,
                                  action_space=env.action_space,
                                  possible_agents=possible_agents,
                                  model=models, buffer=buffers,
                                  device=env.device, cfg=cfg["agent"])
        elif model_manager.model_type == "shared":
            from lib.agent.cooperative_mappo import CooperativeMAPPO
            agent = CooperativeMAPPO(observation_space=env.observation_space,
                                     state_space=env.state_space,
                                     action_space=env.action_space,
                                     possible_agents=possible_agents,
                                     model=models, buffer=buffers,
                                     device=env.device, cfg=cfg["agent"])
        else:
            raise RuntimeError(f"Unsupported model type: {model_manager.model_type}")
    else:
        from lib.agent.ppo import PPO
        agent = PPO(model=models, buffer=buffer,
                    device=env.device, cfg=cfg["agent"])
    return agent


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    try:
        cfg = load_cfg_from_registry(args_cli.task, f"rl_{algorithm}_cfg_entry_point")
        ra_cfg = load_cfg_from_registry(args_cli.task, "ra_cfg_entry_point")
    except ValueError as e:
        print(e)
        return

    sf_cfg = ra_cfg["safe_fall"]
    collection_cfg = sf_cfg["collection"]

    # specify directory for logging experiments (load checkpoint)
    if args_cli.checkpoint is not None:
        base_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
        log_dir = os.path.join(base_dir, "Safe_Fall")
    else:
        raise ValueError("Checkpoint path must be assigned for policy-conditioned RA value function.")

    seed = args_cli.seed if args_cli.seed is not None else cfg.get("seed", 42)
    env_cfg.seed = seed
    cfg["agent"]["seed"] = seed

    env_cfg.total_timesteps = cfg["train"]["timesteps"]

    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        step_dt = env.step_dt
    except AttributeError:
        step_dt = env.unwrapped.step_dt

    env = IsaacLabWrapper(env)

    multi_agent = algorithm == "mappo"
    cfg["models"]["multi_agent"] = multi_agent
    if cfg["buffer"]["buffer_size"] == -1:
        cfg["buffer"]["buffer_size"] = cfg["agent"]["rollouts"]

    possible_agents = env._unwrapped.cfg.possible_agents if multi_agent else None

    # Build & load the nominal policy.
    agent = _build_policy_agent(env, cfg, multi_agent, possible_agents)
    resume_path = os.path.abspath(args_cli.checkpoint)
    agent.load(resume_path)
    print(f"[INFO] Loaded nominal policy checkpoint from {resume_path}")

    # SafeFall observation buffer (online collection mode).
    obs_dim = env._unwrapped.cfg.safe_fall_obs_dim
    max_ep_steps = int(env._unwrapped.max_episode_length)

    # Sync seq_len with the env's actual max_episode_length so that downstream
    # full-episode training (train_offline.py) uses the right window. The yaml
    # value is treated as an upper-bound hint and overridden here.
    cfg_seq_len = sf_cfg["buffer"].get("seq_len")
    if cfg_seq_len != max_ep_steps:
        print(f"[INFO] Overriding safe_fall.buffer.seq_len ({cfg_seq_len}) "
              f"with env max_episode_length ({max_ep_steps}).")
        sf_cfg["buffer"]["seq_len"] = max_ep_steps

    target = int(collection_cfg["target_trajectories"])
    train_val_split = tuple(collection_cfg["train_val_split"])
    shard_size = int(collection_cfg["shard_size"])

    # buffer_size is the number of circular-buffer rows (timesteps); each row
    # already stores num_envs samples in parallel. Size it so that the worst
    # case (all envs survive to max_ep_steps) closes `target` trajectories
    # without wrap-around overwriting still-referenced closed_episodes rows.
    episodes_per_env = math.ceil(target / env.num_envs) + 1
    buffer_rows = max_ep_steps * episodes_per_env

    buffer = RecurrentReplayBuffer(
        buffer_size=buffer_rows,
        num_envs=env.num_envs,
        device=env.device,
        seq_len=sf_cfg["buffer"]["seq_len"],
        fall_lead_seconds=sf_cfg["buffer"]["fall_lead_seconds"],
        step_dt=step_dt,
        max_episode_steps=max_ep_steps,
        max_closed_episodes=target + max_ep_steps,
    )
    buffer.init_buffer(obs_dim=obs_dim)

    if sum(train_val_split) != target:
        raise ValueError(
            f"train_val_split {train_val_split} sum ({sum(train_val_split)}) "
            f"does not match target_trajectories ({target})."
        )

    print(f"[INFO] Collecting {target} trajectories "
          f"(train {train_val_split[0]} / val {train_val_split[1]}) "
          f"from {env.num_envs} envs, max_ep_steps={max_ep_steps}.")

    timestep = 0
    obs, states, infos = env.reset()
    while simulation_app.is_running() and len(buffer.closed_episodes) < target:
        with torch.no_grad():
            actions, _, _, _ = agent.act(obs, infos, timestep=timestep, deterministic=True)
            next_obs, next_states, rewards, terminated, truncated, next_infos = env.step(actions)
            timestep += 1

            buffer.add_samples(
                obs=infos["safe_fall_obs"],
                terminated=terminated,
                truncated=truncated,
            )

        if timestep % 100 == 0:
            print(f"[step {timestep:7d}] closed_episodes = {len(buffer.closed_episodes):6d} / {target}")

        obs = next_obs
        states = next_states
        infos = next_infos

    # Trim to exact target count for clean train/val partition.
    if len(buffer.closed_episodes) > target:
        buffer.closed_episodes = buffer.closed_episodes[:target]

    # Create output directory and dump shards.
    timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(log_dir, f"dataset_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Dumping shards to {output_dir} ...")
    meta = buffer.flush_to_disk(
        output_dir=output_dir,
        shard_size=shard_size,
        train_val_split=train_val_split,
    )
    meta.update({
        "task":              args_cli.task,
        "algorithm":         args_cli.algorithm,
        "checkpoint": resume_path,
        "seed":              seed,
        "num_envs":          env.num_envs,
        "step_dt":           step_dt,
        "obs_dim":           obs_dim,
        "max_episode_steps": max_ep_steps,
        "fall_lead_seconds": sf_cfg["buffer"]["fall_lead_seconds"],
        # Snapshot of cfg so train_offline.py can run without booting Isaac Lab.
        "safe_fall_cfg":     sf_cfg,
    })
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("=" * 72)
    print(f"  num_trajectories  : {meta['num_trajectories']}")
    print(f"  terminated_ratio  : {meta['terminated_ratio']:.4f}")
    print(f"  mean_length       : {meta['mean_length']:.2f}")
    print(f"  shards            : {len(meta['shard_files'])}")
    print(f"  output_dir        : {output_dir}")
    print("=" * 72)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
