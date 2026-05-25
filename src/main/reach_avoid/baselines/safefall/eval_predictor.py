"""
Script to evaluate a fall predictor (Reach-Avoid / SafeFall) and report
False Alarm Rate (FAR) and Lead Time (LT).

Run once per predictor with the same seed; results are appended to
``metrics_summary.csv`` so the two predictors can be compared side by side.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a fall predictor (FAR / Lead Time).")
parser.add_argument("--seed", type=int, default=None, help="Seed of RL environment")
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="G1-fall-play", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to nominal policy checkpoint.")
parser.add_argument("--predictor_checkpoint", type=str, default=None, help="Path to fall predictor checkpoint.")

parser.add_argument("--predictor",
                    type=str,
                    default="ra",
                    choices=["ra", "safefall"],
                    help="Fall predictor type to evaluate.")

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

# evaluation control
parser.add_argument("--num_eval_falls", type=int, default=300, help="Target number of fall episodes.")
parser.add_argument("--num_eval_safe", type=int, default=300, help="Target number of safe episodes.")
parser.add_argument("--sustain_k", type=int, default=1, help="Minimum consecutive steps for a valid alarm.")
parser.add_argument("--gap_tol", type=int, default=5, help="Max gap (steps) merged into one alarm run.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
args_cli.headless = True                    # Headless mode
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import os
import time

import torch
import gymnasium as gym

import lib

from wrapper.isaaclab_wrapper import IsaacLabWrapper
from lib.utils.parse_utils import parse_env_cfg, load_cfg_from_registry
from lib.buffer.rolloutbuffer import RolloutBuffer
from lib.buffer.baselines.fall_eval import FallPredictorEvaluator
from lib.model.model_factory import ModelFactory

# config shortcuts
algorithm = args_cli.algorithm.lower()
model = args_cli.model.lower() if args_cli.model is not None else None

# predictor-specific fixed decision threshold (training-time reference value)
ALARM_THRESHOLD = {"ra": 0.0, "safefall": 0.5}


def main():
    """Main evaluation routine."""

    # ============================= Config Parsing ===============================
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    try:
        cfg = load_cfg_from_registry(args_cli.task, f"rl_{algorithm}_cfg_entry_point")
        ra_cfg = load_cfg_from_registry(args_cli.task, "ra_cfg_entry_point")
    except ValueError as e:
        print(e)
        return

    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint (nominal policy) must be provided.")
    if args_cli.predictor_checkpoint is None:
        raise ValueError("--predictor_checkpoint must be provided.")

    base_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    if args_cli.predictor == "ra":
        log_dir = os.path.join(base_dir, "Reach_Avoid")
    else:
        log_dir = os.path.join(base_dir, "Safe_Fall")
    os.makedirs(log_dir, exist_ok=True)

    # ============================ Env & Wrapper Spawn ================================
    predictor = args_cli.predictor
    pred_cfg = ra_cfg[predictor if predictor == "ra" else "safe_fall"]

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        cfg["agent"]["seed"] = args_cli.seed
        pred_cfg["agent"]["seed"] = args_cli.seed
    else:
        env_cfg.seed = cfg.get("seed", None)
        cfg["agent"]["seed"] = cfg.get("seed", 42)
        pred_cfg["agent"]["seed"] = cfg.get("seed", 42)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # environment step dt
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    env = IsaacLabWrapper(env)

    # ======================= Buffer (nominal policy) =========================
    multi_agent = algorithm == "mappo"
    cfg["models"]["multi_agent"] = multi_agent
    if cfg["buffer"]["buffer_size"] == -1:
        cfg["buffer"]["buffer_size"] = cfg["agent"]["rollouts"]
    else:
        raise RuntimeError("Replaybuffer for Off-policy algorithm is not implemented yet.")

    possible_agents = None
    if multi_agent:
        obs_size, state_size, act_size, buffers = {}, {}, {}, {}
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

    # ====================== Agent Spawn (nominal policy) ==========================
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

    # ======================= Checkpoint Load ========================
    resume_path = os.path.abspath(args_cli.checkpoint)
    agent.load(resume_path)
    print(f"[INFO] Loaded nominal policy from {resume_path}")

    resume_path_pred = os.path.abspath(args_cli.predictor_checkpoint)
    pred_agent.load(resume_path_pred)
    print(f"[INFO] Loaded predictor ({predictor}) from {resume_path_pred}")

    agent.set_running_mode("eval")
    pred_agent.set_running_mode("eval")

    # ======================= Evaluator ============================
    evaluator = FallPredictorEvaluator(
        predictor_type=predictor,
        num_envs=env.num_envs,
        dt=dt,
        device=env.device,
        sustain_k=args_cli.sustain_k,
        gap_tol=args_cli.gap_tol,
        fall_lead_seconds=pred_cfg["buffer"].get("fall_lead_seconds", 0.1),
        max_fall=args_cli.num_eval_falls,
        max_safe=args_cli.num_eval_safe,
    )

    # SafeFall online inference state — per-env GRU hidden state, reset on done.
    if predictor == "safefall":
        H = int(pred_cfg["model"]["hidden_dim"])
        num_layers = int(pred_cfg["model"].get("num_layers", 1))
        h_pred = torch.zeros(num_layers, env.num_envs, H, device=env.device)
    else:
        h_pred = None

    threshold = ALARM_THRESHOLD[predictor]

    # ======================= Evaluation Loop ============================
    obs, states, infos = env.reset()
    timestep = 0
    start_time = time.time()

    while simulation_app.is_running() and (
        evaluator.count_fall < args_cli.num_eval_falls
        or evaluator.count_safe < args_cli.num_eval_safe
    ):
        with torch.no_grad():
            # nominal policy stepping (deterministic; predictor is passive)
            actions, _, _, _ = agent.act(obs, infos, timestep=timestep, deterministic=True)

            # predictor step data — per-env scalar score (RA value or P(falling))
            if predictor == "ra":
                score, _, _ = pred_agent.critic(infos["ra_states"])         # (N, 1)
                step_data = score.reshape(-1)
            else:  # safefall — online stateful GRU
                sf_obs = infos["safe_fall_obs"].unsqueeze(1)                # (N, 1, obs_dim)
                logits, h_pred, _ = pred_agent.critic(sf_obs, h_pred)       # (N, 1, 2)
                step_data = torch.softmax(logits[:, -1, :], dim=-1)[:, 1]   # (N,)

            # env stepping
            next_obs, next_states, _, terminated, truncated, next_infos = env.step(actions)
            timestep += 1

        evaluator.add_step(step_data, terminated, truncated)

        # Reset per-env GRU hidden state for envs whose episode just ended;
        # next iteration's obs for those envs is already from the new episode.
        if predictor == "safefall":
            done = torch.logical_or(terminated.bool(), truncated.bool()).reshape(-1)
            if done.any():
                h_pred[:, done, :] = 0.0

        if timestep % 200 == 0:
            print(f"[INFO] step {timestep} | falls {evaluator.count_fall}/{args_cli.num_eval_falls}"
                  f" | safe {evaluator.count_safe}/{args_cli.num_eval_safe}")

        obs = next_obs
        states = next_states
        infos = next_infos

    env.close()

    # ======================= Metrics ============================
    metrics = evaluator.compute_metrics(threshold)
    elapsed = time.time() - start_time

    print("\n" + "=" * 72)
    print(f" Fall Predictor Evaluation : {predictor}  (threshold tau = {threshold})")
    print("=" * 72)
    print(f"  Episodes        : fall {metrics['n_fall']} | safe {metrics['n_safe']}")
    print(f"  FAR (episode)   : {metrics['FAR_episode']:.4f}")
    print(f"  FAR (step)      : {metrics['FAR_step']:.4f}")
    print(f"  Detection Rate  : {metrics['detection_rate']:.4f}")
    print(f"  Lead Time (s)   : mean {metrics['LT_mean']:.3f} | "
          f"max {metrics['LT_max']:.3f} | min {metrics['LT_min']:.3f}")
    print(f"  LT first-alarm  : mean {metrics['LT_first_mean']:.3f}  (optimistic bound)")
    print(f"  Elapsed         : {elapsed:.1f} s ({timestep} steps)")
    print("=" * 72 + "\n")

    # raw records for offline re-analysis
    raw_path = os.path.join(log_dir, f"raw_{predictor}.pt")
    evaluator.save_raw(raw_path, threshold)
    print(f"[INFO] Raw records saved to {raw_path}")

    # append one row to the shared comparison CSV
    csv_path = os.path.join(log_dir, "metrics_summary.csv")
    header = ["predictor", "threshold", "FAR_episode", "FAR_step", "detection_rate",
              "LT_mean", "LT_max", "LT_min", "LT_first_mean", "n_fall", "n_safe"]
    row = [predictor, threshold, metrics["FAR_episode"], metrics["FAR_step"],
           metrics["detection_rate"], metrics["LT_mean"], metrics["LT_max"],
           metrics["LT_min"], metrics["LT_first_mean"], metrics["n_fall"], metrics["n_safe"]]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
    print(f"[INFO] Metrics row appended to {csv_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
