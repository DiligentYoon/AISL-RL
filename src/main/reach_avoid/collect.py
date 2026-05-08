"""
Script to collect risk-classified initial-condition states using a frozen
nominal policy and a trained Reach-Avoid value function.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect risk-classified initial-condition states.")
parser.add_argument("--seed", type=int, default=None, help="Seed of RL environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during collection.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of environments (overrides cfg default if given).")
parser.add_argument("--task", type=str, default="G1-fall-collect", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to nominal policy checkpoint.")
parser.add_argument("--ra_checkpoint", type=str, required=True, help="Path to trained Reach-Avoid value checkpoint.")

parser.add_argument("--algorithm",
                    type=str,
                    default="MAPPO",
                    choices=["PPO", "SAC", "TD3", "MAPPO"],
                    help="The RL algorithm of the nominal policy.")

parser.add_argument("--model",
                    type=str,
                    default="MLP",
                    choices=["MLP", "Shared", "Communet"],
                    help="The NN model of the nominal policy.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
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

import lib

from wrapper.isaaclab_wrapper import IsaacLabWrapper
from wrapper.record_wrapper import RecordVideo
from lib.utils.parse_utils import parse_env_cfg, load_cfg_from_registry
from lib.buffer.rolloutbuffer import RolloutBuffer
from lib.buffer.risk_buffer import RiskClassifiedBuffer
from lib.model.model_factory import ModelFactory


algorithm = args_cli.algorithm.lower()
model = args_cli.model.lower() if args_cli.model is not None else None


# ============================ Helpers ============================


def extract_physical_snapshot(info) -> dict[str, torch.Tensor]:
    """Read root + joint state"""
    return {
        "root_pos_offset_w": info["root_pos_offset_w"],
        "root_quat_w":       info["root_quat_w"],
        "root_lin_vel_w":    info["root_lin_vel_w"],
        "root_ang_vel_w":    info["root_ang_vel_w"],
        "joint_pos":         info["joint_pos"],
        "joint_vel":         info["joint_vel"],
        "prev_action":       info["prev_action"],
    }


def build_valid_mask(skip_remaining: torch.Tensor,
                     terminated: torch.Tensor,
                     stride_counter: torch.Tensor,
                     stride: int) -> torch.Tensor:
    """Per-env mask combining warmup-skip, terminated exclusion, subsample stride."""
    not_warmup = (skip_remaining == 0)
    not_terminated = ~terminated.flatten().bool()
    on_stride = (stride_counter % stride == 0)
    return not_warmup & not_terminated & on_stride


def update_counters(skip_remaining: torch.Tensor,
                    stride_counter: torch.Tensor,
                    done: torch.Tensor,
                    reset_value: int) -> None:
    """Advance per-env counters for next step.

    - Decrement warmup-skip while > 0.
    - Reset both counters for envs that just finished an episode.
    - Increment stride counter for next step.
    """
    pos = skip_remaining > 0
    skip_remaining[pos] -= 1

    done_flat = done.flatten().bool()
    skip_remaining[done_flat] = reset_value
    stride_counter[done_flat] = 0

    stride_counter += 1


def print_progress_box(fill_status, timestep, max_timestep, elapsed_sec, eta_sec):
    content_width = 64
    e_h = int(elapsed_sec // 3600); e_m = int((elapsed_sec % 3600) // 60); e_s = int(elapsed_sec % 60)
    c_h = int(eta_sec // 3600); c_m = int((eta_sec % 3600) // 60); c_s = int(eta_sec % 60)
    line_step = f"Step Progress {timestep} / {max_timestep}"
    line_time = f"Time Progress  {e_h:02d}:{e_m:02d}:{e_s:02d}/{c_h:02d}:{c_m:02d}:{c_s:02d}"
    print(" ________________________________________________________________")
    print("|                                                                |")
    print(f"|{line_step.center(content_width)}|")
    print(f"|{line_time.center(content_width)}|")
    print("|________________________________________________________________|")
    print("|                                                                |")
    for b in ("low", "mid", "high"):
        cur, cap = fill_status[b]
        ratio = 100.0 * cur / max(cap, 1)
        tag = "FULL" if cur >= cap else f"{ratio:5.1f}%"
        line = f"{b.capitalize():4s} : {cur:>7d} / {cap:<7d}  ({tag})"
        print(f"| {line:<{content_width-1}}|")
    print("|________________________________________________________________|")


# ============================ Main ============================


def main():
    """Main collection routine."""

    # ============================= Config Parsing ===============================
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric)

    try:
        cfg = load_cfg_from_registry(args_cli.task, f"rl_{algorithm}_cfg_entry_point")
        ra_cfg = load_cfg_from_registry(args_cli.task, "ra_cfg_entry_point")
    except ValueError as e:
        print(e)
        return
    
    C = ra_cfg["collection"]

    # save_dir = next to ra_checkpoint
    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(args_cli.ra_checkpoint)),
        C.get("save_subdir", "collected"),)

    # ============================ Env & Wrapper Spawn ================================
    seed = args_cli.seed if args_cli.seed is not None else ra_cfg.get("seed", 42)
    env_cfg.seed = seed
    cfg["agent"]["seed"] = seed
    ra_cfg["agent"]["seed"] = seed

    env_cfg.total_timesteps = cfg["train"]["timesteps"]
    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(save_dir, "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during collection.")
        env = RecordVideo(env, **video_kwargs)

    env = IsaacLabWrapper(env)

    # ======================= Policy buffer / model / agent (frozen) =========================
    multi_agent = algorithm == "mappo"
    cfg["models"]["multi_agent"] = multi_agent
    if cfg["buffer"]["buffer_size"] == -1:
        cfg["buffer"]["buffer_size"] = cfg["agent"]["rollouts"]
    else:
        raise RuntimeError("Replaybuffer for Off-policy algorithm is not implemented yet.")

    possible_agents = None
    if multi_agent:
        obs_size, state_size, act_size = {}, {}, {}
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
            buf = RolloutBuffer(cfg["buffer"]["buffer_size"], env.num_envs, device=env.device)
            buf.init_buffer(observation_space, state_space, action_space)
            buffers[uid] = buf
            obs_size[uid] = buf.tensors["observations"].shape[-1]
            state_size[uid] = buf.tensors["states"].shape[-1] if env.state_space else obs_size[uid]
            act_size[uid] = buf.tensors["actions"].shape[-1]
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

    if model is not None:
        cfg["models"]["model_type"] = model

    model_manager = ModelFactory(cfg=cfg["models"], device=env.device)
    if model_manager.model_class == "mlp":
        models = model_manager.generate_mlp_models(
            observation_size=obs_size, state_size=state_size,
            action_size=act_size, possible_agents=possible_agents,
        )
    else:
        raise RuntimeError("Not supported class")

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
        elif model_manager.model_type == "shared":
            from lib.agent.cooperative_mappo import CooperativeMAPPO
            agent = CooperativeMAPPO(observation_space=env.observation_space,
                                     state_space=env.state_space,
                                     action_space=env.action_space,
                                     possible_agents=possible_agents,
                                     model=models, buffer=buffers,
                                     device=env.device, cfg=cfg["agent"])
        else:
            raise RuntimeError("Unvalid model type.")
    else:
        from lib.agent.ppo import PPO
        agent = PPO(model=models, buffer=buffer,
                    device=env.device, cfg=cfg["agent"])

    # ============= RA Model & Agent (frozen) ===============
    from lib.model.MLP import RA_Critic
    from lib.agent.reach_avoid import ReachAvoid
    from lib.buffer.replaybuffer import HindSightReplayBuffer

    if not hasattr(env._unwrapped.cfg, "ra_state_space"):
        raise RuntimeError("Explicit state space is not defined.")

    # ReachAvoid requires a buffer arg; collection does not write to it.
    ra_buffer = HindSightReplayBuffer(1, env.num_envs, device=env.device)
    ra_buffer.init_buffer(env._unwrapped.cfg.ra_state_space)
    ra_model = {"critic": RA_Critic(env._unwrapped.cfg.ra_state_space, env.device)}
    ra_agent = ReachAvoid(ra_model, ra_buffer, device=env.device, cfg=ra_cfg["agent"])

    # Load checkpoints (both required)
    agent.load(os.path.abspath(args_cli.checkpoint))
    print(f"[INFO] Loaded nominal policy from {args_cli.checkpoint}")
    ra_agent.load(os.path.abspath(args_cli.ra_checkpoint))
    print(f"[INFO] Loaded RA critic from {args_cli.ra_checkpoint}")

    agent.set_running_mode("eval")
    ra_agent.set_running_mode("eval")

    # ============= Risk-classified buffer ===============
    joint_dim = env._unwrapped._robot.data.joint_pos.shape[-1]
    risk_buffer = RiskClassifiedBuffer(
        capacity_per_bucket=int(C["capacity_per_bucket"]),
        thresholds=(float(C["thresholds"]["low_high"]),
                    float(C["thresholds"]["mid_high"])),
        joint_dim=joint_dim,
        device=env.device,
    )

    # ============= Collection loop ===============
    warmup_skip = int(C.get("warmup_skip_steps", 4))
    subsample_stride = max(1, int(C.get("subsample_stride", 1)))
    max_timestep = int(C["max_timestep"])
    log_interval = 500

    skip_remaining = torch.full((env.num_envs,), warmup_skip,
                                dtype=torch.long, device=env.device)
    stride_counter = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)

    obs, states, infos = env.reset()
    timestep = 0
    t_start = time.time()

    while simulation_app.is_running() and timestep < max_timestep:
        with torch.no_grad():
            actions, _, _, _ = agent.act(obs, infos, timestep=timestep, deterministic=True)
            risk_scores, _, _ = ra_agent.critic(infos["ra_states"], update_rms=False)
            snapshot = extract_physical_snapshot(infos["collection"])
            next_obs, next_states, _, terminated, truncated, next_infos = env.step(actions)

        valid_mask = build_valid_mask(skip_remaining, terminated,
                                      stride_counter, subsample_stride)
        risk_buffer.add(snapshot=snapshot, risk_scores=risk_scores, valid_mask=valid_mask)

        update_counters(skip_remaining, stride_counter,
                        done=(terminated | truncated),
                        reset_value=warmup_skip)

        timestep += 1

        if timestep % log_interval == 0:
            elapsed = time.time() - t_start
            eta = elapsed / max(timestep, 1) * max(max_timestep - timestep, 0)
            print_progress_box(risk_buffer.fill_status(), timestep, max_timestep, elapsed, eta)

        if risk_buffer.is_full():
            print("[INFO] All buckets reached capacity. Stopping.")
            break

        obs = next_obs
        states = next_states
        infos = next_infos

    # ============= Save & summary ===============
    risk_buffer.save(save_dir)
    print(f"[INFO] Saved risk-classified buckets to {save_dir}")

    final = risk_buffer.fill_status()
    underfilled = risk_buffer.underfilled_buckets()
    print("[SUMMARY]")
    for b in ("low", "mid", "high"):
        cur, cap = final[b]
        print(f"  {b:4s} : {cur} / {cap}")
    if underfilled:
        print(f"[WARN] underfilled buckets: {underfilled}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
