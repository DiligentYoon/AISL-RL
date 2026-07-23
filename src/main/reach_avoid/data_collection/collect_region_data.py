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
parser.add_argument("--num_scenarios", type=int, default=10, help="Number of sequential scenario sweeps to collect, one file each.")
parser.add_argument("--task", type=str, default="G1-fall-region-collect", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to nominal policy checkpoint.")
parser.add_argument("--predictor_checkpoint", type=str, default=None, help="Path to trained Reach-Avoid value checkpoint.")

parser.add_argument("--algorithm",
                    type=str,
                    default="MAPPO",
                    choices=["PPO", "SAC", "TD3", "MAPPO"],
                    help="The RL algorithm of the nominal policy.")

parser.add_argument("--model",
                    type=str,
                    default="Shared",
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
import datetime
import torch

import lib

from wrapper.isaaclab_wrapper import IsaacLabWrapper
from wrapper.record_wrapper import RecordVideo
from lib.utils.parse_utils import parse_env_cfg, load_cfg_from_registry
from lib.buffer.rolloutbuffer import RolloutBuffer
from lib.buffer.reach_avoid.regionbuffer import RegionBuffer
from lib.model.model_factory import ModelFactory


algorithm = args_cli.algorithm.lower()
model = args_cli.model.lower() if args_cli.model is not None else None


# ============================ Helpers ============================


def extract_snapshot(ra_state, ra_value) -> dict[str, torch.Tensor]:
    """Reach-avoid information"""
    return {
        "reach_avoid_state": ra_state.clone(),
        "reach_avoid_value": ra_value.clone()
    }


def record_disturbance(region_buffer: RegionBuffer, infos: dict) -> None:
    """Record each environment's disturbance once, the first time the push fires.

    ``disturbance_apply_idx`` starts at -1, so it doubles as the "not recorded yet"
    marker and no extra flag has to be consumed.

    Must be called after :meth:`RegionBuffer.add`: the RA state cache is filled before
    the push event runs, so ra_states lag the push by one step and the first
    post-disturbance sample lands at ``write_idx + 1``.
    """
    applied = infos["disturbance_applied"].flatten()
    pending = (applied
               & (region_buffer.metadata["disturbance_apply_idx"].squeeze(-1) < 0)
               & region_buffer.active_mask)
    if not pending.any():
        return

    env_ids = pending.nonzero().flatten()
    delta = infos["disturbance"][env_ids][:, [0, 1, 3, 4]]        # (vx, vy, roll, pitch)
    region_buffer.set_disturbance(delta,
                                  apply_idx=region_buffer.write_idx[env_ids] + 1,
                                  env_ids=env_ids)


def print_progress(region_buffer: RegionBuffer, timestep, max_timestep, elapsed_sec):
    total = region_buffer.num_envs
    pushed = int((region_buffer.metadata["disturbance_apply_idx"] >= 0).sum().item())
    finished = int((~region_buffer.active_mask).sum().item())
    fallen = int((region_buffer.metadata["falling_idx"] >= 0).sum().item())
    print(f"[{timestep:4d}/{max_timestep}] elapsed {elapsed_sec:6.1f}s | "
          f"disturbed {pushed:>5d}/{total} | finished {finished:>5d}/{total} | fallen {fallen:>5d}")


def print_scenario_summary(region_buffer: RegionBuffer) -> None:
    total = region_buffer.num_envs
    pushed = int((region_buffer.metadata["disturbance_apply_idx"] >= 0).sum().item())
    fallen = int(region_buffer.metadata["terminated"].sum().item())
    timed_out = int(region_buffer.metadata["truncated"].sum().item())
    print("[SUMMARY]")
    print(f"  envs       : {total}")
    print(f"  disturbed  : {pushed}")
    print(f"  fallen     : {fallen}")
    print(f"  timed out  : {timed_out}")
    if pushed < total:
        print(f"[WARN] {total - pushed} envs ended before the disturbance was applied "
              f"(filter on disturbance_apply_idx >= 0).")
    if fallen == 0 or fallen == total:
        print("[WARN] no fall/no-fall split; the disturbance range may be badly scaled.")


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
    
    C = ra_cfg["region_collection"]

    # save_dir = <predictor_checkpoint dir>/<save_subdir>/<timestamp>/
    # One run gets its own timestamp folder so re-runs never mix files.
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(args_cli.predictor_checkpoint)),
        C.get("save_subdir", "region"),
        timestamp,)

    # ============================ Env & Wrapper Spawn ================================
    seed = args_cli.seed if args_cli.seed is not None else ra_cfg.get("seed", 42)
    env_cfg.seed = seed
    cfg["agent"]["seed"] = seed
    ra_cfg["ra"]["agent"]["seed"] = seed

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
    from lib.buffer.reach_avoid.replaybuffer import HindSightReplayBuffer

    if not hasattr(env._unwrapped.cfg, "ra_state_space"):
        raise RuntimeError("Explicit state space is not defined.")

    # ReachAvoid requires a buffer arg; collection does not write to it.
    ra_buffer = HindSightReplayBuffer(1, env.num_envs, device=env.device)
    ra_buffer.init_buffer(env._unwrapped.cfg.ra_state_space)
    ra_model = {"critic": RA_Critic(env._unwrapped.cfg.ra_state_space, env.device)}
    ra_agent = ReachAvoid(ra_model, ra_buffer, device=env.device, cfg=ra_cfg["ra"]["agent"])

    # Load checkpoints (both required)
    agent.load(os.path.abspath(args_cli.checkpoint))
    print(f"[INFO] Loaded nominal policy from {args_cli.checkpoint}")
    ra_agent.load(os.path.abspath(args_cli.predictor_checkpoint))
    print(f"[INFO] Loaded RA critic from {args_cli.predictor_checkpoint}")

    agent.set_running_mode("eval")
    ra_agent.set_running_mode("eval")

    # ============= Region buffer ===============
    # One extra slot for the terminal sample appended when an episode ends.
    num_steps = env._unwrapped.max_episode_length + 1
    region_buffer = RegionBuffer(env.num_envs, 
                                 num_steps,
                                 disturbance_dim=4,
                                 device=env.device)

    # ============= Collection loops ===============
    # Every env terminates or truncates within one episode, so is_complete always
    # fires; num_steps is only a safety bound.
    t_anchor = 2.1
    t_phi_delta = [0.07, 0.175, 0.28, 0.42, 0.525, 0.63]
    t_phi = [t_anchor + x for x in t_phi_delta]
    video_end = False
    max_timestep = num_steps
    log_interval = 50
    num_scenarios = args_cli.num_scenarios

    for t_i_phi in t_phi:
        print(f"\n[DISTURBANCE PHASE] : {t_i_phi}")
        env._unwrapped.event_manager._mode_term_cfgs["interval"][0].interval_range_s = (t_i_phi, t_i_phi)
        for scenario in range(num_scenarios):
            if not simulation_app.is_running():
                break

            print(f"\n[SCENARIO {scenario + 1}/{num_scenarios}]")

            # Scenario init. The seed is set once at startup and the global RNG stream
            # keeps advancing, so each scenario draws different disturbances without any
            # per-scenario reseeding. env._reset_once reopens the wrapper's one-shot reset
            # guard so this env.reset() actually re-resets the simulation.
            region_buffer.reset()
            env._reset_once = True
            obs, states, infos = env.reset()

            timestep = 0
            t_start = time.time()

            while simulation_app.is_running() and timestep < max_timestep:
                with torch.no_grad():
                    actions,  _, _ = agent.act(obs, infos, timestep=timestep, deterministic=True)
                    ra_value, _, _ = ra_agent.critic(infos["ra_states"], update_rms=False)
                    snapshot = extract_snapshot(infos["ra_states"], ra_value)

                    next_obs, next_states, _, terminated, truncated, next_infos = env.step(actions)

                    # Captured before the env auto-reset, so this is the state that ended the episode.
                    terminal_value, _, _ = ra_agent.critic(next_infos["terminal_ra_state"], update_rms=False)
                    terminal_snapshot = extract_snapshot(next_infos["terminal_ra_state"], terminal_value)

                # Order matters: record_disturbance reads write_idx as advanced by add().
                region_buffer.add(snapshot)
                record_disturbance(region_buffer, next_infos)
                region_buffer.add_terminal(terminal_snapshot, terminated, truncated)

                timestep += 1

                if timestep % log_interval == 0:
                    print_progress(region_buffer, timestep, max_timestep, time.time() - t_start)
                
                if region_buffer.is_complete:
                    video_end = True
                    print("[INFO] All environments finished. Stopping.")
                    break
                else:
                    if (args_cli.video and timestep == (args_cli.video_length-1)):
                        print("[INFO] Video recording end. Stopping")
                        video_end = True   
                        break
                
                obs = next_obs
                infos = next_infos

            # =============== Summary (per scenario) ===============
            print_scenario_summary(region_buffer)
            
            if args_cli.video and video_end:
                break

            # ============= Save (per scenario) ===============
            phase_dir = os.path.join(save_dir, f"phase_{t_i_phi:.3f}")
            filename = f"region_buffer_{scenario:03d}.pt"
            region_buffer.save(phase_dir, filename=filename)
            print(f"[INFO] scenario {scenario}: saved {os.path.join(phase_dir, filename)}")
        
        if args_cli.video and video_end:
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
