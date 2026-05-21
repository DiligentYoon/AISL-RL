"""
Standalone MuJoCo evaluation script that mirrors the 3-agent setup of
src/main/reach_avoid/play_unified.py:
    - nominal policy
    - safe fallback policy
    - fall predictor (RA / SafeFall)

Each policy is loaded from a separate .pt file. Observations are stubbed
out for now -- actions returned by each agent are zero placeholders, and
the predictor always returns risk=0.0. The simulation loop, switching
scaffold, and random base/pelvis force-push are real.

No ROS2: this is a plain Python script. Only the passive-viewer +
wall-clock catch-up pattern is borrowed from mujoco_ros2_bridge.py.
"""

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """CLI mirrors play_unified.py's checkpoint flag names where it makes sense."""
    p = argparse.ArgumentParser(description="MuJoCo standalone play of the unified policy framework.")
    # Scene + checkpoints
    p.add_argument("--scene", type=str,
                   default="src/lib/assets/robots/G1/G1_hand/xml/g1_box_foot_scene.xml",
                   help="Path to the MuJoCo XML scene file.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to the nominal policy .pt checkpoint.")
    p.add_argument("--predictor_checkpoint", type=str, default=None,
                   help="Path to the fall predictor .pt checkpoint.")
    p.add_argument("--safe_checkpoint", type=str, default=None,
                   help="Path to the safe fclallback policy .pt checkpoint.")
    # Loop pacing
    p.add_argument("--sim_rate_hz", type=float, default=200.0,
                   help="Wall-clock catch-up target rate for the outer loop.")
    # Switching rule
    p.add_argument("--switch_threshold", type=float, default=0.5,
                   help="Predictor risk threshold above which we switch to the safe policy.")
    # Random push
    p.add_argument("--push_body", type=str, default="pelvis",
                   help="Body to apply the random horizontal force-push to.")
    p.add_argument("--push_period", type=float, default=5.0,
                   help="Seconds between push events (sim time).")
    p.add_argument("--push_duration", type=float, default=0.2,
                   help="Seconds each push force is held (sim time).")
    p.add_argument("--push_force_max", type=float, default=200.0,
                   help="Max horizontal force magnitude in Newtons.")
    # Misc
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the push RNG (reproducibility).")
    p.add_argument("--device", type=str, default="cuda",
                   help="Torch device override; defaults to cuda if available.")
    return p.parse_args()


# ---------------------------------------------------------------------- #
# Checkpoint loading                                                     #
# ---------------------------------------------------------------------- #
def load_pt(ckpt_path: str | None, role: str, device: torch.device):
    """Best-effort .pt loader. Returns raw torch.load payload, or None on miss."""
    if ckpt_path is None:
        print(f"[WARN] {role}: no checkpoint provided; using zero-action stub")
        return None
    if not os.path.isfile(ckpt_path):
        print(f"[WARN] {role}: checkpoint '{ckpt_path}' not found; using zero-action stub")
        return None
    try:
        payload = torch.load(ckpt_path, map_location=device)
        print(f"[INFO] {role}: loaded checkpoint from {ckpt_path}")
        return payload
    except Exception as e:
        print(f"[ERROR] {role}: failed to load checkpoint: {e}")
        return None


# ---------------------------------------------------------------------- #
# Agent stubs (real inference TBD)                                       #
# ---------------------------------------------------------------------- #
def agent_act(agent_payload, obs: torch.Tensor, nu: int, device: torch.device) -> torch.Tensor:
    """Zero-action placeholder; replace once obs builder + real model are wired."""
    return torch.zeros((1, nu), device=device)


def predictor_value(predictor_payload, obs: torch.Tensor) -> float:
    """Zero-risk placeholder; replace with real RA / SafeFall head later."""
    return 0.0


# ---------------------------------------------------------------------- #
# Random force-push                                                      #
# ---------------------------------------------------------------------- #
class RandomPush:
    """
    Periodic horizontal force-push applied to a target body via xfrc_applied.
    State is held externally so the main loop owns it.
    """

    def __init__(self, model: mujoco.MjModel, body_name: str,
                 period_sec: float, duration_sec: float, force_max_n: float,
                 rng: np.random.Generator):
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.period_sec = period_sec
        self.duration_sec = duration_sec
        self.force_max_n = force_max_n
        self.rng = rng

        # Schedule state
        self.next_at = period_sec        # sim time of next push start
        self.active_until = -1.0         # sim time at which the current push ends
        self.force = np.zeros(3)         # current world-frame force (N)

        if self.body_id < 0:
            print(f"[WARN] push body '{body_name}' not found; random push disabled")

    def apply(self, data: mujoco.MjData):
        """Call once per mj_step, before stepping."""
        if self.body_id < 0:
            return

        t = float(data.time)

        # Start a new push if scheduled and previous one ended.
        if t >= self.next_at and t >= self.active_until:
            direction = self.rng.normal(size=3)
            direction[2] *= 0.1                                # bias toward horizontal
            direction /= max(np.linalg.norm(direction), 1e-8)  # unit vector
            magnitude = float(self.rng.uniform(0.5, 1.0) * self.force_max_n)
            self.force = direction * magnitude
            self.active_until = t + self.duration_sec
            self.next_at = t + self.period_sec
            print(
                f"[PUSH] t={t:7.3f}s  |F|={magnitude:6.1f}N  "
                f"dir=({direction[0]:+.2f},{direction[1]:+.2f},{direction[2]:+.2f})"
            )

        # Write the 6-DoF wrench (force only; torque kept at zero).
        if t < self.active_until:
            data.xfrc_applied[self.body_id, :3] = self.force
            data.xfrc_applied[self.body_id, 3:] = 0.0
        else:
            data.xfrc_applied[self.body_id, :] = 0.0


# ---------------------------------------------------------------------- #
# Main                                                                   #
# ---------------------------------------------------------------------- #
def main():
    args = parse_args()

    # Torch device -- explicit override > cuda if available > cpu.
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] torch device: {device}")

    # Numpy RNG for the push schedule (so seeding gives reproducible push patterns).
    rng = np.random.default_rng(args.seed)

    # ---- MuJoCo scene + keyframe init ----
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    nu = model.nu                                    # number of actuators
    print(f"[INFO] scene: {args.scene}  nu={nu}  nq={model.nq}  nv={model.nv}")

    # ---- Load three .pt checkpoints (dummy payload usage for now) ----
    nominal_payload = load_pt(args.checkpoint, role="nominal", device=device)
    predictor_payload = load_pt(args.predictor_checkpoint, role="predictor", device=device)
    safe_payload = load_pt(args.safe_checkpoint, role="safe", device=device)

    # ---- Random push helper ----
    push = RandomPush(
        model, body_name=args.push_body,
        period_sec=args.push_period,
        duration_sec=args.push_duration,
        force_max_n=args.push_force_max,
        rng=rng,
    )

    # ---- Switching state (latched, mirrors play_unified.py) ----
    prev_switch = False
    # TODO: reset prev_switch on episode boundary once an "episode" is defined here.

    # ---- Passive viewer ----
    viewer = mujoco.viewer.launch_passive(model, data)

    # ---- Real-time pacing anchors (same idea as the GOAT bridge) ----
    tick_period = 1.0 / max(args.sim_rate_hz, 1.0)
    wall_anchor = time.monotonic()
    sim_anchor = float(data.time)
    max_substeps_per_tick = 20                       # cap to avoid spiral-of-death

    try:
        while viewer.is_running():
            # Catch sim time up to wall clock by stepping until we hit target or cap.
            target_sim_t = sim_anchor + (time.monotonic() - wall_anchor)
            substeps = 0
            while data.time < target_sim_t and substeps < max_substeps_per_tick:
                # 1) 3-agent decision -- observations are dummy zeros for now.
                dummy_obs = torch.zeros((1, 1), device=device)
                with torch.no_grad():
                    nominal = agent_act(nominal_payload, dummy_obs, nu, device)
                    safe = agent_act(safe_payload, dummy_obs, nu, device)
                    risk = predictor_value(predictor_payload, dummy_obs)

                # 2) Latched switching: once risk crosses threshold we stay on safe.
                cur_switch = bool(risk > args.switch_threshold)
                switch = cur_switch or prev_switch
                prev_switch = switch
                actions_t = safe if switch else nominal

                # 3) Defensive shape match before writing to mjData.ctrl.
                actions_np = actions_t.detach().cpu().numpy().reshape(-1)
                if actions_np.shape[0] != nu:
                    actions_np = np.zeros(nu, dtype=np.float64)
                np.copyto(data.ctrl, actions_np)

                # 4) Apply scheduled random push (no-op if outside the active window).
                push.apply(data)

                # 5) Step physics.
                mujoco.mj_step(model, data)
                substeps += 1

            # Render and yield a sliver of CPU.
            viewer.sync()
            time.sleep(max(0.0, tick_period * 0.5))
    except KeyboardInterrupt:
        print("[INFO] interrupted by user")
    finally:
        viewer.close()
        print("[INFO] viewer closed; exiting")


if __name__ == "__main__":
    main()
