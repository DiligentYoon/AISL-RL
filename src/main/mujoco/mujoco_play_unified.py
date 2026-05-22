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
import torch.nn as nn


# ---------------------------------------------------------------------- #
# Fixed tuning constants (edit here, not via CLI)                        #
# ---------------------------------------------------------------------- #
# Body-frame velocity command driving the nominal policy's command_inputs.
CMD_VX = 0.8
CMD_VY = 0.0
CMD_WZ = 0.0

# Gait phase frequency (Hz) producing phase_sin / phase_cos for the nominal obs.
PHASE_FREQ_HZ = 1.5

# Length of the RA predictor's root_state_buffer history (rows of 8 features)
RA_HISTORY_LEN = 4

# Per-uid action scaling — must match the action_scale_factor used at training time.
ACTION_SCALE_ARM = 0.5
ACTION_SCALE_LEG = 0.5

# Tanh squashing on actor outputs. Must match the training cfg's `squash` flag.
SQUASH = True


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
    p.add_argument("--checkpoint", type=str, default="/home/oksusu/Downloads/agent_32000.pt",
                   help="Path to the nominal policy .pt checkpoint.")
    p.add_argument("--predictor_checkpoint", type=str, default="/home/oksusu/Downloads/ra_agent_16000.pt",
                   help="Path to the fall predictor .pt checkpoint.")
    p.add_argument("--instinct_checkpoint", type=str, default="/home/oksusu/Downloads/agent_56000.pt",
                   help="Path to the safe fclallback policy .pt checkpoint.")
    # Loop pacing
    p.add_argument("--sim_rate_hz", type=float, default=200.0,
                   help="Wall-clock catch-up target rate for the outer loop.")
    # Switching rule
    p.add_argument("--switch_threshold", type=float, default=0.1,
                   help="Predictor risk threshold above which we switch to the safe policy. "
                        "[DEBUG] default raised to 1e9 to disable the safe-latch while we isolate "
                        "other suspects; restore to ~0.1 once the obs/policy path is verified.")
    # Random push
    p.add_argument("--push_body", type=str, default="pelvis",
                   help="Body to apply the random horizontal force-push to.")
    p.add_argument("--push_period", type=float, default=5.0,
                   help="Seconds between push events (sim time).")
    p.add_argument("--push_duration", type=float, default=0.2,
                   help="Seconds each push force is held (sim time).")
    p.add_argument("--push_force_max", type=float, default=300.0,
                   help="Max horizontal force magnitude in Newtons.")
    # Misc
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the push RNG (reproducibility).")
    p.add_argument("--device", type=str, default="cuda",
                   help="Torch device override; defaults to cuda if available.")
    return p.parse_args()


# ---------------------------------------------------------------------- #
# Inline policy / critic architectures (mirror lib.model.MLP exactly)    #
# ---------------------------------------------------------------------- #
# These are inference-only re-implementations of the classes used at training
# time. We re-declare them here so the MuJoCo runner stays free of lib.* imports
# (which transitively pull in Isaac Lab via lib/__init__.py).

class _RunningMeanStd(nn.Module):
    """Inference-only mirror of lib.utils.Running_mean_std.RunningMeanStd."""

    def __init__(self, shape: int, epsilon: float = 1e-4):
        super().__init__()
        # Buffer names must match the training class so state_dicts load 1:1.
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float32))

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        # (x - mean) / sqrt(var + epsilon). The training-time update path is a no-op here.
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)


class _SharedBackbone(nn.Module):
    """Mirror of lib.model.MLP.SharedBackbone."""

    def __init__(self, in_dim: int, d_arm: int = 128, d_leg: int = 128):
        super().__init__()
        # NOTE: the training class uses d_arm as the output dim for both heads
        # (see SharedBackbone.__init__). We replicate that exactly so weights load.
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
        )
        self.head_arm = nn.Sequential(nn.Linear(128, d_arm), nn.ELU())
        self.head_leg = nn.Sequential(nn.Linear(128, d_arm), nn.ELU())

    def forward(self, x: torch.Tensor, role: str) -> torch.Tensor:
        g = self.shared(x)
        return self.head_arm(g) if role == "arm" else self.head_leg(g)


class _SharedActor(nn.Module):
    """Inference-only mirror of lib.model.MLP.SharedActor (deterministic path)."""

    def __init__(self, num_obs_arm: int, num_obs_leg: int,
                 num_act_arm: int, num_act_leg: int,
                 encoder_hidden_dim: int, squash: bool):
        super().__init__()
        self.squash = squash

        # Per-role observation normalizer.
        self.actor_standardizer = nn.ModuleDict({
            "arm": _RunningMeanStd(num_obs_arm),
            "leg": _RunningMeanStd(num_obs_leg),
        })

        # Per-role encoder: obs -> encoder_hidden_dim.
        self.encoder = nn.ModuleDict({
            "arm": nn.Sequential(nn.Linear(num_obs_arm, encoder_hidden_dim), nn.ELU()),
            "leg": nn.Sequential(nn.Linear(num_obs_leg, encoder_hidden_dim), nn.ELU()),
        })

        # Shared dual-head trunk over [z_self | z_other].
        self.shared_backbone = _SharedBackbone(in_dim=encoder_hidden_dim * 2)

        # Per-role action head: [z_self | h_self] -> action mean.
        self.head = nn.ModuleDict({
            "arm": nn.Sequential(
                nn.Linear(encoder_hidden_dim + 128, 128), nn.ELU(),
                nn.Linear(128, 64), nn.ELU(),
                nn.Linear(64, num_act_arm),
            ),
            "leg": nn.Sequential(
                nn.Linear(encoder_hidden_dim + 128, 128), nn.ELU(),
                nn.Linear(128, 64), nn.ELU(),
                nn.Linear(64, num_act_leg),
            ),
        })

        # State-independent log std is in the saved state_dict; we keep it so load_state_dict
        # is strict-clean, but it is unused in deterministic inference.
        self.log_std_parameter = nn.ParameterDict({
            "arm": nn.Parameter(torch.zeros(num_act_arm), requires_grad=False),
            "leg": nn.Parameter(torch.zeros(num_act_leg), requires_grad=False),
        })

    @torch.no_grad()
    def act_deterministic(self, obs_arm: torch.Tensor, obs_leg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Standardize -> encode -> shared trunk -> per-role head -> optional tanh squash.
        z_arm = self.encoder["arm"](self.actor_standardizer["arm"].standardize(obs_arm))
        z_leg = self.encoder["leg"](self.actor_standardizer["leg"].standardize(obs_leg))
        h_arm = self.shared_backbone(torch.cat([z_arm, z_leg], dim=-1), role="arm")
        h_leg = self.shared_backbone(torch.cat([z_leg, z_arm], dim=-1), role="leg")
        a_arm = self.head["arm"](torch.cat([z_arm, h_arm], dim=-1))
        a_leg = self.head["leg"](torch.cat([z_leg, h_leg], dim=-1))
        if self.squash:
            a_arm = torch.tanh(a_arm)
            a_leg = torch.tanh(a_leg)
        return a_arm, a_leg


class _RA_Critic(nn.Module):
    """Inference-only mirror of lib.model.MLP.RA_Critic."""

    def __init__(self, num_states: int):
        super().__init__()
        self.critic_standardizer = _RunningMeanStd(num_states)
        self.net = nn.Sequential(
            nn.Linear(num_states, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 1),
        )

    @torch.no_grad()
    def value(self, x: torch.Tensor) -> torch.Tensor:
        # Returns V(s) as a (B, 1) tensor; caller takes .item() when a float is needed.
        return self.net(self.critic_standardizer.standardize(x))


# ---------------------------------------------------------------------- #
# Checkpoint loaders                                                     #
# ---------------------------------------------------------------------- #
def load_cooperative_actor(checkpoint_path: str | None, role: str, device: torch.device,
                           squash: bool) -> "_SharedActor | None":
    """Load a CooperativeMAPPO state-dict and rebuild SharedActor for inference.

    Expects payload['shared']['actor'] to be a SharedActor state_dict.
    Architecture dimensions are inferred from saved tensor shapes.
    """
    if checkpoint_path is None:
        print(f"[WARN] {role}: no checkpoint provided; using zero-action stub")
        return None
    if not os.path.isfile(checkpoint_path):
        print(f"[WARN] {role}: checkpoint '{checkpoint_path}' not found; using zero-action stub")
        return None
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"[ERROR] {role}: failed to torch.load: {e}")
        return None
    if not isinstance(payload, dict) or "actor" not in payload.get("shared", {}):
        print(f"[ERROR] {role}: unexpected layout — expected payload['shared']['actor'] state_dict")
        return None
    sd = payload["shared"]["actor"]
    try:
        # Infer dims directly from saved tensors — robust to training-time changes.
        num_obs_arm = int(sd["actor_standardizer.arm.mean"].shape[0])
        num_obs_leg = int(sd["actor_standardizer.leg.mean"].shape[0])
        enc_dim = int(sd["encoder.arm.0.weight"].shape[0])
        # Final Linear of each per-role head sits at index 4 (Linear-ELU-Linear-ELU-Linear).
        num_act_arm = int(sd["head.arm.4.weight"].shape[0])
        num_act_leg = int(sd["head.leg.4.weight"].shape[0])
    except KeyError as e:
        print(f"[ERROR] {role}: missing expected key in actor state_dict: {e}")
        return None

    actor = _SharedActor(num_obs_arm, num_obs_leg, num_act_arm, num_act_leg,
                         encoder_hidden_dim=enc_dim, squash=squash)
    missing, unexpected = actor.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] {role}: missing keys when loading actor: {missing}")
    if unexpected:
        print(f"[WARN] {role}: unexpected keys when loading actor: {unexpected}")
    actor = actor.to(device).eval()
    print(f"[INFO] {role}: loaded actor from {checkpoint_path}  "
          f"obs=(arm={num_obs_arm}, leg={num_obs_leg})  "
          f"act=(arm={num_act_arm}, leg={num_act_leg})  enc={enc_dim}")
    return actor


def load_ra_critic(checkpoint_path: str | None, role: str, device: torch.device) -> "_RA_Critic | None":
    """Load a ReachAvoid state-dict and rebuild RA_Critic for inference.

    Accepts both {"critic": sd, "optimizer": ...} (Agent.save format) and a raw sd.
    """
    if checkpoint_path is None:
        print(f"[WARN] {role}: no checkpoint provided; using zero-risk stub")
        return None
    if not os.path.isfile(checkpoint_path):
        print(f"[WARN] {role}: checkpoint '{checkpoint_path}' not found; using zero-risk stub")
        return None
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"[ERROR] {role}: failed to torch.load: {e}")
        return None
    sd = payload["critic"] if isinstance(payload, dict) and "critic" in payload else payload
    if not isinstance(sd, dict):
        print(f"[ERROR] {role}: unexpected RA critic payload type: {type(sd)}")
        return None
    try:
        num_states = int(sd["net.0.weight"].shape[1])
    except KeyError as e:
        print(f"[ERROR] {role}: missing 'net.0.weight' in RA critic state_dict: {e}")
        return None

    critic = _RA_Critic(num_states)
    missing, unexpected = critic.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] {role}: missing keys when loading RA critic: {missing}")
    if unexpected:
        print(f"[WARN] {role}: unexpected keys when loading RA critic: {unexpected}")
    critic = critic.to(device).eval()
    print(f"[INFO] {role}: loaded RA critic from {checkpoint_path}  num_states={num_states}")
    return critic


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
# Joint ordering (must match Isaac Lab's total_leg_joint_ids + total_arm_joint_ids) #
# ---------------------------------------------------------------------- #
# Legs first (12), then arms+waist (17). Policies were trained against this exact
# ordering, so it must be preserved when feeding observations.
# Order matches Isaac Lab's articulation joint order exactly. The training-time
# total_leg_joint_ids / total_arm_joint_ids interleave left/right per joint type
# (e.g. left_hip_pitch, right_hip_pitch, left_hip_roll, ...), not all-left-then-all-right.
# These are the names resolved from the IDs:
#   total_leg_joint_ids = [0,1,3,4,6,7,9,10,13,14,17,18]
#   total_arm_joint_ids = [2,5,8,11,12,15,16,19,20,21,22,23,24,25,26,27,28]
LEG_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",   "right_hip_pitch_joint",
    "left_hip_roll_joint",    "right_hip_roll_joint",
    "left_hip_yaw_joint",     "right_hip_yaw_joint",
    "left_knee_joint",        "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint",  "right_ankle_roll_joint",
]
ARM_JOINT_NAMES: list[str] = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",  "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",   "right_shoulder_yaw_joint",
    "left_elbow_joint",          "right_elbow_joint",
    "left_wrist_roll_joint",     "right_wrist_roll_joint",
    "left_wrist_pitch_joint",    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",      "right_wrist_yaw_joint",
]


# ---------------------------------------------------------------------- #
# Observation builder helpers                                            #
# ---------------------------------------------------------------------- #
class JointIndex:
    """Resolve MuJoCo qpos/qvel addresses for a fixed list of joint names."""

    def __init__(self, model: mujoco.MjModel, names: list[str]):
        # qpos_idx[i] / qvel_idx[i] are the qpos/qvel slot of joint names[i].
        self.qpos_idx = np.empty(len(names), dtype=np.int64)
        self.qvel_idx = np.empty(len(names), dtype=np.int64)
        for i, n in enumerate(names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if jid < 0:
                raise ValueError(f"joint not found in scene: {n}")
            self.qpos_idx[i] = model.jnt_qposadr[jid]
            self.qvel_idx[i] = model.jnt_dofadr[jid]

    @property
    def n(self) -> int:
        # Number of joints in this group.
        return int(self.qpos_idx.shape[0])


def build_ctrl_index(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    """Return ctrl_idx[i] = data.ctrl slot that drives joint_names[i].

    Walks every actuator, reads its transmission target joint (actuator_trnid[a, 0])
    and inverts it into a {joint_id: actuator_id} map. Then orders the actuator ids
    to follow joint_names. Used to scatter Isaac-order q_des into MJCF-order ctrl.
    """
    # Map joint_id -> actuator_id for every joint-driving actuator in the model.
    joint_to_actuator: dict[int, int] = {}
    for a_id in range(model.nu):
        j_id = int(model.actuator_trnid[a_id, 0])
        if j_id < 0:
            # Tendon / site actuator with no joint target — skip.
            continue
        # If multiple actuators drive the same joint, first one wins. G1 MJCFs use
        # one position actuator per joint, so this is informational only.
        joint_to_actuator.setdefault(j_id, a_id)

    ctrl_idx = np.empty(len(joint_names), dtype=np.int64)
    for i, n in enumerate(joint_names):
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j_id < 0:
            raise ValueError(f"joint not found in scene: {n}")
        if j_id not in joint_to_actuator:
            raise ValueError(f"no actuator drives joint: {n}")
        ctrl_idx[i] = joint_to_actuator[j_id]
    return ctrl_idx


class RootState:
    """Extract base pose/velocities and rotate vectors into the body frame."""

    # World-frame gravity unit vector; projected gravity == R^T · this.
    _GRAVITY_W = np.array([0.0, 0.0, -1.0])

    def __init__(self):
        # Cached base state, refreshed each call to update().
        self.pos_w = np.zeros(3)                          # world-frame base position
        self.quat_w = np.array([1.0, 0.0, 0.0, 0.0])      # base orientation (w,x,y,z)
        self.lin_b = np.zeros(3)                          # base linear vel, body frame
        self.ang_b = np.zeros(3)                          # base angular vel, body frame
        self.proj_grav = np.array([0.0, 0.0, -1.0])       # projected gravity, body frame

    def update(self, data: mujoco.MjData):
        # Free-joint layout: qpos[0:3]=xyz_world, qpos[3:7]=quat(w,x,y,z);
        # qvel[0:3]=lin_world, qvel[3:6]=ang_world.
        self.pos_w = data.qpos[0:3].copy()
        self.quat_w = data.qpos[3:7].copy()
        lin_w = data.qvel[0:3].copy()
        ang_w = data.qvel[3:6].copy()

        # body->world rotation R from quaternion, then take transpose for world->body.
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.quat_w)
        R = R.reshape(3, 3)
        Rt = R.T
        self.lin_b = Rt @ lin_w
        self.ang_b = Rt @ ang_w
        self.proj_grav = Rt @ self._GRAVITY_W


class PhaseClock:
    """Drive sin/cos of a gait phase variable from sim time."""

    def __init__(self, freq_hz: float):
        # omega so that one full cycle takes 1/freq_hz seconds of sim time.
        self.omega = 2.0 * np.pi * float(freq_hz)

    def tick(self, sim_t: float) -> tuple[float, float, float]:
        # Return (phase, sin, cos) wrapped to [0, 2π).
        phase = (self.omega * float(sim_t)) % (2.0 * np.pi)
        return phase, float(np.sin(phase)), float(np.cos(phase))


class RootHistoryBuffer:
    """Rolling root-state history mirroring G1FallEnv.root_state_buffer.

    Stores a fixed-length window of 8-feature rows:
        [root_ang_vel_b (3), projected_gravity (3),
         dist_from_icp_to_stance (1), phase (1)]
    Writes use the same circular index as training: row = buffer[hist_count % length],
    so the flattened layout matches what the RA critic saw at training time.
    """

    def __init__(self, length: int):
        # length=0 disables the buffer and produces a zero-length flatten().
        self.length = int(length)
        # Pre-allocate with at least 1 row so update() can be a no-op safely.
        self.buffer = np.zeros((max(self.length, 1), 8), dtype=np.float64)
        self.hist_count = 0

    def update(self, ang_b: np.ndarray, proj_grav: np.ndarray,
               d_icp: float, phase: float):
        # No-op when the predictor was trained without history.
        if self.length == 0:
            return
        idx = self.hist_count % self.length
        # Pack the same 8 features in the same order as G1_fall_env.py:80.
        self.buffer[idx, 0:3] = ang_b
        self.buffer[idx, 3:6] = proj_grav
        self.buffer[idx, 6]   = d_icp
        self.buffer[idx, 7]   = phase
        self.hist_count += 1

    def reset(self):
        # Mirrors _reset_idx: hist_count -> 0 and zero the rows.
        self.buffer[:] = 0.0
        self.hist_count = 0

    def flatten(self) -> np.ndarray:
        # Row-major flatten -> shape (length*8,); empty array when disabled.
        if self.length == 0:
            return np.zeros(0, dtype=np.float64)
        return self.buffer.reshape(-1)


class ActionBuffer:
    """Hold the last applied 29-dim action and expose leg/arm splits + the full vector."""

    def __init__(self, leg_idx: JointIndex, arm_idx: JointIndex):
        self.leg = leg_idx
        self.arm = arm_idx
        # Storage laid out as [leg_actions | arm_actions], matching Isaac concat order.
        self.prev_full = np.zeros(leg_idx.n + arm_idx.n)
        self.prev_leg = np.zeros(leg_idx.n)
        self.prev_arm = np.zeros(arm_idx.n)

    def update(self, action_full: np.ndarray):
        # action_full must already be ordered [legs..., arms...] — same as obs concat.
        n_leg = self.leg.n
        self.prev_full[:n_leg] = action_full[:n_leg]
        self.prev_full[n_leg:] = action_full[n_leg:]
        self.prev_leg = self.prev_full[:n_leg]
        self.prev_arm = self.prev_full[n_leg:]


class ObsBuilder:
    """Construct per-agent observations from the current MuJoCo data."""

    def __init__(self, leg_idx: JointIndex, arm_idx: JointIndex, ra_history: "RootHistoryBuffer"):
        self.leg = leg_idx
        self.arm = arm_idx
        # External rolling buffer owned by the main loop so other components
        # (e.g. reset hooks) can call reset() on it directly.
        self.ra_history = ra_history

    def _joint_pos_vel(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        # Concatenate legs first, then arms — matches total_leg_joint_ids + total_arm_joint_ids.
        jp = np.concatenate([data.qpos[self.leg.qpos_idx], data.qpos[self.arm.qpos_idx]])
        jv = np.concatenate([data.qvel[self.leg.qvel_idx], data.qvel[self.arm.qvel_idx]])
        return jp, jv

    def nominal(self, root: RootState, data: mujoco.MjData,
                phase_sin: float, phase_cos: float,
                cmd_b: np.ndarray, prev_arm: np.ndarray, prev_leg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Mirrors G1RecoveryEnv._get_observations multi-agent branch:
        #   arm (65): lin_b(3)+ang_b(3)+grav(3)+cmd(3)+phase_sin(1)+phase_cos(1)+jp_arm(17)+jv_arm(17)+prev_arm(17)
        #   leg (50): lin_b(3)+ang_b(3)+grav(3)+cmd(3)+phase_sin(1)+phase_cos(1)+jp_leg(12)+jv_leg(12)+prev_leg(12)
        jp_leg = data.qpos[self.leg.qpos_idx]
        jv_leg = data.qvel[self.leg.qvel_idx]
        jp_arm = data.qpos[self.arm.qpos_idx]
        jv_arm = data.qvel[self.arm.qvel_idx]
        ps = np.array([phase_sin])
        pc = np.array([phase_cos])

        arm_obs = np.concatenate([
            root.lin_b, root.ang_b, root.proj_grav,
            cmd_b,
            ps, pc,
            jp_arm, jv_arm,
            prev_arm,
        ])
        leg_obs = np.concatenate([
            root.lin_b, root.ang_b, root.proj_grav,
            cmd_b,
            ps, pc,
            jp_leg, jv_leg,
            prev_leg,
        ])
        return arm_obs, leg_obs

    def safe(self, root: RootState, data: mujoco.MjData,
             prev_arm: np.ndarray, prev_leg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Mirrors extras["safe_observations"] multi-agent branch:
        #   arm (60): lin_b(3)+ang_b(3)+grav(3)+jp_arm(17)+jv_arm(17)+prev_arm(17)
        #   leg (45): lin_b(3)+ang_b(3)+grav(3)+jp_leg(12)+jv_leg(12)+prev_leg(12)
        jp_leg = data.qpos[self.leg.qpos_idx]
        jv_leg = data.qvel[self.leg.qvel_idx]
        jp_arm = data.qpos[self.arm.qpos_idx]
        jv_arm = data.qvel[self.arm.qvel_idx]
        arm_obs = np.concatenate([
            root.lin_b, root.ang_b, root.proj_grav,
            jp_arm, jv_arm,
            prev_arm,
        ])
        leg_obs = np.concatenate([
            root.lin_b, root.ang_b, root.proj_grav,
            jp_leg, jv_leg,
            prev_leg,
        ])
        return arm_obs, leg_obs

    def reach_avoid(self, root: RootState, phase: float, dist_icp_stance: float) -> np.ndarray:
        # Mirrors extras["ra_states"]: ang_b(3) + proj_grav(3) + d_icp(1) + phase(1) + history.
        return np.concatenate([
            root.ang_b,
            root.proj_grav,
            np.array([dist_icp_stance]),
            np.array([phase]),
            self.ra_history.flatten(),
        ])

    # def safe_fall(self, root: RootState, data: mujoco.MjData) -> np.ndarray:
    #     # Mirrors extras["safe_fall_obs"] SafeFall baseline (63 dims).
    #     jp, jv = self._joint_pos_vel(data)
    #     return np.concatenate([
    #         root.proj_grav[:2],   # 2   gravity_xy
    #         root.ang_b,           # 3
    #         jp,                   # 29
    #         jv,                   # 29
    #     ])


# ---------------------------------------------------------------------- #
# Main                                                                   #
# ---------------------------------------------------------------------- #
def main():
    args = parse_args()

    # Torch device
    device = torch.device(args.device)
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

    # ---- Load three checkpoints (CooperativeMAPPO state-dicts + ReachAvoid critic) ----
    nominal_actor = load_cooperative_actor(args.checkpoint,      role="nominal", device=device, squash=SQUASH)
    instinct_actor    = load_cooperative_actor(args.instinct_checkpoint, role="safe",    device=device, squash=SQUASH)
    predictor     = load_ra_critic(args.predictor_checkpoint,    role="predictor", device=device)

    # ---- Random push helper ----
    pusher = RandomPush(
        model, body_name=args.push_body,
        period_sec=args.push_period,
        duration_sec=args.push_duration,
        force_max_n=args.push_force_max,
        rng=rng,
    )

    # ---- Observation builder + per-step state holders ----
    leg_idx = JointIndex(model, LEG_JOINT_NAMES)
    arm_idx = JointIndex(model, ARM_JOINT_NAMES)
    # Permutation from Isaac concat order (legs(12) | arms(17)) to MJCF actuator
    # declaration order. data.ctrl[ctrl_idx] scatters q_des into the correct
    # slots regardless of how the XML interleaves left/right.
    ctrl_idx = build_ctrl_index(model, LEG_JOINT_NAMES + ARM_JOINT_NAMES)
    root_state = RootState()
    phase_clock = PhaseClock(freq_hz=PHASE_FREQ_HZ)
    act_buf = ActionBuffer(leg_idx, arm_idx)
    # Rolling history of root states for the RA predictor; updated each substep.
    ra_history = RootHistoryBuffer(length=RA_HISTORY_LEN)
    obs_builder = ObsBuilder(leg_idx, arm_idx, ra_history=ra_history)
    # Velocity command kept in body frame; static across the episode for this eval script.
    cmd_b = np.array([CMD_VX, CMD_VY, CMD_WZ], dtype=np.float64)

    # Default joint targets captured from the home keyframe — used as the offset
    # for position actuators: ctrl = default_qpos + scaled_action. Stored in
    # Isaac concat order ([legs(12) | arms(17)]); mapped to MJCF actuator order
    # via `ctrl_idx` at write time (see step 7).
    default_q_isaac = np.concatenate([
        data.qpos[leg_idx.qpos_idx].copy(),
        data.qpos[arm_idx.qpos_idx].copy(),
    ])
    n_joints = leg_idx.n + arm_idx.n

    def _to_t(arr: np.ndarray) -> torch.Tensor:
        # Add the batch (env) dimension MuJoCo lacks and move to the policy's device.
        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).to(device)

    # Switching latch
    switch_latch = False

    # ---- Passive viewer ----
    viewer = mujoco.viewer.launch_passive(model, data)

    # Camera follows the robot's pelvis. mjCAMERA_TRACKING keeps `lookat` pinned
    # to the body each frame, so the robot stays centered even as it walks away
    # from the origin. distance/azimuth/elevation define the spherical offset.
    _track_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if _track_body >= 0:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = _track_body
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -20.0
    else:
        print("[WARN] 'pelvis' body not found; camera stays in free mode")

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
                # 1) Refresh root state and gait phase from the live MuJoCo data.
                root_state.update(data)
                phase, phase_sin, phase_cos = phase_clock.tick(float(data.time))

                # 1b) Push current ra-row into the rolling history BEFORE building the
                #     RA obs, matching training-time ordering
                #     (_compute_intermediate_values runs before _get_states).
                #     dist_icp_stance is stubbed to 0.0 here — same value used for the
                #     current-step slot below, so distribution stays self-consistent.
                ra_history.update(
                    ang_b=root_state.ang_b,
                    proj_grav=root_state.proj_grav,
                    d_icp=0.0,
                    phase=phase,
                )

                # 2) Build per-agent observations. Both nominal (arm 65, leg 50) and
                #    instinct/safe (arm 60, leg 45) are multi-agent dicts.
                nominal_arm_np, nominal_leg_np = obs_builder.nominal(
                    root_state, data, phase_sin, phase_cos,
                    cmd_b, act_buf.prev_arm, act_buf.prev_leg,
                )
                safe_arm_np, safe_leg_np = obs_builder.safe(
                    root_state, data, act_buf.prev_arm, act_buf.prev_leg
                )
                ra_np = obs_builder.reach_avoid(root_state, phase, dist_icp_stance=0.0)

                # 3) Run actor inference (SharedActor returns (a_arm, a_leg) in [-1, 1]).
                with torch.no_grad():
                    if nominal_actor is not None:
                        a_arm_n, a_leg_n = nominal_actor.act_deterministic(
                            _to_t(nominal_arm_np), _to_t(nominal_leg_np),
                        )
                        nominal_act = torch.cat([a_leg_n, a_arm_n], dim=-1)
                    else:
                        nominal_act = torch.zeros((1, n_joints), device=device)

                    if instinct_actor is not None:
                        a_arm_s, a_leg_s = instinct_actor.act_deterministic(
                            _to_t(safe_arm_np), _to_t(safe_leg_np),
                        )
                        instinct_act = torch.cat([a_leg_s, a_arm_s], dim=-1)
                    else:
                        instinct_act = torch.zeros((1, n_joints), device=device)

                    if predictor is not None:
                        risk = float(predictor.value(_to_t(ra_np)).item())
                    else:
                        risk = 0.0

                # 4) Latched switching: once risk crosses threshold we stay on safe.
                cur_switch = bool(risk > args.switch_threshold)
                switch = cur_switch or switch_latch
                if switch != switch_latch:
                    t = float(data.time)
                    print(f"t = {t} !!Instinct Agent!!")
                switch_latch = switch
                actions_t = instinct_act if switch else nominal_act

                # 5) Move to numpy, with a defensive shape check.
                actions_np = actions_t.detach().cpu().numpy().reshape(-1)
                if actions_np.shape[0] != n_joints:
                    actions_np = np.zeros(n_joints, dtype=np.float64)

                # 6) Record the non-scaled action so the next obs sees prev_actions in [-1, 1].
                act_buf.update(actions_np)

                # 7) Scaling
                scaled = np.empty_like(actions_np)
                scaled[:leg_idx.n] = actions_np[:leg_idx.n] * ACTION_SCALE_LEG
                scaled[leg_idx.n:] = actions_np[leg_idx.n:] * ACTION_SCALE_ARM
                q_des = default_q_isaac + scaled
                data.ctrl[ctrl_idx] = q_des

                # 8) Apply scheduled random push (no-op if outside the active window).
                pusher.apply(data)

                # 9) Step physics.
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
