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
import csv
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
SQUASH = False

# Inner-loop decimation: Isaac Lab runs the policy at sim_dt * decimation = 50 Hz
# (sim_dt = 1/200, decimation = 4). Replicate by re-inferring only every N substeps;
# data.ctrl stays latched between calls, mirroring env.step()'s _apply_action cadence.
POLICY_DECIMATION = 4

# ---- Push configuration (consumed by RandomPush via the main loop) ---- #
# Per-axis (min, max) wrench ranges sampled at every scheduled trigger.
# Missing keys default to (0, 0). Layout matches
# randomizer.apply_external_force_torque's contract.
PUSH_FORCE_RANGE: dict[str, tuple[float, float]] = {
    "x": (50.0, 100.0),
    "y": (350.0, 450.0),
    "z": (-1.0, 1.0),
}
PUSH_TORQUE_RANGE: dict[str, tuple[float, float]] = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
}

# ---- Termination thresholds (mirror G1RecoveryEnvCfg in lib) ---- #
# Training-time values from src/lib/env/G1/recovery/G1_recovery_env_cfg.py:58-60.
TERM_HEIGHT = 0.3        # CoM z below this -> died_fall
TERM_GRAVITY = 0.8       # |projected_gravity_xy| above this -> died_fall_2
TERM_ANG_VEL = 15.0      # ‖root_ang_vel_b‖ above this -> died_ang
COLLISION_FORCE_THRESHOLD = 1000.0   # >1N pure contact on any denied body -> died_collision

DENIED_COLLISION_BODIES: tuple[str, ...] = (
    "torso_link",
    # "left_wrist_yaw_joint",
    # "right_wrist_yaw_link",
)

# Cause codes — int for cheap numpy aggregation in TerminationChecker.cause.
CAUSE_ALIVE = 0
CAUSE_FALL = 1            # CoM low
CAUSE_FALL_TILT = 2       # gravity_xy large
CAUSE_ANG = 3             # angular velocity large
CAUSE_COLLISION = 4       # illegal contact
CAUSE_NAMES = {0: "alive", 1: "died_fall", 2: "died_fall_2",
               3: "died_ang", 4: "died_collision"}


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
    p.add_argument("--push_period", type=float, default=2.6,
                   help="Seconds between push events (sim time).")
    p.add_argument("--push_duration", type=float, default=0.2,
                   help="Seconds each push wrench is held (sim time).")
    # Viewer
    p.add_argument("--headless", action="store_true",
                   help="Skip the passive viewer; step as fast as possible.")
    p.add_argument("--total_sim_time", type=float, default=0.0,
                   help="Secondary stop: cumulative sim seconds across all episodes. 0 = unlimited.")
    # Episode lifecycle
    p.add_argument("--max_episode_sim_time", type=float, default=5.0,
                   help="Truncation: sim seconds in one episode before forced reset (counts as success).")
    p.add_argument("--total_episodes", type=int, default=20,
                   help="Primary stop: number of sequential episodes to run this invocation. "
                        "0 = unlimited.")
    # Misc
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the push RNG (reproducibility).")
    p.add_argument("--device", type=str, default="cuda",
                   help="Torch device override; defaults to cuda if available.")
    p.add_argument("--metrics_csv", type=str, default="",
                   help="If set, append per-episode metrics (success / max torque / "
                        "max contact) to this CSV. Header written once; reuse across runs.")
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
    Periodic wrench-push applied to a target body via xfrc_applied.

    At each (period_sec, duration_sec) trigger, samples force & torque
    uniformly from the per-axis (min, max) dicts passed to apply(); the
    latched wrench is held for duration_sec sim seconds, then zeroed until
    the next trigger. Dict layout mirrors
    randomizer.apply_external_force_torque (keys "x"/"y"/"z"; missing -> 0).
    """

    _AXES = ("x", "y", "z")

    def __init__(self, model: mujoco.MjModel, body_name: str,
                 period_sec: float, duration_sec: float,
                 rng: np.random.Generator):
        # Resolve body id once; negative means the body name was not in the scene.
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.period_sec = period_sec
        self.duration_sec = duration_sec
        self.rng = rng

        # Schedule state.
        self.next_at = period_sec        # sim time of next trigger
        self.active_until = -1.0         # sim time at which the current push ends
        self.force = np.zeros(3)         # latched body-frame force (N) for active window
        self.torque = np.zeros(3)        # latched body-frame torque (Nm) for active window
        self.is_pushing = False          # True while a wrench is actively applied this step

        if self.body_id < 0:
            print(f"[WARN] push body '{body_name}' not found; push disabled")

    @classmethod
    def _ranges_from_dict(cls, d: dict[str, tuple[float, float]] | None) -> np.ndarray:
        if d is None:
            return np.zeros((3, 2), dtype=np.float64)
        return np.asarray([d.get(k, (0.0, 0.0)) for k in cls._AXES], dtype=np.float64)

    def reset(self) -> None:
        """Clear schedule + latched wrench; called by reset_env(i) on episode end."""
        self.next_at = self.period_sec
        self.active_until = -1.0
        self.force[:] = 0.0
        self.torque[:] = 0.0
        self.is_pushing = False

    def apply(self, data: mujoco.MjData,
              force: dict[str, tuple[float, float]] | None = None,
              torque: dict[str, tuple[float, float]] | None = None) -> None:
        """Call once per mj_step, before stepping.

        `force`/`torque` are dicts of `{"x": (lo, hi), ...}` sampled
        uniformly at each scheduled trigger. Missing keys default to zero.
        """
        if self.body_id < 0:
            self.is_pushing = False
            return

        t = float(data.time)

        # Trigger a new push if the schedule has elapsed and the previous one ended.
        if t >= self.next_at and t >= self.active_until:
            f_ranges = self._ranges_from_dict(force)
            tq_ranges = self._ranges_from_dict(torque)
            # rng.uniform broadcasts (3,) lo/hi to a (3,) sample.
            self.force = self.rng.uniform(f_ranges[:, 0], f_ranges[:, 1])
            self.torque = self.rng.uniform(tq_ranges[:, 0], tq_ranges[:, 1])
            self.active_until = t + self.duration_sec
            self.next_at = t + self.period_sec
            print(
                f"[PUSH] t={t:7.3f}s  "
                f"F=({self.force[0]:+5.2f},{self.force[1]:+5.2f},{self.force[2]:+5.2f})N,  "
                f"T=({self.torque[0]:+5.2f},{self.torque[1]:+5.2f},{self.torque[2]:+5.2f})Nm"
            )

        # Write the latched 6-DoF wrench during the active window; zero outside it.
        # is_pushing flags whether a wrench is applied during the step about to run,
        # so the termination checker can suppress collision detection while it lasts.
        if t < self.active_until:
            data.xfrc_applied[self.body_id, :3] = self.force
            data.xfrc_applied[self.body_id, 3:] = self.torque
            self.is_pushing = True
        else:
            data.xfrc_applied[self.body_id, :] = 0.0
            self.is_pushing = False


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
# Termination checker (mirror of G1RecoveryEnv._get_dones)               #
# ---------------------------------------------------------------------- #
class TerminationChecker:
    """
    Single-env episode-end checker mirroring G1RecoveryEnv._get_dones (the four
    `died_*` conditions). Stateful — once the robot dies it stays dead within the
    current episode; reset() clears the latch so a new episode can be evaluated
    from scratch.

    Public state (read-only from outside):
      died:    bool  — True once any death condition fired this episode
      cause:   int   — CAUSE_* code; CAUSE_ALIVE while still alive
      died_t:  float — sim time at death (data.time, episode-relative); NaN while alive
    """

    def __init__(self, model: mujoco.MjModel, pelvis_name: str = "pelvis"):
        self.model = model
        # Pelvis = root of the floating-base subtree.
        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pelvis_name)
        if self.pelvis_id < 0:
            raise ValueError(f"pelvis body '{pelvis_name}' missing in model")

        # Denied-body-id list: every body NOT in DENIED_COLLISION_BODIES and not
        # the worldbody (id 0).
        denied_link_name = set(DENIED_COLLISION_BODIES)
        denied: list[int] = []
        for bid in range(1, model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if name and name in denied_link_name:
                denied.append(bid)
        self.denied_ids = np.asarray(denied, dtype=np.int64)
        print(f"[INFO] termination checker: {len(self.denied_ids)} denied bodies, ")

        # Latched per-episode state.
        self.died = False
        self.cause = CAUSE_ALIVE
        self.died_t = float("nan")

    def update(self, data: mujoco.MjData, root: "RootState",
               contact_per_body: np.ndarray, skip_collision: bool = False) -> None:
        """Re-evaluate the death conditions and latch if any fired.

        `contact_per_body` is the per-body contact-force magnitude from
        mj_contactForce (computed once by the caller and shared with the metric).
        `skip_collision=True` suppresses only the contact check; pass it while a
        push wrench is active. The other conditions stay live throughout.
        """
        if self.died:
            return  # already dead this episode — keep first cause/time
        # 1) died_fall — CoM z below height threshold.
        # com_z = float(data.subtree_com[self.pelvis_id, 2])
        # if com_z <= TERM_HEIGHT:
        #     self._latch(CAUSE_FALL, data, detail=f"CoM_z={com_z:.3f}<={TERM_HEIGHT}")
        #     return
        # # 2) died_fall_2 — projected gravity x or y exceeds tilt threshold.
        # gx, gy = float(root.proj_grav[0]), float(root.proj_grav[1])
        # if abs(gx) >= TERM_GRAVITY or abs(gy) >= TERM_GRAVITY:
        #     self._latch(CAUSE_FALL_TILT, data, detail=f"grav=({gx:+.3f},{gy:+.3f})")
        #     return
        # # 3) died_ang — body-frame angular velocity magnitude.
        # ang_norm = float(np.linalg.norm(root.ang_b))
        # if ang_norm >= TERM_ANG_VEL:
        #     self._latch(CAUSE_ANG, data, detail=f"|ang_b|={ang_norm:.2f}>={TERM_ANG_VEL}")
        #     return

        # 4) died_collision — contact force on any denied body, taken from
        #    mj_contactForce (contact_per_body). Suppressed while a push wrench is
        #    active (skip_collision) per the "collision only after push" rule.
        #    mj_contactForce excludes xfrc_applied, so the push is never counted
        #    as contact; cfrc_ext is not used because it reads 0 for contacts here.
        if not skip_collision and self.denied_ids.size > 0:
            denied_mag = contact_per_body[self.denied_ids]
            worst = int(np.argmax(denied_mag))
            if denied_mag[worst] > COLLISION_FORCE_THRESHOLD:
                bid = int(self.denied_ids[worst])
                bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or "?"
                self._latch(CAUSE_COLLISION, data, detail=f"{bname} |F|={denied_mag[worst]:.1f}N")
                return

    def _latch(self, cause: int, data: mujoco.MjData, detail: str) -> None:
        # First-death only: record cause/time and emit one console line.
        self.died = True
        self.cause = cause
        self.died_t = float(data.time)
        print(f"t={float(data.time):7.3f}s  FAILED cause={CAUSE_NAMES[cause]}  {detail}")

    def reset(self) -> None:
        """Clear death latch so a new episode can start."""
        self.died = False
        self.cause = CAUSE_ALIVE
        self.died_t = float("nan")


# ---------------------------------------------------------------------- #
# Metrics CSV writer                                                     #
# ---------------------------------------------------------------------- #
def _write_metrics_csv(path: str, records: list[tuple], seed: int) -> None:
    """Append per-episode metric rows; write the header only on a fresh/empty file."""
    new_file = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["seed", "episode", "success",
                        "max_torque_nm", "torque_joint",
                        "max_contact_n", "contact_link"])
        for idx, succ, tq, tj, fc, fl in records:
            w.writerow([seed, idx, int(succ), f"{tq:.4f}", tj, f"{fc:.4f}", fl])


# ---------------------------------------------------------------------- #
# Main                                                                   #
# ---------------------------------------------------------------------- #
def main():
    args = parse_args()

    # Torch device
    device = torch.device(args.device)
    print(f"[INFO] torch device: {device}")

    # Single push RNG (reproducible per invocation via --seed).
    rng = np.random.default_rng(args.seed)

    # ---- MuJoCo scene + keyframe init (single scene) ----
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    print(f"[INFO] scene={args.scene}  "
          f"nu={model.nu}  nq={model.nq}  nv={model.nv}  dt={dt:.4f}s")

    # ---- Load three checkpoints (CooperativeMAPPO state-dicts + ReachAvoid critic) ----
    nominal_actor  = load_cooperative_actor(args.checkpoint,          role="nominal",   device=device, squash=SQUASH)
    instinct_actor = load_cooperative_actor(args.instinct_checkpoint, role="safe",      device=device, squash=SQUASH)
    predictor      = load_ra_critic(args.predictor_checkpoint,        role="predictor", device=device)

    # ---- Model-derived indices ----
    leg_idx = JointIndex(model, LEG_JOINT_NAMES)
    arm_idx = JointIndex(model, ARM_JOINT_NAMES)
    ctrl_idx = build_ctrl_index(model, LEG_JOINT_NAMES + ARM_JOINT_NAMES)
    # Default joint targets from the home keyframe — ctrl = default_qpos + scaled_action.
    default_q_isaac = np.concatenate([
        data.qpos[leg_idx.qpos_idx].copy(),
        data.qpos[arm_idx.qpos_idx].copy(),
    ])
    n_joints = leg_idx.n + arm_idx.n
    phase_clock = PhaseClock(freq_hz=PHASE_FREQ_HZ)
    # Velocity command kept in body frame; static across the episode for this eval script.
    cmd_b = np.array([CMD_VX, CMD_VY, CMD_WZ], dtype=np.float64)

    # ---- Single-env helpers (mutated each substep) ----
    pusher = RandomPush(model, body_name=args.push_body,
                        period_sec=args.push_period,
                        duration_sec=args.push_duration,
                        rng=rng)
    root_state = RootState()
    ra_history = RootHistoryBuffer(length=RA_HISTORY_LEN)
    act_buf = ActionBuffer(leg_idx, arm_idx)
    obs_builder = ObsBuilder(leg_idx, arm_idx, ra_history=ra_history)
    switch_latch = False
    checker = TerminationChecker(model, pelvis_name="pelvis")

    # ---- Episode counters ----
    total_episodes = 0
    total_terminated = 0   # died (failure)
    total_truncated = 0    # survived to truncation (success)
    cause_counts = {c: 0 for c in CAUSE_NAMES}

    # Outer-loop time anchor independent of data.time (which resets on episode end).
    global_sim_t = 0.0

    # ---- Per-episode metric accumulators (peak values; reset each episode) ----
    ep_max_torque = 0.0          # peak |motor torque| over all joints this episode (Nm)
    ep_max_torque_joint = ""     # joint name where the torque peak occurred
    ep_max_contact = 0.0         # peak |contact force| over all links this episode (N)
    ep_max_contact_link = ""     # link/body name where the contact-force peak occurred
    episode_records: list[tuple] = []   # (idx, success, tau_max, joint, Fc_max, link)

    # Joint-ordered torque view + body-name table for attributing peaks.
    joint_names_ordered = LEG_JOINT_NAMES + ARM_JOINT_NAMES
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
                  for b in range(model.nbody)]

    def _to_t(arr: np.ndarray) -> torch.Tensor:
        # arr: (dim,) numpy -> (1, dim) float32 torch on device (add batch dim).
        return torch.from_numpy(arr.astype(np.float32)).to(device).unsqueeze(0)

    def reset_env() -> None:
        nonlocal act_buf, root_state, switch_latch
        nonlocal ep_max_torque, ep_max_torque_joint, ep_max_contact, ep_max_contact_link
        # Restore keyframe state (qpos/qvel; data.time -> 0 too); xfrc_applied zeroed by mj_resetData.
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        pusher.reset()
        ra_history.reset()
        # ActionBuffer / RootState are cheap — rebuild fresh.
        act_buf = ActionBuffer(leg_idx, arm_idx)
        root_state = RootState()
        switch_latch = False
        checker.reset()
        # Clear per-episode metric peaks.
        ep_max_torque = 0.0
        ep_max_torque_joint = ""
        ep_max_contact = 0.0
        ep_max_contact_link = ""

    # Decimation counter — policy fires when (decim_counter % POLICY_DECIMATION == 0).
    decim_counter = 0

    # ---- Passive viewer (skipped when --headless) ----
    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(model, data)
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
    else:
        print("[INFO] headless mode: viewer disabled")

    # ---- Real-time pacing anchors (viewer mode); ignored in headless ----
    tick_period = 1.0 / max(args.sim_rate_hz, 1.0)
    wall_anchor = time.monotonic()
    sim_anchor = global_sim_t
    max_substeps_per_tick = 20                       # cap to avoid spiral-of-death

    def keep_running() -> bool:
        if viewer is not None and not viewer.is_running():
            return False
        if args.total_sim_time > 0 and global_sim_t >= args.total_sim_time:
            return False
        if args.total_episodes > 0 and total_episodes >= args.total_episodes:
            return False
        return True

    stopped = False

    try:
        while keep_running():
            if viewer is not None:
                target_sim_t = sim_anchor + (time.monotonic() - wall_anchor)
            substeps = 0
            while substeps < max_substeps_per_tick and (
                viewer is None or global_sim_t < target_sim_t
            ):
                if decim_counter % POLICY_DECIMATION == 0:
                    # 1) Root state + gait phase + ra history.
                    phase, phase_sin, phase_cos = phase_clock.tick(global_sim_t)
                    root_state.update(data)
                    ra_history.update(
                        ang_b=root_state.ang_b,
                        proj_grav=root_state.proj_grav,
                        d_icp=0.0, phase=phase,
                    )

                    # 2) Build obs for each agent.
                    nominal_arm, nominal_leg = obs_builder.nominal(
                        root_state, data, phase_sin, phase_cos, cmd_b,
                        act_buf.prev_arm, act_buf.prev_leg)
                    safe_arm, safe_leg = obs_builder.safe(
                        root_state, data, act_buf.prev_arm, act_buf.prev_leg)
                    ra_obs = obs_builder.reach_avoid(root_state, phase, 0.0)

                    # 3) Forward pass per network.
                    with torch.no_grad():
                        if nominal_actor is not None:
                            a_arm_n, a_leg_n = nominal_actor.act_deterministic(
                                _to_t(nominal_arm), _to_t(nominal_leg))
                            nominal_act = torch.cat([a_leg_n, a_arm_n], dim=-1)
                        else:
                            nominal_act = torch.zeros((1, n_joints), device=device)
                        if instinct_actor is not None:
                            a_arm_s, a_leg_s = instinct_actor.act_deterministic(
                                _to_t(safe_arm), _to_t(safe_leg))
                            instinct_act = torch.cat([a_leg_s, a_arm_s], dim=-1)
                        else:
                            instinct_act = torch.zeros((1, n_joints), device=device)
                        if predictor is not None:
                            risk = float(predictor.value(_to_t(ra_obs)).reshape(-1)[0])
                        else:
                            risk = 0.0

                    # 4) Latched switching — once switched to instinct, stay until reset.
                    if risk > args.switch_threshold and not switch_latch:
                        print(f"t={float(data.time):7.3f}s !!Instinct Agent!!")
                        switch_latch = True
                    action_t = instinct_act if switch_latch else nominal_act

                    # 5) Action buffer + ctrl write.
                    action_np = action_t.detach().cpu().numpy()[0]
                    act_buf.update(action_np)
                    scaled = np.empty(n_joints, dtype=np.float64)
                    scaled[:leg_idx.n] = action_np[:leg_idx.n] * ACTION_SCALE_LEG
                    scaled[leg_idx.n:] = action_np[leg_idx.n:] * ACTION_SCALE_ARM
                    data.ctrl[ctrl_idx] = default_q_isaac + scaled

                # 6) Push + step + death/truncation check + reset.
                pusher.apply(data, force=PUSH_FORCE_RANGE, torque=PUSH_TORQUE_RANGE)
                mujoco.mj_step(model, data)
                root_state.update(data)

                # ---- Per-episode peak metrics (sampled every physics substep) ----
                # Motor torque: actuator output (one position actuator per joint),
                # viewed in joint order via ctrl_idx. Track the peak magnitude + joint.
                tau = np.abs(data.actuator_force[ctrl_idx])           # (29,)
                a_max = int(np.argmax(tau))
                if tau[a_max] > ep_max_torque:
                    ep_max_torque = float(tau[a_max])
                    ep_max_torque_joint = joint_names_ordered[a_max]
                # Contact force per link via the contact solver (cfrc_ext reads 0 here).
                # mj_contactForce reports only geom-geom contact forces, so the applied
                # push wrench (data.xfrc_applied) is excluded by construction — verified
                # empirically that an applied force never appears here. Sum each contact's
                # force magnitude onto both participating bodies. Computed once and shared
                # by the per-episode metric AND the collision termination check below.
                # Worldbody (ground, id 0) is excluded.
                contact_per_body = np.zeros(model.nbody, dtype=np.float64)
                f6 = np.zeros(6, dtype=np.float64)
                for ci in range(data.ncon):
                    con = data.contact[ci]
                    mujoco.mj_contactForce(model, data, ci, f6)
                    fmag = float(np.linalg.norm(f6[:3]))   # normal + 2 friction
                    contact_per_body[model.geom_bodyid[con.geom1]] += fmag
                    contact_per_body[model.geom_bodyid[con.geom2]] += fmag
                contact_per_body[0] = 0.0   # ignore worldbody / ground

                b_max = int(np.argmax(contact_per_body))
                if contact_per_body[b_max] > ep_max_contact:
                    ep_max_contact = float(contact_per_body[b_max])
                    ep_max_contact_link = body_names[b_max]

                checker.update(data, root_state, contact_per_body, skip_collision=pusher.is_pushing)

                if checker.died:
                    c = int(checker.cause)
                    total_terminated += 1
                    cause_counts[c] += 1
                    total_episodes += 1
                    episode_records.append((total_episodes, False, ep_max_torque,
                                            ep_max_torque_joint, ep_max_contact, ep_max_contact_link))
                    print(f"EP_END t_ep={float(data.time):6.3f}s  TERM   "
                          f"cause={CAUSE_NAMES[c]}  "
                          f"tau_max={ep_max_torque:7.2f}Nm@{ep_max_torque_joint}  "
                          f"Fc_max={ep_max_contact:7.1f}N@{ep_max_contact_link}  "
                          f"total={total_episodes}/{args.total_episodes}")
                    reset_env()
                elif data.time >= args.max_episode_sim_time:
                    total_truncated += 1
                    total_episodes += 1
                    episode_records.append((total_episodes, True, ep_max_torque,
                                            ep_max_torque_joint, ep_max_contact, ep_max_contact_link))
                    print(f"EP_END t_ep={float(data.time):6.3f}s  TRUNC(success)  "
                          f"tau_max={ep_max_torque:7.2f}Nm@{ep_max_torque_joint}  "
                          f"Fc_max={ep_max_contact:7.1f}N@{ep_max_contact_link}  "
                          f"total={total_episodes}/{args.total_episodes}")
                    reset_env()

                if args.total_episodes > 0 and total_episodes >= args.total_episodes:
                    stopped = True

                global_sim_t += dt
                substeps += 1
                decim_counter += 1
                if stopped:
                    break  # exit substep loop

            if stopped:
                break  # exit outer loop

            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, tick_period * 0.5))
    except KeyboardInterrupt:
        print("[INFO] interrupted by user")
    finally:
        if viewer is not None:
            viewer.close()
        # ---- Final summary ----
        M = total_episodes
        target_M = args.total_episodes
        denom = max(M, 1)
        print(f"[INFO] === Final summary ===")
        print(f"[INFO] total_episodes      = {M} (target {target_M})")
        print(f"[INFO] success (truncated) = {total_truncated}  "
              f"(success_rate {total_truncated / denom:.4f})")
        print(f"[INFO] failed  (terminated)= {total_terminated}  "
              f"(ratio {total_terminated / denom:.4f})")
        print(f"[INFO] failure cause breakdown:")
        for c, name in CAUSE_NAMES.items():
            if c == CAUSE_ALIVE:
                continue
            cnt = cause_counts.get(c, 0)
            print(f"[INFO]   {name:<18s} {cnt:>5d}  (ratio {cnt / denom:.4f})")
        print(f"[INFO] sim_time(global)= {global_sim_t:.2f}s")
        # ---- Per-episode metric table + aggregate ----
        if episode_records:
            taus = np.array([r[2] for r in episode_records])
            fcs  = np.array([r[4] for r in episode_records])
            print(f"[INFO] motor torque (Nm): peak={taus.max():.2f}  "
                  f"mean_of_ep_peaks={taus.mean():.2f}")
            print(f"[INFO] contact force (N):  peak={fcs.max():.1f}  "
                  f"mean_of_ep_peaks={fcs.mean():.1f}")
            print(f"[INFO] per-episode  idx  success  max_torque(joint)  max_contact(link)")
            for idx, succ, tq, tj, fc, fl in episode_records:
                print(f"[INFO]   {idx:>3d}   {'OK ' if succ else 'DIE'}   "
                      f"{tq:7.2f}Nm @ {tj:<22s}  {fc:7.1f}N @ {fl}")
        if args.metrics_csv:
            _write_metrics_csv(args.metrics_csv, episode_records, args.seed)
            print(f"[INFO] wrote {len(episode_records)} rows to {args.metrics_csv}")


if __name__ == "__main__":
    main()
