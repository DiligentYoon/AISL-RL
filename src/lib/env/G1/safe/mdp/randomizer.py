from __future__ import annotations

import os
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from lib.env.env import Env


class reset_state_from_dataset(ManagerTermBase):
    """Reset both root state and joint state by sampling from a precomputed
    risk-classified initial-condition dataset (saved by collect.py).

    cfg.params:
        dataset_dir (str)
            Directory containing {low,mid,high}_risk.pt. Read on first
            __call__; train script may inject any time before the first
            env.reset().
        bucket_weights (dict[str, float])
            {"low": w_l, "mid": w_m, "high": w_h}. Re-normalized when the
            dict identity changes; buckets with weight=0 or empty file are
            dropped from the categorical sampler.
        asset_cfg (SceneEntityCfg)
            Default SceneEntityCfg("robot").
    """

    BUCKETS: tuple[str, ...] = ("low", "mid", "high")
    REQUIRED_KEYS: tuple[str, ...] = (
        "root_pos_offset_w", "root_quat_w",
        "root_lin_vel_w", "root_ang_vel_w",
        "joint_pos", "joint_vel",
        "prev_action",
    )

    def __init__(self, cfg: EventTermCfg, env: Env):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params.get(
            "asset_cfg", SceneEntityCfg("robot"))
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self._device = self.asset.device

        # Lazy state populated on first __call__
        self._loaded: bool = False
        self._buckets: dict[str, dict[str, torch.Tensor]] = {}
        self._bucket_caps: dict[str, int] = {}

        # Probability cache rebuilt when bucket_weights identity changes
        self._cached_weights_id: int | None = None
        self._valid_buckets: list[str] = []
        self._bucket_probs: torch.Tensor | None = None

        # Staging tensor for prev_action — shared with env via setattr below.
        # Filled on each __call__; env reads it in its _reset_idx.
        num_envs = env.scene.num_envs
        self.last_prev_action = torch.zeros(
            (num_envs, self.asset.num_joints),
            dtype=torch.float32, device=self._device,
        )
        setattr(env, "_dataset_reset_prev_action", self.last_prev_action)

    def _load_dataset(self, dataset_dir: str) -> None:
        for b in self.BUCKETS:
            path = os.path.join(dataset_dir, f"{b}_risk.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing dataset file: {path}")
            payload = torch.load(path, map_location=self._device)
            for k in self.REQUIRED_KEYS:
                if k not in payload:
                    raise KeyError(f"{path} missing key '{k}'")
            cap = payload["root_pos_offset_w"].shape[0]
            self._buckets[b] = {
                k: payload[k].to(self._device) for k in self.REQUIRED_KEYS
            }
            self._bucket_caps[b] = cap

        expected_J = self.asset.num_joints
        for b in self.BUCKETS:
            J = self._buckets[b]["joint_pos"].shape[-1]
            if J != expected_J:
                raise ValueError(
                    f"bucket '{b}' joint_dim={J} != robot.num_joints={expected_J}"
                )

    def _build_probs(self, bucket_weights: dict[str, float]) -> None:
        valid = [
            b for b in self.BUCKETS
            if self._bucket_caps[b] > 0 and bucket_weights.get(b, 0.0) > 0.0
        ]
        if not valid:
            raise RuntimeError(
                "All buckets are empty or have weight=0; nothing to sample."
            )
        w = torch.tensor(
            [bucket_weights[b] for b in valid],
            dtype=torch.float32, device=self._device,
        )
        self._valid_buckets = valid
        self._bucket_probs = w / w.sum()

    def __call__(
        self,
        env: Env,
        env_ids: torch.Tensor,
        dataset_dir: str,
        bucket_weights: dict[str, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        if not self._loaded:
            self._load_dataset(dataset_dir)
            self._loaded = True

        wid = id(bucket_weights)
        if wid != self._cached_weights_id:
            self._build_probs(bucket_weights)
            self._cached_weights_id = wid

        n = int(env_ids.numel())
        bucket_idx = torch.multinomial(
            self._bucket_probs, n, replacement=True
        )

        device = self._device
        J = self.asset.num_joints
        root_pos_offset = torch.empty((n, 3), device=device)
        root_quat       = torch.empty((n, 4), device=device)
        root_lin_vel    = torch.empty((n, 3), device=device)
        root_ang_vel    = torch.empty((n, 3), device=device)
        joint_pos       = torch.empty((n, J), device=device)
        joint_vel       = torch.empty((n, J), device=device)
        prev_action     = torch.empty((n, J), device=device)

        for k, bname in enumerate(self._valid_buckets):
            mask = (bucket_idx == k)
            m = int(mask.sum().item())
            if m == 0:
                continue
            cap = self._bucket_caps[bname]
            row = torch.randint(0, cap, (m,), device=device)
            store = self._buckets[bname]
            root_pos_offset[mask] = store["root_pos_offset_w"][row]
            root_quat[mask]       = store["root_quat_w"][row]
            root_lin_vel[mask]    = store["root_lin_vel_w"][row]
            root_ang_vel[mask]    = store["root_ang_vel_w"][row]
            joint_pos[mask]       = store["joint_pos"][row]
            joint_vel[mask]       = store["joint_vel"][row]
            prev_action[mask]     = store["prev_action"][row]

        root_pos = env.scene.env_origins[env_ids] + root_pos_offset

        self.asset.write_root_pose_to_sim(
            torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids)
        self.asset.write_root_velocity_to_sim(
            torch.cat([root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids)
        self.asset.write_joint_state_to_sim(
            joint_pos, joint_vel, env_ids=env_ids)

        # Stage prev_action for env._reset_idx to consume (env-side state, not sim).
        self.last_prev_action[env_ids] = prev_action