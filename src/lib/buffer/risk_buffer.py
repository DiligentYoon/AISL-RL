from __future__ import annotations

import os
from typing import Optional, Union

import torch


class RiskClassifiedBuffer:
    """Risk-classified replay buffer for initial-condition collection.

    Stores per-step physical-state snapshots into one of three capacity-bounded
    buckets (low / mid / high) by Reach-Avoid risk score. When a bucket reaches
    its capacity, further insertions to that bucket are dropped (not FIFO).

    Stored keys per bucket (all float32, shape ``(capacity, dim)``):
        root_pos_offset_w     (3)
        root_quat_w    (4)
        root_lin_vel_w (3)
        root_ang_vel_w (3)
        joint_pos      (joint_dim)
        joint_vel      (joint_dim)
        prev_action    (joint dim)
        risk_score     (1)
    """

    BUCKETS = ("low", "mid", "high")

    _ROOT_KEYS = {
        "root_pos_offset_w": 3,
        "root_quat_w": 4,
        "root_lin_vel_w": 3,
        "root_ang_vel_w": 3,
    }

    def __init__(
        self,
        capacity_per_bucket: int,
        thresholds: tuple[float, float],
        joint_dim: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.capacity = int(capacity_per_bucket)
        low_high, mid_high = thresholds
        if not (low_high < mid_high):
            raise ValueError(
                f"thresholds must satisfy low_high < mid_high, got ({low_high}, {mid_high})"
            )
        self.low_high = float(low_high)
        self.mid_high = float(mid_high)
        self.joint_dim = int(joint_dim)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        # Schema: key -> last-dim size
        self._key_dims: dict[str, int] = dict(self._ROOT_KEYS)
        self._key_dims["joint_pos"] = self.joint_dim
        self._key_dims["joint_vel"] = self.joint_dim
        self._key_dims["prev_action"] = self.joint_dim
        self._key_dims["risk_score"] = 1

        # Per-bucket flat storage tensors
        self.tensors: dict[str, dict[str, torch.Tensor]] = {b: {} for b in self.BUCKETS}
        for b in self.BUCKETS:
            for k, d in self._key_dims.items():
                self.tensors[b][k] = torch.zeros(
                    (self.capacity, d), dtype=torch.float32, device=self.device
                )

        # Per-bucket write head
        self.write_idx: dict[str, int] = {b: 0 for b in self.BUCKETS}

    # -------- classification --------

    def classify(self, risk_scores: torch.Tensor) -> torch.Tensor:
        """Map risk scores to bucket ids.

        Boundary convention:
            score <= low_high            -> 0 (low)
            low_high < score <= mid_high -> 1 (mid)
            score > mid_high             -> 2 (high)

        Args:
            risk_scores: shape (num_envs,) or (num_envs, 1)

        Returns:
            (num_envs,) long tensor in {0, 1, 2}
        """
        scores = risk_scores.flatten()
        return (scores > self.low_high).long() + (scores > self.mid_high).long()

    # -------- writes --------

    def add(
        self,
        snapshot: dict[str, torch.Tensor],
        risk_scores: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, int]:
        """Insert one multi-env step. Returns per-bucket inserted count.

        Args:
            snapshot: dict with keys
                root_pos_offset_w, root_quat_w, root_lin_vel_w, root_ang_vel_w, prev_action
                joint_pos, joint_vel
                — each tensor of shape (num_envs, dim).
            risk_scores: (num_envs,) or (num_envs, 1).
            valid_mask: optional (num_envs,) bool. Envs with False are skipped
                (used for warmup / terminated / subsample filtering).

        Returns:
            {bucket_name: num_inserted}
        """
        scores = risk_scores.flatten()
        num_envs = scores.shape[0]

        if valid_mask is None:
            mask_v = torch.ones(num_envs, dtype=torch.bool, device=scores.device)
        else:
            mask_v = valid_mask.flatten().bool()

        bucket_ids = self.classify(scores)
        added = {b: 0 for b in self.BUCKETS}

        for bid, bname in enumerate(self.BUCKETS):
            free = self.capacity - self.write_idx[bname]
            if free <= 0:
                continue

            mask = mask_v & (bucket_ids == bid)
            count = int(mask.sum().item())
            if count == 0:
                continue

            n = min(count, free)
            env_idx = torch.nonzero(mask, as_tuple=False).flatten()[:n]

            start = self.write_idx[bname]
            end = start + n
            store = self.tensors[bname]

            store["root_pos_offset_w"][start:end].copy_(snapshot["root_pos_offset_w"][env_idx])
            store["root_quat_w"][start:end].copy_(snapshot["root_quat_w"][env_idx])
            store["root_lin_vel_w"][start:end].copy_(snapshot["root_lin_vel_w"][env_idx])
            store["root_ang_vel_w"][start:end].copy_(snapshot["root_ang_vel_w"][env_idx])
            store["joint_pos"][start:end].copy_(snapshot["joint_pos"][env_idx])
            store["joint_vel"][start:end].copy_(snapshot["joint_vel"][env_idx])
            store["prev_action"][start:end].copy_(snapshot["prev_action"][env_idx])
            store["risk_score"][start:end, 0].copy_(scores[env_idx])

            self.write_idx[bname] = end
            added[bname] = n

        return added

    # -------- status --------

    def is_full(self) -> bool:
        """True if every bucket has reached capacity."""
        return all(self.write_idx[b] >= self.capacity for b in self.BUCKETS)

    def fill_status(self) -> dict[str, tuple[int, int]]:
        """Return ``{bucket: (current, capacity)}`` for each bucket."""
        return {b: (self.write_idx[b], self.capacity) for b in self.BUCKETS}

    def underfilled_buckets(self) -> list[str]:
        """Return bucket names that did not reach capacity."""
        return [b for b in self.BUCKETS if self.write_idx[b] < self.capacity]

    # -------- save --------

    def save(self, save_dir: str) -> None:
        """Persist each bucket as ``<save_dir>/<bucket>_risk.pt``.

        Tensors are truncated to the bucket's current write index and moved to
        CPU before saving. No metadata file is written.
        """
        os.makedirs(save_dir, exist_ok=True)
        for b in self.BUCKETS:
            n = self.write_idx[b]
            payload = {
                k: v[:n].detach().cpu().contiguous()
                for k, v in self.tensors[b].items()
            }
            torch.save(payload, os.path.join(save_dir, f"{b}_risk.pt"))
