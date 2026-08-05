from __future__ import annotations

import json
import os

import gymnasium
import torch

from collections import deque
from typing import Any, Optional, Union, Tuple, List, Deque, Dict

from lib.buffer.buffer import Buffer


class RecurrentReplayBuffer(Buffer):
    """Sequence replay buffer for RNN fall predictor (SafeFall).

    Storage layout follows the Buffer base convention:
        tensors[name] has shape (buffer_size, num_envs, data_size)

    Per-env in-progress episode is tracked by a row-index deque. When an
    episode terminates, label and mask are written retroactively into the
    circular tensor (the same pattern used by HindSightReplayBuffer for
    g_value relabeling). Closed-episode metadata is appended to
    self.closed_episodes for sequence sampling.

    Labeling rule (per Step 2 of plan):
        Let T_e = episode length
            t1 = floor(2*T_e/3)
            t2 = max(t1, T_e - W_f)   with W_f = round(fall_lead_seconds/step_dt)

        terminated env:
            [0   : t1 ]  label=0, mask=1   (safe)
            [t1  : t2 ]  label=*, mask=0   (ambiguous, excluded from loss)
            [t2  : T_e]  label=1, mask=1   (pre-fall lead window)

        truncated-only env (timeout, no failure):
            [0   : T_e]  label=0, mask=1

    NOTE: row indices in self.closed_episodes point into a circular tensor.
    Wrap-around can overwrite those rows. Recommended to size buffer_size
    such that wrap-around does not occur within a single training session
    (>= max_episode_steps * num_envs). See plan §4.
    """

    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        device: Optional[Union[str, torch.device]],
        seq_len: int,
        fall_lead_seconds: float,
        step_dt: float,
        max_episode_steps: int,
        max_closed_episodes: Optional[int] = None,
    ) -> None:
        super().__init__(buffer_size, num_envs, device)

        self.seq_len = int(seq_len)
        self.fall_lead_seconds = float(fall_lead_seconds)
        self.step_dt = float(step_dt)
        self.max_episode_steps = int(max_episode_steps)

        # Pre-fall lead window in environment steps
        self.W_f = max(1, int(round(fall_lead_seconds / step_dt)))

        # Per-env in-progress row history
        self.recent_rows_per_env: List[Deque[int]] = [
            deque(maxlen=self.max_episode_steps) for _ in range(num_envs)
        ]

        # Closed-episode metadata: list of {"rows": [...], "env_id": int, "length": T_e}
        self.closed_episodes: List[Dict] = []
        self.max_closed_episodes = (
            max_closed_episodes
            if max_closed_episodes is not None
            else max(1, buffer_size * num_envs // max(1, self.max_episode_steps))
        )

    def init_buffer(self, obs_dim: int) -> None:
        """Initialize storage tensors for SafeFall sequence learning.

        Args:
            obs_dim: Dimension of safe_fall observation vector.
        """
        self.create_tensor("obs", obs_dim, dtype=torch.float32)
        self.create_tensor("label", 1, dtype=torch.long)
        self.create_tensor("mask", 1, dtype=torch.bool)

        # Override Buffer.create_tensor's NaN fill for non-float tensors
        # and zero-initialize obs (Buffer fills floats with NaN, which would
        # contaminate the lead-window samples before any label is written).
        self.tensors["obs"].zero_()
        self.tensors["label"].zero_()
        self.tensors["mask"].zero_()

    def reset(self) -> None:
        super().reset()
        self.recent_rows_per_env = [
            deque(maxlen=self.max_episode_steps) for _ in range(self.num_envs)
        ]
        self.closed_episodes = []

    def add_samples(
        self,
        obs: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Store one multi-env transition step and label closed episodes.

        Args:
            obs: SafeFall observation, shape (num_envs, obs_dim)
            terminated: Termination flags, shape (num_envs,) or (num_envs, 1)
            truncated: Truncation flags, shape (num_envs,) or (num_envs, 1)
        """
        terminated = self._ensure_2d_column(terminated, dtype=torch.bool)
        truncated = self._ensure_2d_column(truncated, dtype=torch.bool)

        current_row = self.memory_index

        # Push obs only; label/mask are filled at episode close time.
        super().add_samples(obs=obs)

        for env_id in range(self.num_envs):
            self.recent_rows_per_env[env_id].append(current_row)

        # Label closed episodes
        done = torch.logical_or(terminated, truncated).squeeze(-1)
        done_env_ids = torch.nonzero(done, as_tuple=False).flatten().tolist()
        if len(done_env_ids) == 0:
            return

        term_flat = terminated.squeeze(-1)
        for env_id in done_env_ids:
            queue = self.recent_rows_per_env[env_id]
            T_e = len(queue)
            if T_e == 0:
                continue

            rows = torch.as_tensor(list(queue), device=self.device, dtype=torch.long)
            term = bool(term_flat[env_id])

            self._label_episode(rows, env_id, term, T_e)

            self.closed_episodes.append({
                "rows": list(queue),
                "env_id": env_id,
                "length": T_e,
                "terminated": term,
            })
            queue.clear()

        # FIFO cap on closed_episodes
        if len(self.closed_episodes) > self.max_closed_episodes:
            drop = len(self.closed_episodes) - self.max_closed_episodes
            del self.closed_episodes[:drop]

    def sample_sequences(
        self,
        batch_size: int,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of fixed-length sequences from closed episodes.

        Args:
            batch_size: Number of sequences to sample.
            seq_len: Sequence length. Defaults to self.seq_len.

        Returns:
            obs_seq: (B, seq_len, obs_dim)
            label_seq: (B, seq_len) long
            mask_seq: (B, seq_len) bool
        """
        if seq_len is None:
            seq_len = self.seq_len

        n_closed = len(self.closed_episodes)
        if n_closed == 0:
            raise ValueError("Cannot sample sequences: no closed episodes in buffer.")

        # Random sampling with replacement if fewer closed episodes than batch_size
        idxs = torch.randint(low=0, high=n_closed, size=(batch_size,), dtype=torch.long).tolist()

        obs_dim = self.tensors["obs"].shape[-1]
        obs_seq = torch.zeros((batch_size, seq_len, obs_dim), device=self.device, dtype=torch.float32)
        label_seq = torch.zeros((batch_size, seq_len), device=self.device, dtype=torch.long)
        mask_seq = torch.zeros((batch_size, seq_len), device=self.device, dtype=torch.bool)

        for b, idx in enumerate(idxs):
            ep = self.closed_episodes[idx]
            T_e = ep["length"]
            env_id = ep["env_id"]

            if T_e >= seq_len:
                offset = int(torch.randint(low=0, high=T_e - seq_len + 1, size=(1,)).item())
                row_slice = ep["rows"][offset:offset + seq_len]
                rows_t = torch.as_tensor(row_slice, device=self.device, dtype=torch.long)
                obs_seq[b] = self.tensors["obs"][rows_t, env_id]
                label_seq[b] = self.tensors["label"][rows_t, env_id, 0]
                mask_seq[b] = self.tensors["mask"][rows_t, env_id, 0]
            else:
                # Left zero-pad shorter episodes
                pad = seq_len - T_e
                rows_t = torch.as_tensor(ep["rows"], device=self.device, dtype=torch.long)
                obs_seq[b, pad:] = self.tensors["obs"][rows_t, env_id]
                label_seq[b, pad:] = self.tensors["label"][rows_t, env_id, 0]
                mask_seq[b, pad:] = self.tensors["mask"][rows_t, env_id, 0]
                # mask_seq[b, :pad] stays False (already zero-initialized)

        return obs_seq, label_seq, mask_seq

    def _label_episode(
        self,
        rows: torch.Tensor,
        env_id: int,
        terminated: bool,
        T_e: int,
    ) -> None:
        """Apply the 3-segment label/mask in-place for a single closed episode.

        Shared by online add_samples and offline load_from_disk paths.
        """
        if terminated:
            t1 = (2 * T_e) // 3
            t2 = max(t1, T_e - self.W_f)

            if t1 > 0:
                safe_rows = rows[:t1]
                self.tensors["label"][safe_rows, env_id, 0] = 0
                self.tensors["mask"][safe_rows, env_id, 0] = True

            if t2 > t1:
                ambig_rows = rows[t1:t2]
                self.tensors["mask"][ambig_rows, env_id, 0] = False

            if T_e > t2:
                fall_rows = rows[t2:]
                self.tensors["label"][fall_rows, env_id, 0] = 1
                self.tensors["mask"][fall_rows, env_id, 0] = True
        else:
            self.tensors["label"][rows, env_id, 0] = 0
            self.tensors["mask"][rows, env_id, 0] = True

    def flush_to_disk(
        self,
        output_dir: str,
        shard_size: int = 1024,
        train_val_split: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Dump closed_episodes' raw obs sequences to .pt shards.

        Labels are NOT persisted; load_from_disk will regenerate them via
        self.fall_lead_seconds / self.step_dt using the same _label_episode rule.

        Args:
            output_dir: Destination directory (created if missing).
            shard_size: Trajectories per shard file.
            train_val_split: Optional (train, val) trajectory counts to record
                in the returned meta dict.

        Returns:
            meta dict with shard_files, trajectories_per_shard, statistics.
        """
        shards_dir = os.path.join(output_dir, "shards")
        os.makedirs(shards_dir, exist_ok=True)

        obs_tensor = self.tensors["obs"]
        obs_dim = obs_tensor.shape[-1]

        shard_files: List[str] = []
        traj_per_shard: List[int] = []
        n_terminated = 0
        total_length = 0

        cur_shard: List[Dict[str, Any]] = []
        shard_idx = 0

        def _flush(payload: List[Dict[str, Any]], idx: int) -> None:
            path = os.path.join(shards_dir, f"traj_{idx:06d}.pt")
            torch.save(payload, path)
            shard_files.append(os.path.relpath(path, output_dir).replace("\\", "/"))
            traj_per_shard.append(len(payload))

        for ep in self.closed_episodes:
            rows_t = torch.as_tensor(ep["rows"], device=obs_tensor.device, dtype=torch.long)
            env_id = ep["env_id"]
            T_e = ep["length"]
            terminated = bool(ep.get("terminated", False))
            obs_seq = obs_tensor[rows_t, env_id].detach().cpu().clone()
            cur_shard.append({
                "obs": obs_seq,
                "terminated": terminated,
                "length": T_e,
            })
            n_terminated += int(terminated)
            total_length += T_e
            if len(cur_shard) >= shard_size:
                _flush(cur_shard, shard_idx)
                shard_idx += 1
                cur_shard = []

        if cur_shard:
            _flush(cur_shard, shard_idx)

        n_total = len(self.closed_episodes)
        meta: Dict[str, Any] = {
            "num_trajectories": n_total,
            "terminated_ratio": n_terminated / max(n_total, 1),
            "mean_length": total_length / max(n_total, 1),
            "obs_dim": obs_dim,
            "shard_files": shard_files,
            "trajectories_per_shard": traj_per_shard,
            "shard_size": shard_size,
            "fall_lead_seconds": self.fall_lead_seconds,
            "step_dt": self.step_dt,
            "max_episode_steps": self.max_episode_steps,
        }
        if train_val_split is not None:
            meta["train_val_split"] = list(train_val_split)
        return meta

    def load_from_disk(self, dataset_dir: str, split: str = "train") -> None:
        """Populate tensors[obs/label/mask] and closed_episodes from .pt shards.

        Reallocates internal storage to a packed (total_steps, 1, *) layout and
        regenerates labels using the current self.fall_lead_seconds via the same
        3-segment rule as add_samples.

        Args:
            dataset_dir: Directory containing meta.json and shards/.
            split: "train" or "val".
        """
        meta_path = os.path.join(dataset_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if "train_val_split" not in meta:
            raise KeyError("meta.json missing 'train_val_split'.")
        train_size, val_size = meta["train_val_split"]
        if split == "train":
            start_traj, end_traj = 0, int(train_size)
        elif split == "val":
            start_traj, end_traj = int(train_size), int(train_size) + int(val_size)
        else:
            raise ValueError(f"Unknown split: {split}")

        # Walk shards in record order, collect trajectories within [start_traj, end_traj).
        shard_rel_files: List[str] = list(meta.get("shard_files", []))
        if not shard_rel_files:
            raise ValueError(f"meta.json has no shard_files; dataset_dir={dataset_dir}")

        selected: List[Dict[str, Any]] = []
        traj_idx = 0
        for rel in shard_rel_files:
            if traj_idx >= end_traj:
                break
            shard_path = os.path.join(dataset_dir, rel)
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            for entry in shard:
                if start_traj <= traj_idx < end_traj:
                    selected.append(entry)
                traj_idx += 1
                if traj_idx >= end_traj:
                    break

        if not selected:
            raise ValueError(
                f"No trajectories selected for split='{split}' "
                f"(start={start_traj}, end={end_traj}, found_total={traj_idx})."
            )

        total_steps = int(sum(int(e["length"]) for e in selected))
        obs_dim = int(meta.get("obs_dim", self.tensors["obs"].shape[-1] if "obs" in self.tensors else 0))
        if obs_dim <= 0:
            raise ValueError("Cannot determine obs_dim from meta.json or current tensors.")

        # Reallocate to packed (total_steps, 1, *) layout.
        self.tensors.clear()
        self.tensors_view.clear()
        self.tensors_keep_dimensions.clear()
        for attr in [a for a in list(self.__dict__) if a.startswith("_tensor_")]:
            delattr(self, attr)
        self.buffer_size = total_steps
        self.num_envs = 1
        super().reset()
        self.recent_rows_per_env = [deque(maxlen=self.max_episode_steps)]
        self.closed_episodes = []

        self.create_tensor("obs", obs_dim, dtype=torch.float32)
        self.create_tensor("label", 1, dtype=torch.long)
        self.create_tensor("mask", 1, dtype=torch.bool)
        self.tensors["obs"].zero_()
        self.tensors["label"].zero_()
        self.tensors["mask"].zero_()

        # Write and label each trajectory.
        row_cursor = 0
        for entry in selected:
            T_e = int(entry["length"])
            obs_seq = entry["obs"]
            if obs_seq.shape[0] != T_e:
                T_e = int(obs_seq.shape[0])
            obs_seq = obs_seq.to(device=self.device, dtype=self.tensors["obs"].dtype)

            row_list = list(range(row_cursor, row_cursor + T_e))
            rows_t = torch.as_tensor(row_list, device=self.device, dtype=torch.long)
            self.tensors["obs"][rows_t, 0, :] = obs_seq

            terminated = bool(entry.get("terminated", False))
            self._label_episode(rows_t, env_id=0, terminated=terminated, T_e=T_e)

            self.closed_episodes.append({
                "rows": row_list,
                "env_id": 0,
                "length": T_e,
                "terminated": terminated,
            })
            row_cursor += T_e

        self.memory_index = row_cursor
        self.filled = row_cursor >= self.buffer_size
        self.meta = meta

    def _ensure_2d_column(self, tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        return tensor.to(dtype)
