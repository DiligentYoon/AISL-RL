from __future__ import annotations

import gymnasium
import torch

from collections import deque
from typing import Optional, Union, Tuple, List, Deque, Dict

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

            if bool(term_flat[env_id]):
                # Failure episode: 0..t1 safe, t1..t2 ambiguous, t2..T_e fall
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
                # Truncated-only (normal timeout): entire episode is safe
                self.tensors["label"][rows, env_id, 0] = 0
                self.tensors["mask"][rows, env_id, 0] = True

            self.closed_episodes.append({
                "rows": list(queue),
                "env_id": env_id,
                "length": T_e,
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

    def _ensure_2d_column(self, tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        return tensor.to(dtype)
