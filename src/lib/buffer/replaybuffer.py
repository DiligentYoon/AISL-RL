from __future__ import annotations

import gymnasium
import torch

from collections import deque
from typing import Optional, Union, Tuple, List, Deque

from lib.buffer.buffer import Buffer


class HindSightReplayBuffer(Buffer):
    def __init__(
        self,
        buffer_size: int = 1,
        num_envs: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        hindsight_window: int = 10,
        hindsight_min_value: float = -0.8,
        hindsight_max_value: float = 1.0,
    ) -> None:
        """Replay buffer for Reach-Avoid value learning.

        Features:
            1. Circular replay buffer
            2. Failure-based hindsight relabeling for g_values

        Args:
            buffer_size: Maximum number of elements in the first dimension
            num_envs: Number of parallel environments
            device: Torch device
            hindsight_window: Number of recent steps to relabel on failure
            hindsight_min_value: Start value of hindsight ramp
            hindsight_max_value: Final value of hindsight ramp
        """
        super().__init__(buffer_size, num_envs, device)

        self.hindsight_window = hindsight_window
        self.hindsight_min_value = hindsight_min_value
        self.hindsight_max_value = hindsight_max_value

        # Per-env recent row history
        # Stores only row indices in temporal order
        self.recent_rows_per_env: List[Deque[int]] = [deque(maxlen=hindsight_window) for _ in range(num_envs)]

    def init_buffer(self, state_space: gymnasium.Space) -> None:
        """Initialize replay buffer tensors for RA learning.

        Args:
            state_space: State space of RA critic
        """
        self.create_tensor("states", state_space, dtype=torch.float32)
        self.create_tensor("next_states", state_space, dtype=torch.float32)
        self.create_tensor("l_values", 1, dtype=torch.float32)
        self.create_tensor("g_values", 1, dtype=torch.float32)
        self.create_tensor("terminated", 1, dtype=torch.bool)
        self.create_tensor("truncated", 1, dtype=torch.bool)

    def reset(self) -> None:
        """Reset replay buffer and recent history queues."""
        super().reset()
        self.recent_rows_per_env = [deque(maxlen=self.hindsight_window) for _ in range(self.num_envs)]

    def add_samples(self,
                    states: torch.Tensor,
                    next_states: torch.Tensor,
                    l_values: torch.Tensor,
                    g_values: torch.Tensor,
                    truncated: torch.Tensor,
                    terminated: torch.Tensor,) -> None:
        """Store one multi-env transition step and optionally apply hindsight relabeling.

        Expected transition semantics:
            states      = s_t
            next_states = s_{t+1}
            l_values    = l(s_t)
            g_values    = g(s_t)

        Args:
            states: Current RA states, shape (num_envs, state_dim)
            next_states: Next RA states, shape (num_envs, state_dim)
            l_values: Current l(s), shape (num_envs,) or (num_envs, 1)
            g_values: Current g(s), shape (num_envs,) or (num_envs, 1)
            truncated: Truncation flags, shape (num_envs,) or (num_envs, 1)
            terminated: Termination flags, shape (num_envs,) or (num_envs, 1)
        """
        l_values = self._ensure_2d_column(l_values, dtype=torch.float32)
        g_values = self._ensure_2d_column(g_values, dtype=torch.float32)
        truncated = self._ensure_2d_column(truncated, dtype=torch.bool)
        terminated = self._ensure_2d_column(terminated, dtype=torch.bool)

        # Save the current write row before super().add_samples updates memory_index
        current_row = self.memory_index

        # Store transition in circular replay memory
        super().add_samples(
            states=states,
            next_states=next_states,
            l_values=l_values,
            g_values=g_values,
            truncated=truncated,
            terminated=terminated)

        # Track this row as recent history for each env
        for env_id in range(self.num_envs):
            self.recent_rows_per_env[env_id].append(current_row)

        # Apply hindsight labeling immediately after insertion
        if torch.any(truncated):
            self.apply_hindsight_labeling(truncated)

        # Clear per-env recent history when episode ends
        done = torch.logical_or(truncated, terminated).squeeze(-1)
        done_env_ids = torch.nonzero(done, as_tuple=False).flatten()

        for env_id in done_env_ids.tolist():
            self.recent_rows_per_env[env_id].clear()

    def apply_hindsight_labeling(self, failure: torch.Tensor) -> None:
        """Relabel recent g_values only for failure environments.

        For each failed environment, relabel recent g_values with a linear ramp:
            hindsight_min_value -> hindsight_max_value

        The order of relabeling follows temporal order stored in recent_rows_per_env.

        Args:
            failure: Bool tensor of shape (num_envs,) or (num_envs, 1)
        """
        failure = self._ensure_2d_column(failure, dtype=torch.bool).squeeze(-1)
        failure_env_ids = torch.nonzero(failure, as_tuple=False).flatten()

        for env_id in failure_env_ids.tolist():
            refs = self.recent_rows_per_env[env_id]
            if len(refs) == 0:
                continue

            # Important:
            # refs must preserve temporal order, even under circular wrap-around
            row_indices = torch.as_tensor(list(refs), device=self.device, dtype=torch.long)

            ramp = torch.linspace(self.hindsight_min_value,
                                  self.hindsight_max_value,
                                  steps=len(refs),
                                  device=self.device,
                                  dtype=torch.float32,)

            old_g = self.tensors["g_values"][row_indices, env_id, 0]
            self.tensors["g_values"][row_indices, env_id, 0] = torch.maximum(old_g, ramp)

    def sample(self, names: Tuple[str], mini_batch: int) -> List[List[torch.Tensor]]:
        """Random sampling for off-policy replay learning.

        Args:
            names: Tensor names to sample
            mini_batch: Number of mini-batches

        Returns:
            List of mini-batches. Each mini-batch is a list of tensors ordered by names.
        """
        size = len(self)
        if size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        indexes = torch.randperm(size, dtype=torch.long, device=self.device)

        self.sampling_indexes = indexes
        return self.sample_by_index(names=names, indexes=indexes, mini_batches=mini_batch)

    def sample_batch(self, names: Tuple[str], batch_size: int) -> List[torch.Tensor]:
        """Sample one random batch.

        Args:
            names: Tensor names to sample
            batch_size: Number of samples in one batch

        Returns:
            List of sampled tensors ordered by names
        """
        size = len(self)
        if size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        batch_size = min(batch_size, size)
        indexes = torch.randint(
            low=0,
            high=size,
            size=(batch_size,),
            dtype=torch.long,
            device=self.device,
        )

        self.sampling_indexes = indexes
        return self.sample_by_index(names=names, indexes=indexes, mini_batches=1)[0]

    def _ensure_2d_column(self, tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Convert input tensor to shape (num_envs, 1) on target device/dtype."""
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)

        return tensor