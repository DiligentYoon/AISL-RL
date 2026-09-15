from __future__ import annotations

import gymnasium
import torch

from collections import deque
from typing import Optional, Union, Tuple, List, Deque

from lib.buffer.buffer import Buffer


class ReplayBuffer(Buffer):
    def __init__(
        self,
        buffer_size: int = 1,
        num_envs: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Replay buffer for Safety value learning.

        Args:
            buffer_size: Maximum number of elements in the first dimension
            num_envs: Number of parallel environments
            device: Torch device
        """
        super().__init__(buffer_size, num_envs, device)

    def init_buffer(self, state_space: gymnasium.Space) -> None:
        """Initialize replay buffer tensors for Safety Value learning.

        Args:
            state_space: State space of Safety critic
        """
        self.create_tensor("states", state_space, dtype=torch.float32)
        self.create_tensor("next_states", state_space, dtype=torch.float32)
        self.create_tensor("g_values", 1, dtype=torch.float32)
        self.create_tensor("terminated", 1, dtype=torch.bool)
        self.create_tensor("truncated", 1, dtype=torch.bool)


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