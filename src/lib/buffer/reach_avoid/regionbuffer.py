from __future__ import annotations

import os
from typing import Optional, Union

import torch


class RegionBuffer:
    """Buffer for fall-predictor region analysis.

    One buffer stores one synchronized parallel sweep.

    - One environment corresponds to one disturbance condition.
    - Time-varying data are written at every collection step.
    - Disturbance metadata are written when the disturbance is applied.
    - The terminal sample is appended and the done metadata finalized once,
      when each environment terminates or truncates.
    """

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        ra_state_dim: int = 10,
        disturbance_dim: int = 4,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:

        self.num_envs = int(num_envs)
        self.num_steps = int(num_steps)
        self.ra_state_dim = int(ra_state_dim)
        self.disturbance_dim = int(disturbance_dim)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        # ============================================================
        # Time-varying trajectory data
        # ============================================================

        self.tensors: dict[str, torch.Tensor] = {
            "reach_avoid_state": torch.zeros((self.num_envs, self.num_steps, self.ra_state_dim), dtype=torch.float32, device=self.device),
            "reach_avoid_value": torch.full((self.num_envs, self.num_steps, 1), torch.nan, dtype=torch.float32, device=self.device),
        }

        # ============================================================
        # Per-environment metadata
        # ============================================================

        self.metadata: dict[str, torch.Tensor] = {
            # Applied disturbance:
            "disturbance": torch.zeros((self.num_envs, self.disturbance_dim), dtype=torch.float32, device=self.device),

            # Whether the environment ended through each condition.
            "terminated": torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device),
            "truncated": torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device),

            # -1 means that the event did not occur.
            "disturbance_apply_idx": torch.full((self.num_envs, 1), -1, dtype=torch.long, device=self.device),
            "falling_idx": torch.full((self.num_envs, 1), -1, dtype=torch.long, device=self.device),
            "done_idx": torch.full((self.num_envs, 1), -1, dtype=torch.long, device=self.device),
        }

        # Number of valid trajectory samples stored per environment.
        self.write_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # False after an environment terminates, truncates, or fills capacity.
        self.active_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    # ================================================================
    # Disturbance metadata
    # ================================================================

    @torch.no_grad()
    def set_disturbance(
        self,
        disturbance: torch.Tensor,
        apply_idx: Union[int, torch.Tensor],
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Record the disturbance applied to each environment.

        Args:
            disturbance:
                Shape [N, disturbance_dim], where N is len(env_ids), or
                num_envs when env_ids is None.
            apply_idx:
                Buffer index corresponding to the post-disturbance sample.
                Either a scalar shared by all environments, or a [N] tensor when
                environments are at different write positions.
            env_ids:
                Environments receiving the disturbance. Defaults to all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        self.metadata["disturbance"][env_ids] = disturbance.to(device=self.device, dtype=torch.float32)
        self.metadata["disturbance_apply_idx"][env_ids, 0] = apply_idx

    # ================================================================
    # Per-step writes
    # ================================================================

    def _writable_env_ids(self) -> torch.Tensor:
        """Active environments that still have capacity left."""
        insert_mask = self.active_mask & (self.write_idx < self.num_steps)
        return torch.nonzero(insert_mask, as_tuple=False).squeeze(-1)

    def _write(self, env_ids: torch.Tensor, snapshot: dict[str, torch.Tensor]) -> torch.Tensor:
        """Write one sample per environment and advance its write index.

        Returns the indices the samples were written to.
        """
        # Each environment can have its own write index.
        step_ids = self.write_idx[env_ids]

        # Correct per-environment advanced indexing.
        self.tensors["reach_avoid_state"][env_ids, step_ids] = snapshot["reach_avoid_state"][env_ids]
        self.tensors["reach_avoid_value"][env_ids, step_ids] = snapshot["reach_avoid_value"][env_ids]

        # The current sample is now valid.
        self.write_idx[env_ids] += 1

        # Stop collection when capacity has been reached.
        self.active_mask[self.write_idx >= self.num_steps] = False

        return step_ids

    def add(self, snapshot: dict[str, torch.Tensor]) -> None:
        """Insert one synchronized multi-environment sample."""
        env_ids = self._writable_env_ids()
        if env_ids.numel() == 0:
            return

        self._write(env_ids, snapshot)

    def add_terminal(
        self,
        snapshot: dict[str, torch.Tensor],
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Insert the terminal sample for environments that just ended, then finalize.

        ``snapshot`` carries the pre-reset terminal state for every environment; only
        the entries of environments that actually ended are consumed. Call this after
        :meth:`add` so the terminal sample lands after the last regular one.
        """
        # The environment wrapper reports these as [E, 1]; this buffer indexes flat.
        terminated = terminated.flatten().bool()
        truncated = truncated.flatten().bool()

        env_ids = self._writable_env_ids()
        env_ids = env_ids[terminated[env_ids] | truncated[env_ids]]
        if env_ids.numel() == 0:
            return

        step_ids = self._write(env_ids, snapshot)

        self.finalize_buffer(env_ids=env_ids,
                             step_ids=step_ids,
                             terminated=terminated[env_ids],
                             truncated=truncated[env_ids])

    # ================================================================
    # Done finalization
    # ================================================================

    def finalize_buffer(
        self,
        env_ids: torch.Tensor,
        step_ids: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Finalize metadata for environments that have just ended."""

        self.metadata["terminated"][env_ids, 0] = terminated
        self.metadata["truncated"][env_ids, 0] = truncated
        self.metadata["done_idx"][env_ids, 0] = step_ids

        # In the current environment, terminated represents an actual fall.
        fall_env_ids = env_ids[terminated]
        fall_step_ids = step_ids[terminated]

        self.metadata["falling_idx"][fall_env_ids, 0] = fall_step_ids

        # Latch these environments as inactive.
        self.active_mask[env_ids] = False

    # ================================================================
    # Status
    # ================================================================

    @property
    def is_complete(self) -> bool:
        return not bool(self.active_mask.any().item())

    def get_valid_time_mask(self) -> torch.Tensor:
        """Return a [num_envs, num_steps] valid-data mask."""
        time_idx = torch.arange(
            self.num_steps,
            device=self.device,
        ).unsqueeze(0)

        return time_idx < self.write_idx.unsqueeze(1)

    # ================================================================
    # Reset
    # ================================================================

    def reset(self) -> None:
        self.tensors["reach_avoid_state"].zero_()
        self.tensors["reach_avoid_value"].fill_(torch.nan)

        self.metadata["disturbance"].zero_()
        self.metadata["terminated"].zero_()
        self.metadata["truncated"].zero_()
        self.metadata["disturbance_apply_idx"].fill_(-1)
        self.metadata["falling_idx"].fill_(-1)
        self.metadata["done_idx"].fill_(-1)

        self.write_idx.zero_()
        self.active_mask.fill_(True)

    # ================================================================
    # Save
    # ================================================================

    def save(
        self,
        save_dir: str,
        filename: str = "region_buffer.pt",
    ) -> str:
        """Save trajectory tensors and metadata in one payload."""
        os.makedirs(save_dir, exist_ok=True)

        max_steps = int(self.write_idx.max().item())
        valid_time_mask = self.get_valid_time_mask()[:, :max_steps]

        payload = {
            "trajectory": {
                key: value[:, :max_steps].detach().cpu().contiguous()
                for key, value in self.tensors.items()
            },
            "metadata": {
                key: value.detach().cpu().contiguous()
                for key, value in self.metadata.items()
            },
            "write_idx": self.write_idx.detach().cpu().contiguous(),
            "valid_time_mask": valid_time_mask.detach().cpu().contiguous(),
            "config": {
                "num_envs": self.num_envs,
                "num_steps": self.num_steps,
                "ra_state_dim": self.ra_state_dim,
                "disturbance_dim": self.disturbance_dim,
            },
        }

        save_path = os.path.join(save_dir, filename)
        torch.save(payload, save_path)

        return save_path