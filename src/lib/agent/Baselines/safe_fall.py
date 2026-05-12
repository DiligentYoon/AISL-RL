from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.agent.agent import Agent
from lib.buffer.recurrent_replay import RecurrentReplayBuffer


class SafeFall(Agent):
    """SafeFall (GRU) fall predictor baseline.

    External interface matches ReachAvoid for polymorphic use in train.py:
        insert_data(infos, next_infos, terminated, truncated)
        update() -> float
        save(path), load(path)  (inherited from Agent)

    Model dict key is unified as "critic" with RA so that base Agent.load()
    dispatch and checkpoint_modules indexing are predictor-agnostic. The
    underlying network is a GRU (see lib.model.Baselines.SafeFall.safe_fall).
    """

    def __init__(
        self,
        model: Dict[str, nn.Module],
        buffer: RecurrentReplayBuffer,
        device: Union[str, torch.device],
        cfg: Dict,
    ) -> None:
        super().__init__(cfg, model, device)

        self.critic = self.model.get("critic", None).to(self.device)
        self.buffer = buffer
        self.checkpoint_modules["critic"] = self.critic

        self.learning_epochs = self.cfg["learning_epochs"]
        self.batch_size = self.cfg["batch_size"]
        self.seq_len = self.cfg["seq_len"]
        self.learning_rate = self.cfg["learning_rate"]
        self.weight_decay = self.cfg.get("weight_decay", 0.0)
        self.grad_norm_clip = self.cfg["grad_norm_clip"]
        self.learning_starts = self.cfg["learning_starts"]

        self.optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.checkpoint_modules["optimizer"] = self.optimizer

        self.set_running_mode("eval")

    def insert_data(
        self,
        infos: Dict[str, torch.Tensor],
        next_infos: Dict[str, torch.Tensor],
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Store one multi-env observation step.

        next_infos is accepted for interface symmetry with ReachAvoid but
        unused here.
        """
        self.buffer.add_samples(
            obs=infos["safe_fall_obs"],
            terminated=terminated,
            truncated=truncated,
        )

    def update(self) -> float:
        """One update cycle: sample sequences, masked cross-entropy step."""
        if len(self.buffer.closed_episodes) == 0:
            return float("nan")

        self.set_running_mode("train")

        cumulative_loss = 0.0
        num_updates = 0

        for _ in range(self.learning_epochs):
            obs_seq, label_seq, mask_seq = self.buffer.sample_sequences(
                batch_size=self.batch_size,
                seq_len=self.seq_len,
            )

            logits, _ = self.critic(obs_seq)                  # (B, T, 2)

            flat_logits = logits.reshape(-1, 2)
            flat_label = label_seq.reshape(-1)
            flat_mask = mask_seq.reshape(-1).float()

            ce = F.cross_entropy(flat_logits, flat_label, reduction="none")
            loss = (ce * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_clip)
            self.optimizer.step()

            cumulative_loss += loss.item()
            num_updates += 1

        self.set_running_mode("eval")

        return cumulative_loss / max(num_updates, 1)
