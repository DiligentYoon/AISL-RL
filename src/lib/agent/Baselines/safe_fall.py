from typing import Dict, Union

import torch
import torch.nn as nn

from lib.agent.agent import Agent


class SafeFall(Agent):
    """SafeFall (GRU) fall predictor — inference-only model holder.

    Training is performed offline by
    `main/reach_avoid/baselines/safefall/train_offline.py`; this class only
    serves as a checkpoint container compatible with `Agent.load()` so that
    `play_unified.py` / `eval_predictor.py` can load the trained GRU and call
    `agent.critic(obs_seq, h)` directly.

    Model dict key is unified as "critic" with ReachAvoid so the base
    checkpoint dispatch is predictor-agnostic.
    """

    def __init__(
        self,
        model: Dict[str, nn.Module],
        device: Union[str, torch.device],
        cfg: Dict,
    ) -> None:
        super().__init__(cfg, model, device)

        self.critic = self.model.get("critic", None).to(self.device)
        self.checkpoint_modules["critic"] = self.critic

        self.set_running_mode("eval")
