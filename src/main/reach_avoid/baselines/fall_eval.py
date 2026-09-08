from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch


class FallPredictorEvaluator:
    """Two metrics follow from that single boolean per episode:

        Detection Rate (DR) = TP / n_fall   (fall episodes that raised an alarm)
        False Alarm Rate    = FP / n_safe   (safe episodes that raised an alarm)

    A fall episode is one that ``terminated`` (failure contact); a safe episode
    is one that only ``truncated`` (timeout).
    """

    def __init__(self,
                 dt: float,
                 num_envs: int,
                 device: Union[str, torch.device],
                 max_episode_steps: int,
                 max_fall: Optional[int] = None,
                 max_safe: Optional[int] = None) -> None:
        self.dt = dt
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.max_episode_steps = max_episode_steps

        self.max_fall = max_fall if max_fall is not None else None
        self.max_safe = max_safe if max_safe is not None else None

        self._score_buf = torch.zeros((self.max_episode_steps + 1, self.num_envs), dtype=torch.float32)
        self._cursor = torch.zeros(self.num_envs, dtype=torch.long)
        self._env_ids = torch.arange(self.num_envs, dtype=torch.long)

        self.records: List[Dict] = []
        self.count_fall = 0
        self.count_safe = 0

    # ------------------------------------------------------------------ #
    # Data collection
    # ------------------------------------------------------------------ #
    def add_step(self,
                 scores: torch.Tensor,
                 terminated: torch.Tensor,
                 truncated: torch.Tensor) -> None:
        """Record one multi-env step and finalize the episodes that ended.

        Args:
            scores: Per-env danger score, shape (N,) or (N, 1). Larger = more dangerous.
            terminated: Termination (fall) flags, shape (N,) or (N, 1).
            truncated: Truncation (timeout) flags, shape (N,) or (N, 1).
        """
        scores = scores.detach().reshape(-1).float().cpu()
        terminated = terminated.detach().reshape(-1).bool().cpu()
        truncated = truncated.detach().reshape(-1).bool().cpu()

        self._score_buf[self._cursor, self._env_ids] = scores
        self._cursor += 1
        self._cursor.clamp_(max=self.max_episode_steps)

        done = torch.logical_or(terminated, truncated)
        for env_id in torch.nonzero(done, as_tuple=False).flatten().tolist():
            self._finalize_episode(env_id, is_fall=bool(terminated[env_id]))
            self._cursor[env_id] = 0

    def _finalize_episode(self, env_id: int, is_fall: bool) -> None:
        """Turn the env's in-progress score buffer into an episode record."""
        T = self._cursor[env_id]
        if T == 0:
            return
        if is_fall and self.max_fall is not None and self.count_fall >= self.max_fall:
            return
        if not is_fall and self.max_safe is not None and self.count_safe >= self.max_safe:
            return

        self.records.append({"is_fall": is_fall, 
                             "T": T, 
                             "score": self._score_buf[:T, env_id].clone()})

        if is_fall:
            self.count_fall += 1
        else:
            self.count_safe += 1

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def _has_alarm(self, score: torch.Tensor, threshold: float) -> bool:
        decision = score > threshold
        return bool(decision.any())

    def _get_lead_time(self, record: dict[torch.Tensor], threshold: float) -> float:
        T_end = record["T"]
        T_first = int(torch.argwhere(record["score"] > threshold)[0]) # First index
        return (T_end - T_first) * self.dt

    def compute_metrics(self, threshold: float) -> Dict:
        """Build the 2x2 contingency table at ``threshold`` and derive DR / FAR."""
        tp = fn = fp = tn = lt = 0

        for rec in self.records:
            alarm = self._has_alarm(rec["score"], threshold)
            if rec["is_fall"]:
                tp += int(alarm)
                fn += int(not alarm)
                lt += self._get_lead_time(rec, threshold)

            else:
                fp += int(alarm)
                tn += int(not alarm)

        n_fall = tp + fn
        n_safe = fp + tn

        return {"threshold": float(threshold),
                "n_fall": n_fall,
                "n_safe": n_safe,
                "TP":  tp,
                "FN":  fn,
                "FP":  fp,
                "TN":  tn,
                "LT":  lt / tp,
                "DR":  tp / n_fall if n_fall > 0 else float("nan"),
                "FAR": fp / n_safe if n_safe > 0 else float("nan"),
                }
