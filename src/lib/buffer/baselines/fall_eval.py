from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import torch


class FallPredictorEvaluator:
    """Episode-level evaluator for fall predictors.

    The evaluator is predictor-agnostic: it consumes a single per-environment
    danger score per step, where a *larger score means more dangerous*. It does
    not know (and must not know) which predictor produced the score, so
    Reach-Avoid and SafeFall are measured with the same ruler.

    An episode raises an alarm at threshold ``tau`` if ``score > tau`` holds for
    ``sustain_k`` consecutive steps anywhere in the episode. Only the existence
    of such an alarm matters, not when it happened. This mirrors the deployed
    switching rule in ``play_unified.py``, which latches the safe policy the
    first time the score crosses the threshold.

    Two metrics follow from that single boolean per episode:

        Detection Rate (DR) = TP / n_fall   (fall episodes that raised an alarm)
        False Alarm Rate    = FP / n_safe   (safe episodes that raised an alarm)

    A fall episode is one that ``terminated`` (failure contact); a safe episode
    is one that only ``truncated`` (timeout).
    """

    def __init__(
        self,
        num_envs: int,
        device: Union[str, torch.device],
        max_episode_steps: int,
        sustain_k: int = 1,
        max_fall: Optional[int] = None,
        max_safe: Optional[int] = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.max_episode_steps = int(max_episode_steps)
        self.sustain_k = max(1, int(sustain_k))

        # Per-category caps. Once a category is full, further episodes of that
        # category are dropped so the fall/safe sample sizes stay on target.
        self.max_fall = int(max_fall) if max_fall is not None else None
        self.max_safe = int(max_safe) if max_safe is not None else None

        # In-progress episodes: pre-allocated CPU score buffer + per-env cursor.
        # One spare row absorbs a step past max_episode_steps if the env ever
        # reports done one step late.
        self._score_buf = torch.zeros((self.max_episode_steps + 1, self.num_envs), dtype=torch.float32)
        self._cursor = torch.zeros(self.num_envs, dtype=torch.long)
        self._env_ids = torch.arange(self.num_envs, dtype=torch.long)

        # Finalized episodes: {"is_fall": bool, "T": int, "score": (T,) float32}
        self.records: List[Dict] = []
        self.count_fall = 0
        self.count_safe = 0

    # ------------------------------------------------------------------ #
    # Data collection
    # ------------------------------------------------------------------ #
    def add_step(
        self,
        scores: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
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
        T = int(self._cursor[env_id])
        if T == 0:
            return

        # Silently drop the episode once its category is full.
        if is_fall and self.max_fall is not None and self.count_fall >= self.max_fall:
            return
        if not is_fall and self.max_safe is not None and self.count_safe >= self.max_safe:
            return

        self.records.append({
            "is_fall": is_fall,
            "T": T,
            "score": self._score_buf[:T, env_id].clone(),
        })

        if is_fall:
            self.count_fall += 1
        else:
            self.count_safe += 1

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def _has_alarm(self, score: torch.Tensor, threshold: float) -> bool:
        """Whether ``score > threshold`` holds for ``sustain_k`` consecutive steps."""
        decision = score > threshold
        if self.sustain_k <= 1:
            return bool(decision.any())
        if decision.numel() < self.sustain_k:
            return False
        return bool(decision.unfold(0, self.sustain_k, 1).all(dim=-1).any())

    def compute_metrics(self, threshold: float) -> Dict:
        """Build the 2x2 contingency table at ``threshold`` and derive DR / FAR."""
        tp = fn = fp = tn = 0

        for rec in self.records:
            alarm = self._has_alarm(rec["score"], threshold)
            if rec["is_fall"]:
                tp += int(alarm)
                fn += int(not alarm)
            else:
                fp += int(alarm)
                tn += int(not alarm)

        n_fall = tp + fn
        n_safe = fp + tn

        return {
            "threshold": float(threshold),
            "sustain_k": self.sustain_k,
            "n_fall": n_fall,
            "n_safe": n_safe,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
            "detection_rate": _safe_div(tp, n_fall),
            "FAR": _safe_div(fp, n_safe),
        }

    def default_thresholds(self, num: int = 200, include: Sequence[float] = ()) -> List[float]:
        """Threshold grid drawn from the quantiles of every collected score.

        Quantiles (rather than a uniform grid) keep the sweep meaningful for both
        predictors, whose scores live on different scales — Reach-Avoid values are
        unbounded reals, SafeFall outputs are probabilities in [0, 1].
        """
        if not self.records:
            return sorted(set(float(t) for t in include))

        all_scores = torch.cat([rec["score"] for rec in self.records])
        qs = torch.linspace(0.0, 1.0, num)
        grid = torch.quantile(all_scores.double(), qs.double()).tolist()

        # A threshold below the minimum score alarms everywhere; one at/above the
        # maximum alarms nowhere. Both anchor the ROC curve at its endpoints.
        lo = float(all_scores.min()) - 1.0
        hi = float(all_scores.max())
        return sorted(set([lo, hi] + [float(g) for g in grid] + [float(t) for t in include]))

    def sweep(self, thresholds: Optional[Sequence[float]] = None) -> List[Dict]:
        """Compute metrics across a threshold grid (ascending by threshold)."""
        if thresholds is None:
            thresholds = self.default_thresholds()
        return [self.compute_metrics(t) for t in sorted(thresholds)]

    def roc(self, thresholds: Optional[Sequence[float]] = None) -> Dict:
        """FAR-vs-DR curve and its area, from the same episode records.

        Comparing the two predictors at their own fixed operating points is not
        apples-to-apples because their scores are on different scales; the curve
        is what makes the comparison fair.
        """
        rows = self.sweep(thresholds)

        # Anchor the curve at (0, 0) and (1, 1) so the area is well defined even
        # if the grid does not reach the extremes.
        points = {(0.0, 0.0), (1.0, 1.0)}
        for row in rows:
            far, dr = row["FAR"], row["detection_rate"]
            if far == far and dr == dr:  # skip NaN (empty category)
                points.add((far, dr))

        curve = sorted(points)
        far = [p[0] for p in curve]
        dr = [p[1] for p in curve]
        auc = sum((far[i + 1] - far[i]) * (dr[i + 1] + dr[i]) / 2.0 for i in range(len(curve) - 1))

        return {"FAR": far, "detection_rate": dr, "AUC": auc, "rows": rows}

    def detection_at_far(
        self,
        target_far: float,
        thresholds: Optional[Sequence[float]] = None,
    ) -> Dict:
        """Best detection rate reachable without exceeding ``target_far``.

        This is the fair head-to-head number: both predictors are compared at a
        matched false-alarm budget rather than at arbitrary thresholds.
        """
        feasible = [row for row in self.sweep(thresholds) if row["FAR"] <= target_far]
        if not feasible:
            return {"target_far": target_far, "threshold": float("nan"),
                    "detection_rate": float("nan"), "FAR": float("nan")}

        best = max(feasible, key=lambda row: row["detection_rate"])
        return {
            "target_far": target_far,
            "threshold": best["threshold"],
            "detection_rate": best["detection_rate"],
            "FAR": best["FAR"],
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_raw(self, path: str, threshold: float) -> None:
        """Save per-episode records so metrics can be recomputed offline."""
        meta = {
            "sustain_k": self.sustain_k,
            "threshold": threshold,
            "n_fall": self.count_fall,
            "n_safe": self.count_safe,
        }
        torch.save({"meta": meta, "records": self.records}, path)


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")
