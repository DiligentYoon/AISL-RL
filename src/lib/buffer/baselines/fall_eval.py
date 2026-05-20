from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch


class FallPredictorEvaluator:
    """Evaluation collector for fall predictors (Reach-Avoid / SafeFall).

    Records per-environment rollouts, finalizes them at episode boundaries,
    and computes False Alarm Rate (FAR) and Lead Time (LT) metrics.

    Design (see plan.md):
        - FAR is measured only on safe (truncated-only) episodes. An early
          alarm inside a fall episode is an "early warning", not a false
          alarm, so it is excluded from FAR.
        - Detection / LT are run-based. The predictor decision sequence is
          split into alarm runs (noise filtered by ``sustain_k``, merged
          across gaps up to ``gap_tol``). A fall is detected only if an
          alarm run extends into the danger window ``[t_fall - W, t_fall]``.
        - ``W`` is the predictor's training-time labelling window:
          RA -> number of positive entries in the hindsight ramp;
          SafeFall -> ``round(fall_lead_seconds / dt)``.
        - SafeFall scores are computed with a sliding window of length
          ``seq_len`` (zero hidden state, left zero-padding) to match the
          training input distribution.
    """

    def __init__(
        self,
        predictor_type: str,
        num_envs: int,
        dt: float,
        device: Union[str, torch.device],
        sustain_k: int = 1,
        gap_tol: int = 5,
        hindsight_window: int = 10,
        hindsight_min_value: float = -0.8,
        hindsight_max_value: float = 1.0,
        fall_lead_seconds: float = 0.1,
        seq_len: int = 16,
    ) -> None:
        if predictor_type not in ("ra", "safefall"):
            raise ValueError(f"Unknown predictor_type: {predictor_type}")

        self.predictor_type = predictor_type
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.device = torch.device(device)

        self.sustain_k = int(sustain_k)
        self.gap_tol = int(gap_tol)

        # RA GT-window parameters (hindsight ramp)
        self.hindsight_window = int(hindsight_window)
        self.hindsight_min_value = float(hindsight_min_value)
        self.hindsight_max_value = float(hindsight_max_value)

        # SafeFall GT-window / inference parameters
        self.fall_lead_seconds = float(fall_lead_seconds)
        self.seq_len = int(seq_len)

        # SafeFall GRU model reference (set via set_predictor_model)
        self.gru = None

        # Per-env in-progress episode buffers
        #   ra       -> list of scalar score tensors
        #   safefall -> list of (obs_dim,) observation tensors
        self._buf: List[List[torch.Tensor]] = [[] for _ in range(self.num_envs)]

        # Finalized episode records
        self.records: List[Dict] = []
        self.count_fall = 0
        self.count_safe = 0

    def set_predictor_model(self, model: torch.nn.Module) -> None:
        """Register the SafeFall GRU model used for sliding-window scoring."""
        self.gru = model

    # ------------------------------------------------------------------ #
    # Data collection
    # ------------------------------------------------------------------ #
    def add_step(
        self,
        step_data: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Store one multi-env step and finalize episodes that ended.

        Args:
            step_data: RA -> per-env score, shape (N,) or (N, 1).
                       SafeFall -> per-env observation, shape (N, obs_dim).
            terminated: Termination flags, shape (N,) or (N, 1).
            truncated: Truncation flags, shape (N,) or (N, 1).
        """
        terminated = terminated.reshape(-1).bool()
        truncated = truncated.reshape(-1).bool()

        if self.predictor_type == "ra":
            scores = step_data.reshape(-1).detach()
            for env_id in range(self.num_envs):
                self._buf[env_id].append(scores[env_id])
        else:
            obs = step_data.detach()
            for env_id in range(self.num_envs):
                self._buf[env_id].append(obs[env_id])

        done = torch.logical_or(terminated, truncated)
        done_env_ids = torch.nonzero(done, as_tuple=False).flatten().tolist()
        for env_id in done_env_ids:
            self._finalize_episode(env_id, is_fall=bool(terminated[env_id]))
            self._buf[env_id] = []

    def _finalize_episode(self, env_id: int, is_fall: bool) -> None:
        """Build an episode record from the env's in-progress buffer."""
        items = self._buf[env_id]
        T_e = len(items)
        if T_e == 0:
            return

        if self.predictor_type == "ra":
            score = torch.stack(items).reshape(-1).float().cpu()
        else:
            obs_seq = torch.stack(items, dim=0)  # (T_e, obs_dim)
            score = self._safefall_scores(obs_seq).float().cpu()

        record = {
            "predictor": self.predictor_type,
            "is_fall": is_fall,
            "T_e": T_e,
            "t_fall": T_e - 1,
            "W_pred": self._compute_window(T_e) if is_fall else 0,
            "score": score,
        }
        self.records.append(record)

        if is_fall:
            self.count_fall += 1
        else:
            self.count_safe += 1

    @torch.no_grad()
    def _safefall_scores(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """Sliding-window SafeFall scoring (matches training input format).

        For step t, the GRU is fed obs[t-L+1 : t+1] left zero-padded to
        length L=seq_len with a zero hidden state; the fall probability is
        read from the last position.

        Args:
            obs_seq: (T_e, obs_dim) observation sequence of one episode.

        Returns:
            (T_e,) per-step fall probability.
        """
        if self.gru is None:
            raise RuntimeError("SafeFall GRU model is not set. Call set_predictor_model().")

        T_e, obs_dim = obs_seq.shape
        L = self.seq_len
        windows = torch.zeros((T_e, L, obs_dim), device=obs_seq.device, dtype=obs_seq.dtype)
        for t in range(T_e):
            chunk = obs_seq[max(0, t - L + 1): t + 1]   # (<=L, obs_dim)
            windows[t, L - chunk.shape[0]:] = chunk     # left zero-pad

        logits, _, _ = self.gru(windows)                # (T_e, L, 2)
        prob = torch.softmax(logits[:, -1, :], dim=-1)[:, 1]
        return prob

    def _compute_window(self, T_e: int) -> int:
        """Per-predictor training-time danger window length (in steps)."""
        if self.predictor_type == "ra":
            n = min(self.hindsight_window, T_e)
            if n < 1:
                return 1
            ramp = torch.linspace(self.hindsight_min_value, self.hindsight_max_value, n)
            return max(1, int((ramp > 0).sum().item()))
        else:
            return max(1, int(round(self.fall_lead_seconds / self.dt)))

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def _extract_runs(self, decision: torch.Tensor) -> List[Tuple[int, int]]:
        """Split a decision sequence into alarm runs.

        Raw contiguous True segments shorter than ``sustain_k`` are dropped
        (noise filter); remaining segments separated by a gap of at most
        ``gap_tol`` steps are merged into a single run.

        Returns:
            List of (start, end) inclusive index pairs.
        """
        idx = torch.nonzero(decision, as_tuple=False).flatten().tolist()
        if not idx:
            return []

        # raw contiguous True runs
        raw: List[Tuple[int, int]] = []
        start = prev = idx[0]
        for i in idx[1:]:
            if i == prev + 1:
                prev = i
            else:
                raw.append((start, prev))
                start = prev = i
        raw.append((start, prev))

        # noise filter by sustain_k
        raw = [(a, b) for (a, b) in raw if b - a + 1 >= self.sustain_k]
        if not raw:
            return []

        # merge runs separated by a gap <= gap_tol
        merged = [list(raw[0])]
        for a, b in raw[1:]:
            if a - merged[-1][1] - 1 <= self.gap_tol:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        return [(a, b) for a, b in merged]

    def compute_metrics(self, threshold: float) -> Dict:
        """Compute FAR / detection / Lead Time at the given decision threshold.

        Args:
            threshold: Decision threshold. A step is an alarm if score > threshold.

        Returns:
            Dict with FAR_step, FAR_episode, detection_rate, LT_mean/max/min,
            LT_first_mean, n_fall, n_safe.
        """
        far_fp = 0
        far_total = 0
        n_safe = 0
        n_safe_alarm = 0

        n_fall = 0
        detected = 0
        lt_list: List[float] = []
        lt_first_list: List[float] = []

        for rec in self.records:
            decision = rec["score"] > threshold
            runs = self._extract_runs(decision)

            if not rec["is_fall"]:
                n_safe += 1
                far_total += rec["T_e"]
                far_fp += int(decision.sum().item())
                if runs:
                    n_safe_alarm += 1
                continue

            # fall episode -> detection / lead time
            n_fall += 1
            t_fall = rec["t_fall"]
            window_start = t_fall - rec["W_pred"]

            # runs whose alarm persists into the danger window
            leading = [r for r in runs if r[1] >= window_start]
            if leading:
                detected += 1
                best = max(leading, key=lambda r: r[1])   # closest to the fall
                lt_list.append((t_fall - best[0]) * self.dt)
                lt_first_list.append((t_fall - runs[0][0]) * self.dt)

        def _safe_div(a: float, b: float) -> float:
            return a / b if b > 0 else float("nan")

        def _stat(values: List[float], fn) -> float:
            return fn(values) if values else float("nan")

        return {
            "FAR_step": _safe_div(far_fp, far_total),
            "FAR_episode": _safe_div(n_safe_alarm, n_safe),
            "detection_rate": _safe_div(detected, n_fall),
            "LT_mean": _stat(lt_list, lambda v: sum(v) / len(v)),
            "LT_max": _stat(lt_list, max),
            "LT_min": _stat(lt_list, min),
            "LT_first_mean": _stat(lt_first_list, lambda v: sum(v) / len(v)),
            "n_fall": n_fall,
            "n_safe": n_safe,
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_raw(self, path: str, threshold: float) -> None:
        """Save raw per-episode records and metadata for offline re-analysis."""
        meta = {
            "predictor_type": self.predictor_type,
            "dt": self.dt,
            "sustain_k": self.sustain_k,
            "gap_tol": self.gap_tol,
            "threshold": threshold,
            "hindsight_window": self.hindsight_window,
            "hindsight_min_value": self.hindsight_min_value,
            "hindsight_max_value": self.hindsight_max_value,
            "fall_lead_seconds": self.fall_lead_seconds,
            "seq_len": self.seq_len,
        }
        torch.save({"meta": meta, "records": self.records}, path)
