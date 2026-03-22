from __future__ import annotations

from typing import Any, Callable

from .curriculum_cfg import CurriculumManagerCfg, CurriculumParamCfg
from .schedules import SCHEDULE_REGISTRY


class CurriculumManager:
    """A manager for Curriculum Learning in RL.

    Workflow:
        common_step_counter / total_timesteps → t (0→1)
        → warmup → effective_t
        → schedule function → difficulty
        → start_value ~ end_value interpolation
        → apply randomization parameters
    """

    def __init__(self, cfg: CurriculumManagerCfg, env):
        self.cfg = cfg
        self.env = env
        self._difficulty: float = 0.0
        # (getter, setter, param_cfg) 튜플 목록
        self._resolvers: list[tuple[Callable, Callable, CurriculumParamCfg]] = []
        self._build_resolvers()

    # ── 초기화 ────────────────────────────────────────────────────────────────

    def _build_resolvers(self):
        """Spawn getter/setter of CurriculumParamCfg and initialize by applying difficulty=0."""
        for param_cfg in self.cfg.params:
            if param_cfg.schedule not in SCHEDULE_REGISTRY:
                raise ValueError(
                    f"CurriculumManager: unknown schedule '{param_cfg.schedule}' "
                    f"for param '{param_cfg.name}'. "
                    f"Available: {list(SCHEDULE_REGISTRY.keys())}"
                )
            getter, setter = self._make_accessor(self.env, param_cfg.attr_path)
            self._resolvers.append((getter, setter, param_cfg))
        self._apply(0.0)

    def _make_accessor(self, root: Any, path: str) -> tuple[Callable, Callable]:
        """dict key (isinstance dict) or getattr for setting getter/setter.

        Examples:
            "cfg/events/push_robot/params/velocity_range"
            "commands/cfg/ranges/lin_vel_x"
        """
        parts = path.split("/")

        def getter() -> Any:
            cur = root
            for p in parts:
                cur = cur[p] if isinstance(cur, dict) else getattr(cur, p)
            return cur

        def setter(value: Any) -> None:
            cur = root
            for p in parts[:-1]:
                cur = cur[p] if isinstance(cur, dict) else getattr(cur, p)
            last = parts[-1]
            if isinstance(cur, dict):
                cur[last] = value
            else:
                setattr(cur, last, value)

        return getter, setter

    # ── Update ──────────────────────────────────────────────────────────────

    def update(self, current_step: int):
        """Update Curriculum variables.

        Args:
            current_step: Current env step.
        """
        total = self.env.cfg.total_timesteps
        if total is None or total <= 0:
            return

        t = min(current_step / total, 1.0)

        # warmup phase (difficulty = 0)
        if t < self.cfg.warmup:
            effective_t = 0.0
        else:
            effective_t = (t - self.cfg.warmup) / (1.0 - self.cfg.warmup)
            effective_t = min(effective_t, 1.0)

        self._difficulty = effective_t
        self._apply(effective_t)

        # # extras["Curriculum"]에 기록 → TensorBoard 모니터링용
        # self.env.extras.setdefault("Curriculum", {})
        # self.env.extras["Curriculum"]["difficulty"] = self._difficulty

    def _apply(self, t: float):
        """apply curriculum by scheduling function"""
        for getter, setter, param_cfg in self._resolvers:
            schedule_fn = SCHEDULE_REGISTRY[param_cfg.schedule]
            difficulty = schedule_fn(t, **param_cfg.schedule_kwargs)
            new_value = self._interpolate(param_cfg.start_value, param_cfg.end_value, difficulty)
            setter(new_value)

    # ── 보간 ──────────────────────────────────────────────────────────────────

    def _interpolate(self, start: Any, end: Any, t: float) -> Any:
        """
        Types:
            float / int          → Linear Interpolation
            tuple / list         → Linear Interpolation for each element
            dict[str, Any]       → Recursive Linear Interpolation for each key 
            others                → If t >= 0.5 end, otherwise start
        """
        if isinstance(start, dict):
            return {k: self._interpolate(start[k], end[k], t) for k in start}
        elif isinstance(start, (tuple, list)):
            result = [s + (e - s) * t for s, e in zip(start, end)]
            return type(start)(result)
        elif isinstance(start, (int, float)):
            return start + (end - start) * t
        else:
            return end if t >= 0.5 else start

    # ── 조회 ──────────────────────────────────────────────────────────────────

    def get_difficulty(self) -> float:
        """Current difficulty value (0.0 → 1.0)."""
        return self._difficulty

    def get_param_value(self, name: str) -> Any:
        """
        Args:
            name: CurriculumParamCfg.name.

        Raises:
            KeyError: There is no param with the given name.
        """
        for getter, _, param_cfg in self._resolvers:
            if param_cfg.name == name:
                return getter()
        raise KeyError(f"CurriculumManager: param '{name}' not found.")
