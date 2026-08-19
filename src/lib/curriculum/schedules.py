from __future__ import annotations


def linear(t: float, warmup: float = 0.0, endup: float = 1.0, **kwargs) -> float:
    """difficulty = t  (0 → 1 선형 증가)"""
    if t < warmup:
        # warmup phase (difficulty = 0)
        return 0.0
    else:
        # apply phase (difficulty = 0 -> 1)
        effective_t = (t - warmup) / (endup - warmup)
        effective_t = min(effective_t, 1.0)
    return t


def step(t: float, steps: list[float] = [0.33, 0.66], **kwargs) -> float:
    """Discrete difficulty jump.

    steps 리스트에 n개의 임계값을 지정하면 (n+1)단계 난이도로 나뉨.

    Examples:
        steps=[0.33, 0.66]
        → t < 0.33 : difficulty = 0.0
        → t < 0.66 : difficulty = 0.5
        → t >= 0.66: difficulty = 1.0
    """
    n = len(steps)
    for i, threshold in enumerate(steps):
        if t < threshold:
            return i / n
    return 1.0


SCHEDULE_REGISTRY: dict[str, callable] = {
    "linear": linear,
    "step": step,
}
