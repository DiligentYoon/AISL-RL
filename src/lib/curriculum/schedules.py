from __future__ import annotations


def linear(t: float, **kwargs) -> float:
    """difficulty = t  (0 → 1 선형 증가)"""
    return t


def step(t: float, steps: list[float] | None = None, **kwargs) -> float:
    """difficulty를 구간별로 점프.

    steps 리스트에 n개의 임계값을 지정하면 (n+1)단계 난이도로 나뉨.

    예시:
        steps=[0.33, 0.66]
        → t < 0.33 : difficulty = 0.0
        → t < 0.66 : difficulty = 0.5
        → t >= 0.66: difficulty = 1.0
    """
    if steps is None:
        steps = [0.33, 0.66]
    n = len(steps)
    for i, threshold in enumerate(steps):
        if t < threshold:
            return i / n
    return 1.0


SCHEDULE_REGISTRY: dict[str, callable] = {
    "linear": linear,
    "step": step,
}
