from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CurriculumParamCfg:
    """커리큘럼으로 제어할 파라미터 하나의 Spec card.

    CurriculumManager는 env 안의 여러 파라미터를 일괄 제어한다.
    각 파라미터가 어디에 있는지(attr_path), 쉬울 때 값(start_value),
    어려울 때 값(end_value), 어떤 속도로 키울지(schedule)를 하나의 객체로 묶는다.

        Examples:
            "cfg/events/push_robot/params/velocity_range"
                env.cfg  →  .events  →  .push_robot  →  .params["velocity_range"]

            "commands/cfg/ranges/lin_vel_x"
                env.commands  →  .cfg  →  .ranges  →  .lin_vel_x

    schedule:
        "linear" : difficulty = t  (0 → 1 linear)
        "step"   : discrete jump at the threshold values  (schedule_kwargs에 steps 지정)
    """

    name: str
    attr_path: str
    start_value: Any
    end_value: Any
    schedule: str = "linear"
    schedule_kwargs: dict = field(default_factory=dict)


@dataclass
class CurriculumManagerCfg:
    """CurriculumManager 전체 설정.

    total_timesteps는 env.cfg.total_timesteps에서 읽으므로 여기에 정의하지 않는다.
    train.py에서 gym.make() 호출 전에 env_cfg.total_timesteps를 주입한다.
    """

    warmup: float = 0.1
    """학습 초반 몇 % 동안 최저 난이도(difficulty=0)를 유지할지. 기본값 0.1 (= 10%)."""

    params: list[CurriculumParamCfg] = field(default_factory=list)
    """제어할 파라미터 목록."""
