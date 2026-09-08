from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CurriculumParamCfg:
    """Curriculum variable class controlled by CurriculumManager.

    where is the params(attr_path), 
    easy level (start_value),
    hard value(end_value), 
    curriculum scheduling (schedule).

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
    """CurriculumManager Settings"""
    
    params: list[CurriculumParamCfg] = field(default_factory=list)
    """list of controlled parameters."""
