from dataclasses import MISSING
from isaaclab.utils import configclass
from .gear_actuator import GearDelayedPDActuator
from isaaclab.actuators.actuator_pd_cfg import DelayedPDActuatorCfg

@configclass
class GearDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for GearDelayedPDActuator."""

    class_type: type = GearDelayedPDActuator

    gear_ratio: float = MISSING
    
    gamma: float = MISSING