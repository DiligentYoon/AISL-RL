from isaaclab.utils import configclass
from isaaclab.actuators.actuator_pd_cfg import IdealPDActuatorCfg
from .torque_delay_actuator import TorqueDelayedPDActuator


@configclass
class TorqueDelayedPDActuatorCfg(IdealPDActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = TorqueDelayedPDActuator

    decimation: int = 0
    """Downsampling ratio between Control loop and physics step decimation."""

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""