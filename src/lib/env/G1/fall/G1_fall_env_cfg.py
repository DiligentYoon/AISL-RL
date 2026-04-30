
from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from lib.env.G1.recovery.G1_recovery_env_cfg import G1RecoveryEnvCfg
from lib.utils.plot_utils import PNGSavePlotter
from isaaclab.managers import SceneEntityCfg

@configclass
class G1FallEnvCfg(G1RecoveryEnvCfg):

    # === RA agent config === #
    ra_state_space = 40
    body_hist_length = 4

    # === RA Setting === #
    l_max = 1.0
    target_set_threshold = 0.2

    # ==== Viz data ==== #
    # plotter = CapturabilityPlotter
    # viz_data = {
    #     "com_pos": 0,                  # (3,)
    #     "left_foot_pos": 0,            # (3,)
    #     "right_foot_pos": 0,           # (3,)
    #     "icp_pos": 0,                  # (2,)
    #     "capture_region_center": 0,    # (2,)
    #     "capture_region_radius": 0,    # scalar
    #     "time_hist": 0,                # scalar
    #     "m_step_hist": 0,              # scalar
    #     "icp_ankle_dist_hist": 0,      # scalar
    # }

    def __post_init__(self):
        super().__post_init__()

        self.events.push_robot.params["interval_range_s"] = (2.0, 3.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-2.0, 2.0),
            "y": (-2.0, 2.0),
            "roll": (-3.0, 3.0),
            "pitch": (-3.0, 3.0),
        }