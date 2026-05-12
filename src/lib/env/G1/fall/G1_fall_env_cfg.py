
from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from lib.env.G1.recovery.G1_recovery_env_cfg import G1RecoveryEnvCfg
from lib.domain_randomizer import randomizer
from lib.utils.plot_utils import CapturabilityPlotter
from lib.curriculum.curriculum_cfg import CurriculumManagerCfg, CurriculumParamCfg

@configclass
class G1FallEnvCfg(G1RecoveryEnvCfg):

    # === RA agent config === #
    ra_state_space = 40
    body_hist_length = 4

    # === SafeFall baseline config === #
    safe_fall_obs_dim = 63

    # === RA Setting === #
    l_max = 1.0
    target_set_threshold = 0.1

    # === Curriculum === #
    push_x_end = (-2.0, 2.0)
    push_y_end = (-2.0, 2.0)
    push_roll_end = (-5.0, 5.0)
    push_pitch_end = (-5.0, 5.0)

    def __post_init__(self):
        super().__post_init__()

        self.curriculum: CurriculumManagerCfg = CurriculumManagerCfg(
            warmup=0.2,
            endup=0.5,
            params=[
                CurriculumParamCfg(
                    name="push_range_x",
                    attr_path="cfg/events/push_robot/params/velocity_range/x",
                    start_value=self.events.push_robot.params["velocity_" \
                    "range"]["x"],
                    end_value=self.push_x_end
                ),
                CurriculumParamCfg(
                    name="push_range_y",
                    attr_path="cfg/events/push_robot/params/velocity_range/y",
                    start_value=self.events.push_robot.params["velocity_range"]["y"],
                    end_value=self.push_y_end
                ),
                CurriculumParamCfg(
                    name="push_range_roll",
                    attr_path="cfg/events/push_robot/params/velocity_range/roll",
                    start_value=self.events.push_robot.params["velocity_range"]["roll"],
                    end_value=self.push_roll_end
                ),
                CurriculumParamCfg(
                    name="push_range_pitch",
                    attr_path="cfg/events/push_robot/params/velocity_range/pitch",
                    start_value=self.events.push_robot.params["velocity_range"]["pitch"],
                    end_value=self.push_pitch_end
                ),
            ]
        )

        self.events.push_robot.interval_range_s = (2.0, 3.0)
        # self.events.push_robot.params["velocity_range"] = {
        #     "x": (-2.0, 2.0),
        #     "y": (-2.0, 2.0),
        #     "roll": (-5.0, 5.0)
        #     "pitch": (-5.0, 5.0),
        # }

@configclass
class G1FallPlayEnvCfg(G1FallEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # curriculum
        self.curriculum = None
        self.events.push_robot.params["velocity_range"] = {
            "x": self.push_x_end,
            "y": self.push_y_end,
            "roll": self.push_roll_end,
            "pitch": self.push_pitch_end,
        }

        # viewer
        self.viewer = ViewerCfg(
            origin_type="asset_root",
            asset_name="robot",
            env_index=0,
            eye=(0.0, 3.0, 0.5),
            lookat=(0.0, 0.0, 0.0)
        )

        # ==== Viz data ==== #
        self.viz_data = {
            "com_pos": 0,                  # (3,)
            "left_foot_pos": 0,            # (3,)
            "right_foot_pos": 0,           # (3,)
            "icp_pos": 0,                  # (2,)
            "capture_region_center": 0,    # (2,)
            "capture_region_radius": 0,    # scalar
            "time_hist": 0,                # scalar
            "risk_value": 0,               # scalar
            "icp_ankle_dist_hist": 0,      # scalar
        }

        self.scene.num_envs = 1

        self.plotter: CapturabilityPlotter = CapturabilityPlotter


# Initial Data Collection Environment
@configclass
class G1FallCollectEnvCfg(G1FallEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # curriculum
        self.curriculum = None
        self.events.push_robot.params["velocity_range"] = {
            "x": self.push_x_end,
            "y": self.push_y_end,
            "roll": self.push_roll_end,
            "pitch": self.push_pitch_end,
        }


# Unified Policy (Nominal Policy + RA Network + Safety Policy) Test Environment
@configclass
class G1FallUnifiedPlayEnvCfg(G1FallPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # visualization
        self.viz_data = None
        self.plotter = None

        # Disturbance
        self.events.push_robot.interval_range_s = (4.0, 5.0)

        # Safe Policy info
        self.safe_action_space = self.action_space
        self.safe_observation_space = {"arm": 60, "leg": 45}
        self.safe_state_space = {"arm": 97, "leg": 97}

        

