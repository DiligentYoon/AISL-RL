
from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from lib.env.G1.recovery.G1_recovery_env_cfg import G1RecoveryEnvCfg
from lib.domain_randomizer import randomizer
from lib.env.G1.fall.mdp.randomizer import push_and_log
from lib.utils.plot_utils import CapturabilityPlotter, PNGSavePlotter
from lib.curriculum.curriculum_cfg import CurriculumManagerCfg, CurriculumParamCfg

# Environment for training Reach-avoid network
@configclass
class G1FallEnvCfg(G1RecoveryEnvCfg):

    # === RA agent config === #
    ra_state_space = 10
    body_hist_length = 4

    # === SafeFall baseline config === #
    safe_fall_obs_dim = 63

    # === RA Setting === #
    l_max = 1.0
    target_set_threshold = 0.1

    # === Curriculum === #
    push_x_end = (-1.5, 1.5)
    push_y_end = (-1.5, 1.5)
    push_roll_end = (-4.0, 4.0)
    push_pitch_end = (-4.0, 4.0)

    def __post_init__(self):
        super().__post_init__()

        # self collision off for deleting fishy and chaotic collisions
        self.robot.spawn.articulation_props.enabled_self_collisions = False

        self.termination_height = 0.2

        self.curriculum: CurriculumManagerCfg = CurriculumManagerCfg(
            warmup=0.3,
            endup=0.5,
            params=[
                CurriculumParamCfg(
                    name="push_range_x",
                    attr_path="cfg/events/push_robot/params/velocity_range/x",
                    start_value=self.events.push_robot.params["velocity_range"]["x"],
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

# Environment for eveluating Reach-avoid network
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
            "risk_value": 0,               # scalar
            "icp_ankle_dist_hist": 0,      # scalar
        }

        self.scene.num_envs = 1

        self.plotter: PNGSavePlotter = PNGSavePlotter


# Data Collection Environment for Initial dataset construction
@configclass
class G1FallCollectEnvCfg(G1FallEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Episode
        self.episode_length_s = 5.0

        # curriculum
        self.curriculum = None

        # events
        self.events.push_robot.interval_range_s = (2.0, 3.0) 
        self.events.push_robot.params["velocity_range"] = {
            "x": self.push_x_end,
            "y": self.push_y_end,
            "roll" : self.push_roll_end,
            "pitch": self.push_pitch_end,
        }


# Data Collection Environment for disturbance region analysis.
# One env carries one disturbance condition, so the push must be logged and must
# fire exactly once per episode.
@configclass
class G1FallRegionCollectEnvCfg(G1FallCollectEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # commands
        self.commands.resampling_time_range = (3.0, 3.0)
        self.commands.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.ranges.lin_vel_y = (0.0, 0.0)

        # events
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}

        self.events.push_robot.func = push_and_log
        self.events.push_robot.interval_range_s = (3.0, 3.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": self.push_x_end,
            "y": self.push_y_end,
            "roll" : (0.0, 0.0),    # disabled for the (vx, vy) sweep
            "pitch": (0.0, 0.0),
        }


# Unified Policy (Nominal Policy + Predictor + Safety Policy) Test Environment
@configclass
class G1FallUnifiedPlayEnvCfg(G1FallPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 5.0

        self.robot.spawn.articulation_props.enabled_self_collisions = True

        # visualization
        self.viz_data = None
        self.plotter = None

        # Disturbance
        self.events.push_robot.interval_range_s = (3.0, 3.0)

        # Safe Policy info (multi agent)
        self.possible_agents = ["arm", "leg"]
        self.num_safe_agents = 2
        self.safe_action_space = self.action_space
        self.safe_observation_space = {"arm": 60, "leg": 45}
        self.safe_state_space = {"arm": 97, "leg": 97}

        # Safe Policy info (single agent)
        # self.num_safe_agents = 1
        # self.safe_action_space = 29
        # self.safe_observation_space = 96
        # self.safe_state_space = 96

        # plotter
        self.plotter = PNGSavePlotter

        self.viz_data = {
            "contact_num": 0.0,
            "action_diff": 0.0,
            "risk_value": 0.0,         
            "max_torque": 0.0,
            "max_contact_force": 0.0,
            "max_contact_impulse": 0.0,
            "torso_contact_force": 0.0,
            "mean_joint_deviation": 0.0,
        }

        

