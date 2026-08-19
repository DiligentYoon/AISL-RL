from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm

from lib.domain_randomizer.noise_model import build_noise_uniform_vector
from lib.curriculum.curriculum_cfg import CurriculumManagerCfg, CurriculumParamCfg
from lib.domain_randomizer.randomizer import push_by_setting_velocity

from lib.env.WF_GOAT.stand.WF_GOAT_stand_env_cfg import WFGOATStandEnvCfg, WFGOATStandPlayEnvCfg
from lib.env.WF_GOAT.track.mdp.commander import UniformVelocityHeightCommandCfg

@configclass
class WFGOATTrackEnvCfg(WFGOATStandEnvCfg):
    ## ==================== Environment parameters ==================== ##
    observation_space = 32                      # Observation space
    state_space = 38                            # State space including privilege information

    ## ======================= Reward Shaping ====================== ##
    r_height_weight = 6.0
    r_upright_weight = 4.0
    r_lin_vel_tracking_weight = 6.0
    r_ang_vel_tracking_weight = 4.0

    p_hip_deviation_weight = 1.0
    p_joint_deviation_lr_weight = 4.0
    p_joint_limit_weight = 10.0

    # Jig Delete Logic
    jig_release_height = 0.4
    jig_release_hold_step = 10
    jig_release_depth = -5.0                    # world-frame z the jig is teleported to on release


    # Per-axis observation noise groups
    obs_noise_groups_end = {
        "base_ang_vel":      {"dim": 3,  "min": -0.1,  "max": 0.1},
        "gravity_vector":    {"dim": 3,  "min": -0.05, "max": 0.05},
        "command":           {"dim": 4,  "min": 0.0,   "max": 0.0},
        "joint_pos":         {"dim": 6,  "min": -0.01, "max": 0.01},
        "joint_vel":         {"dim": 8,  "min": -1.5,  "max": 1.5},
        "previous_actions":  {"dim": 8,  "min": 0.0,   "max": 0.0},
    }
    obs_noise_min, obs_noise_max = build_noise_uniform_vector(obs_noise_groups_end)    # list

    # Noise Model
    observation_noise_type: str = "uniform" # [gaussian, uniform, constant]
    observation_noise_params: dict = {
        "min": obs_noise_min,
        "max": obs_noise_max,
        "operation": "add",
    }
    
    ## ======================== Curriculum ======================= ##

    curriculum = CurriculumManagerCfg(
        params=[
            CurriculumParamCfg(
                name="lin_vel_range",
                attr_path="cfg/commands/ranges/lin_vel_x",
                start_value=(0.0, 0.0),
                end_value=(-0.5, 0.5),
                schedule="step",
                schedule_kwargs={
                    "steps": [0.3, 0.7],
                }
            ),
            CurriculumParamCfg(
                name="ang_vel_range",
                attr_path="cfg/commands/ranges/ang_vel_z",
                start_value=(0.0, 0.0),
                end_value=(-0.5, 0.5),
                schedule="step",
                schedule_kwargs={
                    "steps": [0.3, 0.7],
                }
            ),
        ]
    )

    # Command
    commands: UniformVelocityHeightCommandCfg = UniformVelocityHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 5.0),
        height_resampling_time_range=(3.0, 4.0),
        prob_standing_envs=0.1,
        prob_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        ranges=UniformVelocityHeightCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5), heading=(0.0, 0.0), height=(0.45, 0.55)
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0
        self.max_episode_length = self.episode_length_s / (self.sim_dt * self.decimation) 

        # Randomization
        self.events.add_base_mass = None
        self.events.add_link_mass = None
        self.events.robot_base_center_of_mass = None
        self.events.robot_link_center_of_mass = None 
        self.events.robot_leg_physics_material = None
        self.events.robot_wheel_physics_material = None
        self.events.robot_hip_actuator_gain = None
        self.events.robot_thigh_actuator_gain = None
        self.events.robot_knee_actuator_gain = None
        self.events.robot_wheel_actuator_gain = None
        self.observation_noise_type = None
        self.observation_noise_params = None

        self.events.push_robot = EventTerm(
            func=push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 4.0),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
                "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
        )

@configclass
class WFGOATTrackPlayEnvCfg(WFGOATTrackEnvCfg, WFGOATStandPlayEnvCfg):
    """Play variant of the track environment.

    The MRO is TrackPlay -> TrackEnvCfg -> StandPlayEnvCfg -> StandEnvCfg, so a single
    super().__post_init__() applies the play-side deactivations first and the track-side
    settings on top of them.
    """

    def __post_init__(self):
        super().__post_init__()
        self.curriculum = None
        # TrackEnvCfg.__post_init__ runs after the play deactivations and re-enables the push
        # event, so it has to be switched off again here.
        self.events.push_robot = None