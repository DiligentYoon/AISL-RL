from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm

from lib.domain_randomizer.noise_model import build_noise_uniform_vector
from lib.curriculum.curriculum_cfg import CurriculumManagerCfg, CurriculumParamCfg

from lib.env.WF_GOAT.stand.WF_GOAT_stand_env_cfg import WFGOATStandEnvCfg, WFGOATStandPlayEnvCfg


@configclass
class WFGOATTrackEnvCfg(WFGOATStandEnvCfg):
    ## ==================== Environment parameters ==================== ##
    observation_space = 26                      # Observation space
    state_space = 32                            # State space including privilege information

    ## ======================= Reward Shaping ====================== ##
    r_height_weight = 10.0
    r_upright_weight = 1.0
    r_lin_vel_tracking_weight = 8.0
    r_ang_vel_tracking_weight = 8.0

    p_hip_deviation_weight = 2.0
    p_illegal_contact_weight = 2.0
    p_joint_deviation_lr_weight = 4.0             

    # Per-axis observation noise groups
    obs_noise_groups_end = {
        "base_ang_vel":      {"dim": 3,  "min": -0.1,  "max": 0.1},
        "gravity_vector":    {"dim": 3,  "min": -0.05, "max": 0.05},
        "command":           {"dim": 4,  "min": 0.0,   "max": 0.0},
        "joint_pos":         {"dim": 4,  "min": -0.03, "max": 0.03},
        "joint_vel":         {"dim": 6,  "min": -1.5,  "max": 1.5},
        "previous_actions":  {"dim": 6,  "min": 0.0,   "max": 0.0},
    }
    obs_noise_min, obs_noise_max = build_noise_uniform_vector(obs_noise_groups_end)    # list

    # Noise Model
    observation_noise_type: str = "uniform" # [gaussian, uniform, constant]
    observation_noise_params: dict = {
        "min": obs_noise_min,
        "max": obs_noise_max,
        "operation": "add",
    }

    def __post_init__(self):
        super().__post_init__()

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