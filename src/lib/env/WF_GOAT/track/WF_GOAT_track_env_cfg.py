from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm

from lib.domain_randomizer.noise_model import build_noise_uniform_vector
from lib.env.WF_GOAT.stand.mdp.commander import UniformVelocityHeightCommandCfg
from lib.env.WF_GOAT.stand.WF_GOAT_stand_env_cfg import WFGOATStandEnvCfg, WFGOATStandPlayEnvCfg

from lib.env.WF_GOAT.track.mdp.randomizer import reset_joint_offset_bias


@configclass
class WFGOATTrackEnvCfg(WFGOATStandEnvCfg):
    ## ======================= Reward Shaping ====================== ##
    p_wheel_velocity_weight = 0.0

    ## ================== Post Initialization ================== ##
    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = self.sim_dt
        self.sim.render_interval = self.decimation_apply

        self.commands.prob_standing_envs = 0.2
        self.commands.ranges.lin_vel_x=(-1.0, 1.0)
        self.commands.ranges.ang_vel_z=(-1.0, 1.0)

        # only for TorqueDelayedPDActuator class
        # self.is_torque_delayed = False 
        # self.GOAT_cfg.actuators["thigh"].decimation = self.decimation_apply
        # self.GOAT_cfg.actuators["knee"].decimation  = self.decimation_apply
        # self.GOAT_cfg.actuators["wheel"].decimation = self.decimation_apply


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

        # self.events.reset_joint_bias = EventTerm(
        #     func=reset_joint_offset_bias,
        #     mode="reset",
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", joint_names="knee_.*"),
        #         "bias_range": (-0.05, -0.05),
        #     }

        # )