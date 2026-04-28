
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs.common import ViewerCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

from lib.domain_randomizer import randomizer
from lib.domain_randomizer.commander import UniformVelocityCommandCfg
from lib.env.G1.base.G1_base_env_cfg import G1BaseEnvCfg

@configclass
class G1RecoveryEnvCfg(G1BaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 10.0
    sim_dt = 1/200
    decimation = 4         

    ## ========== Multi Agent Setting =========== ##
    possible_agents = ["arm", "leg"]
    action_space = {"arm": 17, "leg": 12}                         
    observation_space = {"arm": 50, "leg": 40}                
    state_space = {"arm": 74, "leg": 74}
    num_agents = 2
    action_scale_factor = {"arm": [0.5, ()], 
                           "leg": [0.5, ()]}

    ## ========== Single Agent Setting ========== ##  
    # action_space = 37                     
    # observation_space = 106                  
    # state_space = 0
    # num_agents = 1
    # action_scale_factor = 0.5

    ## ==================== Reward Shaping ==================== ##
    w_track_lin_vel: float = 4.0
    w_track_heading: float = 2.0
    w_track_height : float = 1.0

    w_feet_gait:      float = 6.0
    w_feet_slide:     float = 2.0
    w_support_xy:     float = 0.2
    w_self_collision: float = 0.01
    w_flat:           float = 2.0

    w_lin_vel_z:          float = 0.5
    w_ang_vel_xy:         float = 0.1
    w_joint_torque:       float = 1.0e-5
    w_joint_torque_limit: float = 1.0e-4
    w_joint_acc:          float = 1.0e-6
    w_joint_vel:          float = 5.0e-4

    w_limits:            float = 10.0
    w_deviation_hip:     float = 2.0
    w_deviation_torso:   float = 2.0
    w_deviation_arm:     float = 0.2
    w_action_rate:       float = 0.05

    w_termination: float = 200
    termination_height: float = 0.3
    termination_gravity: float = 0.5
    termination_ang_vel: float = 15.0

    # ===== Gait guidance ===== #
    time_period = 0.34
    z_c = 0.75


    ## ============== Self collision =============== ##
    allowed_collision_bodies = ["left_ankle_pitch_link",
                                "left_ankle_roll_link",
                                "right_ankle_pitch_link",
                                "right_ankle_roll_link", 
                                "waist_yaw_link",
                                "waist_roll_link"]

    # Commander
    commands: UniformVelocityCommandCfg = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 5.0),
        prob_standing_envs=0.02,
        prob_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        ),
    )

    # visualization
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )

    # Set the scale of the visualization markers
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = None


@configclass
class G1RecoveryPlayEnvCfg(G1RecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
    
        # viewer
        self.viewer = ViewerCfg(
            origin_type="asset_root",
            asset_name="robot",
            env_index=0,
            eye=(0.0, 3.0, 0.5),
            lookat=(0.0, 0.0, 0.0)
        )

        self.scene.num_envs = 1

        self.events.push_robot = None
