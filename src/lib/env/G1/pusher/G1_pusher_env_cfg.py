
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs.common import ViewerCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, CUBOID_MARKER_CFG, FRAME_MARKER_CFG

from lib.domain_randomizer import randomizer
from lib.domain_randomizer.commander import UniformVelocityCommandCfg
from lib.env.G1.base.G1_base_env_cfg import G1BaseEnvCfg


# @configclass
# class EventCfg:
#     """Configuration for events."""

#     # startup
#     # physics_material = EventTerm(
#     #     func=randomizer.randomize_rigid_body_material,
#     #     mode="startup",
#     #     params={
#     #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
#     #         "static_friction_range": (0.8, 0.8),
#     #         "dynamic_friction_range": (0.6, 0.6),
#     #         "restitution_range": (0.0, 0.0),
#     #         "num_buckets": 64,
#     #     },
#     # )

#     # add_base_mass = EventTerm(
#     #     func=randomizer.randomize_rigid_body_mass,
#     #     mode="startup",
#     #     params={
#     #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
#     #         "mass_distribution_params": (-5.0, 5.0),
#     #         "operation": "add",
#     #     },
#     # )

#     # base_com = EventTerm(
#     #     func=randomizer.randomize_rigid_body_com,
#     #     mode="startup",
#     #     params={
#     #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
#     #         "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
#     #     },
#     # )

#     # reset
#     reset_base = EventTerm(
#         func=randomizer.reset_root_state_uniform,
#         mode="reset",
#         params={
#             "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-3.14, 3.14)},
#             "velocity_range": {
#                 "x": (0.0, 0.0),
#                 "y": (-0.0, 0.0),
#                 "z": (-0.0, 0.0),
#                 "roll": (-0.0, 0.0),
#                 "pitch": (-0.0, 0.0),
#                 "yaw": (-0.0, 0.0),
#             },
#         },
#     )

#     reset_robot_joints = EventTerm(
#         func=randomizer.reset_joints_by_scale,
#         mode="reset",
#         params={
#             "position_range": (1.0, 1.0),
#             "velocity_range": (0.0, 0.0),
#         },
#     )

#     # interval
#     push_robot = EventTerm(
#         func=randomizer.push_by_setting_velocity,
#         mode="interval",
#         interval_range_s=(4.0, 6.0),
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
#             "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "roll": (-1.5, 1.5), "pitch": (-1.5, 1.5)}},
#     )

@configclass
class G1PusherEnvCfg(G1BaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 10.0
    sim_dt = 1/200
    decimation = 4         

    ## ========== Multi Agent Setting =========== ##
    possible_agents = ["arm", "leg", "adv"]
    adversarial_agents = ["adv"]                             # Adversarial agent
    action_space = {"arm": 17, "leg": 12, "adv": 10}                         
    observation_space = {"arm": 65, "leg": 50, "adv": 74}    # TODO: 나중에 수정            
    state_space = {"arm": 102, "leg": 102, "adv": 74}
    num_agents = 3
    action_scale_factor = {"arm": [0.5, ()], 
                           "leg": [0.5, ()],
                           "adv": [0.5, ()]}
    adv_binary_decode_map = [0,                             # pelvis                             
                             1, 2,                          # hip_pitch
                             4, 5,                          # hip_roll
                             9,                             # torso
                             10, 11,                        # knee
                             18, 19,                        # ankle_roll
                             20, 21,                        # shoulder_yaw
                             22, 23,                        # elbow
                             28, 29]                        # wrist_yaw (include hand)
    adv_agent_action_max = 1                                # Adversarial agent's max num of actions 

    ## ========== Single Agent Setting ========== ##  
    # action_space = 37                     
    # observation_space = 106                  
    # state_space = 0
    # num_agents = 1
    # action_scale_factor = 0.5

    ## ==================== Reward Shaping ==================== ##
    w_track_lin_vel: float = 4.0
    w_track_heading: float = 4.0
    w_track_height : float = 1.0
    w_feet_gait:      float = 6.0
    w_support_xy:     float = 0.2
    w_flat:           float = 2.0

    w_lin_vel_z:          float = 2.0
    w_ang_vel_xy:         float = 0.1
    w_joint_torque:       float = 1.0e-5
    w_joint_torque_limit: float = 0.0
    w_joint_vel:          float = 5.0e-4

    w_limits:            float = 10.0
    w_deviation_swing:   float = 0.5
    w_deviation_hip:     float = 2.0
    w_deviation_torso:   float = 2.0
    w_deviation_arm:     float = 2.0
    w_action_rate:       float = 0.01

    w_termination: float = 200
    termination_height: float = 0.3
    termination_gravity: float = 0.8
    termination_ang_vel: float = 15.0

    # Adversarial agent
    w_falling_adv:          float = 1.0
    w_orientation_adv:      float = 0.5
    w_angular_vel_adv:      float = 0.1
    w_action_budget_adv:    float = 0.1                 # Not using

    # ===== Gait guidance ===== #
    time_period = 0.35
    z_c = 0.75

    ## ============== Self collision =============== ##
    allowed_collision_bodies = ["left_ankle_pitch_link",
                                "left_ankle_roll_link",
                                "right_ankle_pitch_link",
                                "right_ankle_roll_link", 
                                "waist_yaw_link",
                                "waist_roll_link"]

    # Simulation
    # sim: SimulationCfg = SimulationCfg(dt=sim_dt, render_interval=decimation)

    # Event
    # events: EventCfg = EventCfg()

    # Curriculum
    # curriculum = CurriculumManagerCfg(
    #     warmup=0.4,
    #     params=[
    #         CurriculumParamCfg(
    #             name="push_velocity",
    #             attr_path="event_manager/cfg/push_robot/params/velocity_range",
    #             start_value={"x": (0.0, 0.0), "y": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0)},
    #             end_value={"x": (-1.5, 1.5), "y": (-1.5, 1.5), "roll": (-3.5, 3.5), "pitch": (-3.5, 3.5)},
    #             schedule="linear",
    #         ),
    #     ]
    # )

    # Commander
    commands: UniformVelocityCommandCfg = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 5.0),
        prob_standing_envs=0.0,
        prob_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-0.0, 0.0), heading=(0.0, 0.0)
        ),
    )

    # visualization
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )

    goal_ang_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/angular_velocity_goal"
    )

    current_ang_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/angular_velocity_current"
    )

    # Set the scale of the visualization markers
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    goal_ang_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_ang_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = None

@configclass
class G1PusherPlayEnvCfg(G1PusherEnvCfg):
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

        # self.events.push_robot = None
