
from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, CUBOID_MARKER_CFG, FRAME_MARKER_CFG

from lib.domain_randomizer import randomizer
from lib.domain_randomizer.commander import UniformVelocityCommandCfg
from lib.env.G1.base.G1_base_env_cfg import G1BaseEnvCfg
from lib.utils.plot_utils import CapturabilityPlotter
from isaaclab.managers import SceneEntityCfg


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    # physics_material = EventTerm(
    #     func=randomizer.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.8, 0.8),
    #         "dynamic_friction_range": (0.6, 0.6),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    # add_base_mass = EventTerm(
    #     func=randomizer.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "mass_distribution_params": (-5.0, 5.0),
    #         "operation": "add",
    #     },
    # )

    # base_com = EventTerm(
    #     func=randomizer.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
    #     },
    # )

    # reset
    reset_base = EventTerm(
        func=randomizer.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=randomizer.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=randomizer.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 1.5),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "velocity_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "roll": (-6.0, 6.0), "pitch": (-6.0, 6.0)}},
    )



@configclass
class G1FallEnvCfg(G1BaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 10.0
    sim_dt = 1/200
    decimation = 4          

    ## ========== Multi Agent Setting =========== ##
    possible_agents = ["arm", "leg"]
    action_space = {"arm": 17, "leg": 12}                         
    observation_space = {"arm": 50, "leg": 40}                
    state_space = {"arm": 74, "leg": 74}
    ra_state_space = 74
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
    w_deviation_arm:     float = 1.0
    w_action_rate:       float = 0.05

    w_termination: float = 200
    termination_height: float = 0.3
    termination_gravity: float = 0.5
    termination_ang_vel: float = 15.0
    termination_target_foot: float = 1.0

    soft_torque_limit: float = 0.8

    ## ============== Self collision =============== ##
    allowed_collision_bodies = ["left_ankle_pitch_link",
                                "left_ankle_roll_link",
                                "right_ankle_pitch_link",
                                "right_ankle_roll_link", 
                                "waist_yaw_link",
                                "waist_roll_link"]

    # ===== Gait guidance ===== #
    self_collision_threshold = 0.2
    time_period_min = 0.34
    time_period_max = 0.34
    dstep_min = 0.25
    dstep_max = 0.25
    z_c_min = 0.75
    z_c_max = 0.75
    l_max = 1.0

    # === Surface Function === #
    target_set_threshold = 0.01

    # ==== Viz data ==== #
    plotter = CapturabilityPlotter
    viz_data = {
        "com_pos": 0,                  # (3,)
        "left_foot_pos": 0,            # (3,)
        "right_foot_pos": 0,           # (3,)
        "icp_pos": 0,                  # (2,)
        "capture_region_center": 0,    # (2,)
        "capture_region_radius": 0,    # scalar
        "time_hist": 0,                # scalar
        "m_step_hist": 0,              # scalar
        "icp_ankle_dist_hist": 0,      # scalar
    }


    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=sim_dt, render_interval=decimation)

    # Event
    events: EventCfg = EventCfg()

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
        is_body_frame=False,
    )

    # Terrain
    terrain_importer_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        env_spacing=3.0,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    contact_forces = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", 
                                      history_length=3, 
                                      track_air_time=True)

    # visualization
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )

    torso_rotation_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Torso_rotation"
    )

    # Set the scale of the visualization markers
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    torso_rotation_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
