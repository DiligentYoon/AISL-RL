
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

    # reset
    reset_base = EventTerm(
        func=randomizer.reset_root_state_orientation_biased_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.2, 0.4), 
                           "roll": (-3.14/3, 3.14/3) , "pitch": (-3.14/3, 3.14/3), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-2.0, 2.0),
                "y": (-2.0, 2.0),
                "z": (-0.0, 0.0),
                "roll": (-3.0, 3.0),
                "pitch": (-3.0, 3.0),
                "yaw": (-0.0, 0.0),
            },
            "bias": 3.14/4,
        },
    )

    reset_robot_joints = EventTerm(
        func=randomizer.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.7, 1.3),
            "velocity_range": (-0.0, 0.0),
        },
    )

    # interval
    # push_robot = EventTerm(
    #     func=randomizer.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(2.0, 3.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "velocity_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "roll": (-5.0, 5.0), "pitch": (-5.0, 5.0)}},
    # )



@configclass
class G1SafeEnvCfg(G1BaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 3.0
    sim_dt = 1/200
    decimation = 4          

    ## ========== Multi Agent Setting =========== ##
    possible_agents = ["arm", "leg"]
    action_space = {"arm": 17, "leg": 12}                         
    observation_space = {"arm": 44, "leg": 34}                
    state_space = {"arm": 68, "leg": 68}
    ra_state_space = 11
    num_agents = 2
    action_scale_factor = {"arm": [0.5, ()], 
                           "leg": [0.5, ()]}
    
    ## ========== Safety policy setting ========== ##
    safe_action_space = action_space
    safe_observation_space = {"arm": 44, "leg": 34}
    safe_state_space = {"arm": 68, "leg": 68}

    ## ========== Single Agent Setting ========== ##  
    # action_space = 37                     
    # observation_space = 106                  
    # state_space = 0
    # num_agents = 1
    # action_scale_factor = 0.5

    ## ==================== Reward Shaping ==================== ##
    w_alive:          float = 0.0

    w_ang_vel_xy:         float = 0.01
    w_joint_torque:       float = 1.0e-5
    w_joint_torque_limit: float = 1.0e-4
    w_joint_acc:          float = 1.0e-6
    w_joint_vel:          float = 5.0e-4

    w_limits:               float = 10.0
    w_deviation_hip:        float = 0.5
    w_deviation_torso:      float = 0.0
    w_deviation_arm:        float = 1.0
    w_action_rate:          float = 0.5
    w_prefer_collision:     float = 0.001
    w_not_prefer_collision: float = 0.01

    w_termination: float = 400

    soft_torque_limit: float = 0.8

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
