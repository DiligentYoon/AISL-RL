
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

from lib.domain_randomizer.noise_model import build_noise_std_vector
from lib.domain_randomizer.commander import UniformVelocityCommandCfg
from lib.env.PF_GOAT.base.PF_GOAT_base_env_cfg import PFGOATBaseEnvCfg
from lib.utils.plot_utils import PNGSavePlotter

@configclass
class PFGOATLocomotionEnvCfg(PFGOATBaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 10.0
    sim_dt = 0.005                             
    decimation = 2                          
    action_space = 6   
    observation_space = 28                   
    state_space = 32                     
    max_episode_length = episode_length_s / (sim_dt * decimation) 

    ## ======================== Controller gain ======================= ##
    action_scale_factor = {"joint" : [0.5, ()]}
    torque_limits = [4.5, 4.5, 4.5, 4.5, 9.0, 9.0]

    ## ==================== Reward Shaping ==================== ##
    r_lin_vel_weight:  float = 4.0
    r_ang_vel_weight:  float = 4.0
    r_height_weight:   float = 3.0
    r_gait_weight:      float = 5.0
    r_upright_weight:           float = 3.0

    p_lin_vel_z_weight:          float = 0.5
    p_ang_vel_xy_weight:         float = 0.1
    p_all_torque_weight:         float = 0.001
    p_joint_velocity_weight:     float = 0.01
    p_joint_vel_limit_weight:    float = 2.0
    p_joint_accel_weight:        float = 5.0e-5

    p_joint_limits_weight:      float = 10.0
    p_deviation_hip_weight:     float = 2.0
    p_action_rate_weight:       float = 0.01
    p_termination_weight:       float = 200

    termination_height: float = 0.2
    termination_gravity: float = 0.7
    termination_ang_vel: float = 10.0

    target_height = 0.525   # Height guidance
    time_period = 0.35      # Gait guidance
    soft_torque_limit = 0.7
    joint_vel_limit = 3.14

    # Per-axis observation noise groups
    obs_noise_groups_end = {
        "base_ang_vel":      {"dim": 3,  "std": 0.2},
        "base_rot_w":        {"dim": 4,  "std": 0.05},
        "command":           {"dim": 3,  "std": 0.0},
        "joint_pos":         {"dim": 6,  "std": 0.01},
        "joint_vel":         {"dim": 8,  "std": 1.5},
        "previous_actions":  {"dim": 8,  "std": 0.0},
    }
    obs_noise_end   = build_noise_std_vector(obs_noise_groups_end)    # list

    # Noise Model
    observation_noise_type: str = "gaussian" # [gaussian, uniform, constant]
    observation_noise_params: dict = {
        "mean": 0.0,
        "std": obs_noise_end,
        "operation": "add",
    }

    # Commander
    commands: UniformVelocityCommandCfg = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 5.0),
        prob_standing_envs=0.0,
        prob_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5), heading=(0.0, 0.0)
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = self.sim_dt
        self.sim.render_interval = self.decimation

        # Interactive Scene for DR : replicate_physics parameter shoule be 'False' for USD-level randomization
        self.scene.replicate_physics = False

        self.GOAT_cfg.init_state.pos = (0.0, 0.0, 0.525)
        self.GOAT_cfg.init_state.joint_pos = {"hip_L_Joint": 0.0,
                                              "hip_R_Joint": 0.0,
                                              "thigh_L_Joint": 0.628,
                                              "thigh_R_Joint": -0.628,
                                              "knee_L_Joint": 1.222,
                                              "knee_R_Joint": -1.222}

        # disable event (Initial version)
        self.events.add_base_mass = None
        self.events.add_link_mass = None
        self.events.robot_center_of_mass = None
        self.events.robot_leg_physics_material = None
        self.events.robot_hip_actuator_gain = None
        self.events.robot_thigh_actuator_gain = None
        self.events.robot_knee_actuator_gain = None
        self.observation_noise_type = None
        self.observation_noise_params = None

        # event
        # self.events.robot_leg_physics_material.params["static_friction_range"] = (0.6, 1.2)
        # self.events.robot_leg_physics_material.params["dynamic_friction_range"] = (0.6, 1.0)

        # self.events.robot_hip_actuator_gain.params["stiffness_distribution_params"] = (0.8, 1.1)
        # self.events.robot_hip_actuator_gain.params["damping_distribution_params"] = (0.8, 1.1)
        # self.events.robot_thigh_actuator_gain.params["stiffness_distribution_params"] = (0.8, 1.1)
        # self.events.robot_thigh_actuator_gain.params["damping_distribution_params"] = (0.8, 1.1)
        # self.events.robot_knee_actuator_gain.params["stiffness_distribution_params"] = (0.8, 1.1)
        # self.events.robot_knee_actuator_gain.params["damping_distribution_params"] = (0.8, 1.1)

        # self.events.add_base_mass.params["mass_distribution_params"] = (0.9, 1.05)
        # self.events.add_link_mass.params["mass_distribution_params"] = (0.9, 1.05)
        # self.events.robot_center_of_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["^(?!wheel_).*$"]) 
        # self.events.robot_center_of_mass.params["com_distribution_params"] = ((-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01))

        # NOTE: Initial version
        self.events.reset_robot_joints.params["position_range"] = (-0.05, 0.05)
        self.events.reset_robot_joints.params["velocity_range"] = (-0.01, 0.01)

        self.events.push_robot.interval_range_s = (4.0, 5.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5), 
            "y": (-0.5, 0.5), 
            "roll": (0.0, 0.0), 
            "pitch": (0.0, 0.0)
        }


@configclass
class PFGOATLocomotionPlayEnvCfg(PFGOATLocomotionEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # visualization
        self.goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Command/velocity_goal"
        )

        self.current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Command/velocity_current"
        )

        self.goal_ang_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Command/angular_velocity_goal"
        )

        self.current_ang_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Command/angular_velocity_current"
        )

        # Set the scale of the visualization markers
        self.goal_vel_visualizer_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
        self.current_vel_visualizer_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
        self.goal_ang_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
        self.current_ang_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    
        # viewer
        self.viewer = ViewerCfg(
            origin_type="asset_root",
            asset_name="robot",
            env_index=0,
            eye=(0.0, 3.0, 0.5),
            lookat=(0.0, 0.0, 0.0)
        )

        # self.plotter = PNGSavePlotter
