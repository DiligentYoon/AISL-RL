import isaaclab.sim as sim_utils
import gymnasium
import torch

from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from lib.env.GOAT.base.GOAT_base_env_cfg import GOATBaseEnvCfg, GOAT_Cfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG

from lib.domain_randomizer import randomizer
from lib.domain_randomizer.noise_model import build_noise_std_vector
from lib.utils.plot_utils import PNGSavePlotter
from isaaclab.managers import EventTermCfg as EventTerm
from lib.curriculum.curriculum_cfg import CurriculumManagerCfg, CurriculumParamCfg
from lib.domain_randomizer.commander import UniformVelocityCommandCfg
from isaaclab.managers import SceneEntityCfg

@configclass
class EventCfg:
    """Configuration for domain-randomization events."""

    reset_body = EventTerm(
        func=randomizer.reset_root_state_uniform,
        mode='reset',
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
                "roll": (-0.005, 0.005),
                "pitch": (-0.005, 0.005),
                "yaw": (-0.005, 0.005)},
        },
    )

    reset_robot_joints = EventTerm(
        func=randomizer.reset_joints_by_offset_and_bias,
        mode="reset",
        params={
            "bias": (0.0, 0.0, 0.17, 0.17, 0.135, 0.135, 0.0, 0.0),
            "position_range": (-0.03, 0.03),
            "velocity_range": (0.0, 0.0),
        },
    )

    wheel_physics_material = EventTerm(
      func=randomizer.randomize_rigid_body_material_shared,
      mode='reset',
      params={
          "asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"),
          "static_friction_range": (0.5, 0.6),
          "dynamic_friction_range": (0.4, 0.5),
          "restitution_range": (0.0, 0.02),
          "num_buckets": 1000,
          "make_consistent": True,
      },
  )

@configclass
class GOATTrackEnvCfg(GOATBaseEnvCfg):

    ## =========== Robot Variation (Init pos) ============== ##
    GOAT_cfg: ArticulationCfg = GOAT_Cfg.replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5), # biased initial pos
            joint_pos={
                "hip_L_Joint": 0.0,
                "hip_R_Joint": 0.0,
                "thigh_L_Joint": 0.738,
                "thigh_R_Joint": -0.738,
                "knee_L_Joint": 1.462,
                "knee_R_Joint": -1.462,
                "wheel_L_Joint": 0.0,
                "wheel_R_Joint": 0.0,
                },
            ),
        )

    ## ==================== Environment parameters ==================== ##
    episode_length_s = 10.0
    sim_dt = 0.005                              # 200Hz torque controller
    decimation = 2                              # 100Hz policy
    action_space = 8                            # [L + R, joint pos + wheel velocity]
    observation_space = 32                      # Observation space
    state_space = 38                            # State space including privilege information
    max_episode_length = episode_length_s / (sim_dt * decimation) 

    ## ======================== Controller gain ======================= ##
    action_scale_factor = {"joint" : [1.0, ()],
                           "wheel" : [1.0, ()]}
    
    train_action_scale_factor = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0] # NOTE: Temporary
    # torque_limits = [4.5, 4.5, 4.5, 4.5, 9.0, 9.0, 2.5, 2.5]
    
    ## ==================== Robot configuration ==================== ##
    leg_dof = 3                                 # Hip, Thigh, Knee
    num_leg = 2                                 # Bipedal
    n_leg_j = leg_dof * num_leg
    num_total_joints = n_leg_j + num_leg        # Wheel per legs
    
    ## ========================== Terrain ========================== ##
    default_terrain_static_friction = 0.7       # Default terrain configuration
    default_terrain_dynamic_friction = 0.5
    default_terrain_restitution = 0.0

    ## ==================== Terminal condition ===================== ##
    height_reset_condition = 0.1                # meter (m)
    termination_gravity = 0.6

    ## ======================= Reward Shaping ====================== ##
    soft_torque_limit = 0.7

    r_lin_vel_tracking_weight = 3.0
    r_ang_vel_tracking_weight = 1.0
    r_upright_weight = 1.0

    p_joint_deviation_weight = 1.0
    p_ang_vel_weight = 0.5
    p_joint_limit_weight = 10.0
    p_all_torque_limit_weight = 2.0
    p_all_torque_weight = 0.01
    p_joint_velocity_weight = 0.05
    p_wheel_velocity_weight = 5e-3
    p_joint_accel_weight = 5.0e-7
    p_action_rate_weight = 0.01
    p_terminated_weight = 200.0

    ## ==================== ERFI Configuration ==================== ##
    erfi_enabled: bool = False
    vel_hist_length: int = 4

    ## ======================== Curriculum ======================= ##
    warmup = 0.2
    endup = 0.6
    static_friction_start: tuple[float, float] = (0.9, 1.2)
    static_friction_end: tuple[float, float] = (0.6, 1.2)
    dynamic_friction_start: tuple[float, float] = (0.7, 1.0)
    dynamic_friction_end: tuple[float, float] = (0.5, 1.0)

    # Per-axis observation noise groups (must match _get_observations concat order)
    # std=0.0 for internal values that require no sensor noise injection
    obs_noise_groups_start = {
        "base_ang_vel":      {"dim": 3,  "std": 0.1},   # IMU gyroscope
        "base_rot_w":        {"dim": 4,  "std": 0.01},  # Quaternion (normalized, sensitive)
        "command_inputs_b":  {"dim": 3,  "std": 0.0},   # Internal command (no noise)
        "joint_pos":         {"dim": 6,  "std": 0.005}, # Joint encoder
        "joint_vel":         {"dim": 8,  "std": 0.5},   # Encoder derivative (noisy)
        "previous_actions":  {"dim": 8,  "std": 0.0},   # Internal action buffer (no noise)
    }
    obs_noise_groups_end = {
        "base_ang_vel":      {"dim": 3,  "std": 0.2},
        "base_rot_w":        {"dim": 4,  "std": 0.03},
        "command_inputs_b":  {"dim": 3,  "std": 0.0},
        "joint_pos":         {"dim": 6,  "std": 0.01},
        "joint_vel":         {"dim": 8,  "std": 1.5},
        "previous_actions":  {"dim": 8,  "std": 0.0},
    }
    obs_noise_start = build_noise_std_vector(obs_noise_groups_start)  # list
    obs_noise_end   = build_noise_std_vector(obs_noise_groups_end)    # list

    rfi_start = [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.01, 0.01]
    rfi_end = [0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.02, 0.02]
    rao_start = [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.05, 0.05]
    rao_end = [0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.1, 0.1]
    stop_ratio_start = 0.6
    stop_ratio_end = 0.01

    curriculum = CurriculumManagerCfg(
        warmup=warmup,
        endup=endup,
        params=[
            CurriculumParamCfg(
                name="observation_noise",
                attr_path="cfg/observation_noise_params/std",
                start_value=obs_noise_start,
                end_value=obs_noise_end,
                schedule="linear",
            ),
            CurriculumParamCfg(
                name="rfi_torque_limit",
                attr_path="cfg/rfi_torque_limit",
                start_value=rfi_start,
                end_value=rfi_end,
                schedule="linear",
            ),
            CurriculumParamCfg(
                name="rao_torque_limit",
                attr_path="cfg/rao_torque_limit",
                start_value=rao_start,
                end_value=rao_end,
                schedule="linear",
            ),
            # CurriculumParamCfg(
            #     name="static_friction_coefficient",
            #     attr_path="event_manager/cfg/wheel_physics_material/params/static_friction_range",
            #     start_value= static_friction_start,
            #     end_value=static_friction_end,
            #     schedule="linear",
            # ),
            # CurriculumParamCfg(
            #     name="dynamic_friction_coefficient",
            #     attr_path="event_manager/cfg/wheel_physics_material/params/dynamic_friction_range",
            #     start_value= dynamic_friction_start,
            #     end_value=dynamic_friction_end,
            #     schedule="linear",
            # ),
            # CurriculumParamCfg(
            #     name="stop_command_ratio",
            #     attr_path="cfg/commands/prob_standing_envs",
            #     start_value=stop_ratio_start,
            #     end_value=stop_ratio_end,
            #     schedule="linear"
            # ),
        ]
    )

    ## =================== Domain Randomization =================== ##
    events = EventCfg()
    events.wheel_physics_material.params["static_friction_range"] = static_friction_end
    events.wheel_physics_material.params["dynamic_friction_range"] = dynamic_friction_end

    # Command
    commands: UniformVelocityCommandCfg = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 4.0),
        prob_standing_envs=stop_ratio_end,
        prob_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.7), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5), heading=(0.0, 0.0)
        ),
    )
    
    # ERFI
    rfi_torque_limit: list[float] = rfi_end    # N·m 
    rao_torque_limit: list[float] = rao_end    # N·m

    # Noise Model — std is a list for per-axis control; initialized to max difficulty
    # and overridden to start values by CurriculumManager on env init.
    observation_noise_type: str = "gaussian" # [gaussian, uniform, constant]
    observation_noise_params: dict = {
        "mean": 0.0,
        "std": obs_noise_end,
        "operation": "add",
    }

    ## ==================== Plot variables ==================== ##
    viz_data: dict = {
        "left_hip_torque (Nm)": 0.0,
        "right_hip_torque (Nm)": 0.0,
        "left_thigh_torque (Nm)": 0.0,
        "right_thigh_torque (Nm)": 0.0,
        "left_knee_torque (Nm)": 0.0,
        "right_knee_torque (Nm)": 0.0,
        "left_wheel_torque (Nm)": 0.0,
        "right_wheel_torque (Nm)": 0.0,
        "left_hip_velocity (deg/s)": 0.0,
        "right_hip_velocity (deg/s)": 0.0,
        "left_thigh_velocity (deg/s)": 0.0,
        "right_thigh_velocity (deg/s)": 0.0,
        "left_knee_velocity (deg/s)": 0.0,
        "right_knee_velocity (deg/s)": 0.0,
        "left_wheel_velocity (deg/s)": 0.0,
        "right_wheel_velocity (deg/s)": 0.0,
        "base_linear_velocity (m/s)": 0.0,
        "command_velocity (m/s)": 0.0,
        "command_angular_velocity (deg/s)": 0.0,
        }

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
    )

    # Interactive Scene for DR : replicate_physics parameter shoule be 'False' for USD-level randomization
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=False)
    
    # Terrain
    terrain_importer_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        env_spacing=3.0,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=default_terrain_static_friction,
            dynamic_friction=default_terrain_dynamic_friction,
            restitution=default_terrain_restitution                 # Collision
        ),
        debug_vis=False
    )

    # Sensor
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*wheel.*",
        history_length=0,
        update_period=0.0                                           # Update every period
    )

    # Visualization
    root_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Root"
    )

    root_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

@configclass
class GOATTrackPlayEnvCfg(GOATTrackEnvCfg):
    curriculum = None

    # visualization
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )

    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # plot
    plotter = PNGSavePlotter