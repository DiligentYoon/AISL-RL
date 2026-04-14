import isaaclab.sim as sim_utils
import gymnasium
import torch

from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import DCMotorCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from lib.env.GOAT.base.GOAT_base_env_cfg import GOATBaseEnvCfg, GOAT_Cfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG

from lib.domain_randomizer import randomizer
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
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-0.01, 0.01),
                "pitch": (-0.01, 0.01),
                "yaw": (-0.01, 0.01)},
        },
    )

    reset_robot_joints = EventTerm(
        func=randomizer.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.3),
            "velocity_range": (0.0, 0.0),
        },
    )

    wheel_physics_material = EventTerm(
      func=randomizer.randomize_rigid_body_material,
      mode='reset',
      params={
          "asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"),
          "static_friction_range": (0.5, 0.6),
          "dynamic_friction_range": (0.4, 0.5),
          "restitution_range": (0.0, 0.02),
          "num_buckets": 500,
          "make_consistent": True,
      },
  )
    
@configclass
class EventPlayCfg:
    """Configuration for domain-randomization events."""

    reset_body = EventTerm(
        func=randomizer.reset_root_state_uniform,
        mode='reset',
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-0.01, 0.01),
                "pitch": (-0.01, 0.01),
                "yaw": (-0.01, 0.01)},
        },
    )

    reset_robot_joints = EventTerm(
        func=randomizer.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.3),
            "velocity_range": (0.0, 0.0),
        },
    )

    wheel_physics_material = EventTerm(
      func=randomizer.randomize_rigid_body_material,
      mode='reset',
      params={
          "asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"),
          "static_friction_range": (0.3, 0.8),
          "dynamic_friction_range": (0.2, 0.7),
          "restitution_range": (0.0, 0.02),
          "num_buckets": 500,
          "make_consistent": True,
      },
  )

@configclass
class GOATStandDRPPEnvCfg(GOATBaseEnvCfg):
    ## =========== Domain Randomization ============ ##
    events = EventCfg()

    ## =========== Robot Variation (Init pos) ============== ##
    GOAT_cfg: ArticulationCfg = GOAT_Cfg.replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.53),
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
    observation_space = 73                      # Observation space
    state_space = 73                            # State space including privilege information
    max_episode_length = episode_length_s / (sim_dt * decimation) 

    ## ==================== Controller gain ==================== ##
    joint_kp = torch.tensor([[2.0, 2.0, 2.0]])
    joint_kd = torch.tensor([[0.1, 0.1, 0.1]])
    wheel_kp = torch.tensor([[0.05]])
    wheel_ki = torch.tensor([[0.0]])
    PD_LPF_gain = 0.9
    PI_LPF_gain = 0.9
    action_scale_factor = {"joint" : [1.0, ()],
                           "wheel" : [1.0, ()]}
    pos_margin_factor = 1.5
    
    ## ==================== Robot configuration ==================== ##
    leg_dof = 3                                 # Hip, Thigh, Knee
    num_leg = 2                                 # Bipedal
    n_leg_j = leg_dof * num_leg
    num_total_joints = n_leg_j + num_leg        # Wheel per legs
    torque_limits = [2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 2.5, 2.5]
    
    ## ==================== Terrain ==================== ##
    default_terrain_static_friction = 0.7       # Default terrain configuration
    default_terrain_dynamic_friction = 0.5
    default_terrain_restitution = 0.0

    ## ==================== Terminal condition ==================== ##
    height_reset_condition = 0.15                # meter (m)
    termination_gravity = 0.6

    ## ==================== Reward Shaping ==================== ##
    soft_torque_limit = 0.8

    r_joint_deviation_weight = 4.0
    r_lin_vel_tracking_weight = 6.0
    r_ang_vel_tracking_weight = 2.0
    r_upright_weight = 1.0

    p_ang_vel_weight = 2.0
    p_joint_limit_weight = 10.0
    p_all_torque_limit_weight = 1.0
    p_all_torque_weight = 0.1
    p_joint_velocity_weight = 0.05
    p_action_rate_weight = 0.5
    p_terminated_weight = 200.0

    ## ==================== ERFI Configuration ==================== ##
    erfi_enabled: bool = True
    rfi_torque_limit: float = [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.08, 0.08]    # N·m 
    rao_torque_limit: float = [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.08, 0.08]   # N·m
    vel_hist_length: int = 4

    ## ==================== Commands ==================== ##
    commands: UniformVelocityCommandCfg = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 5.0),
        prob_standing_envs=1.0,
        prob_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.1, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5), heading=(0.0, 0.0)
        ),
    )

    ## ==================== Curriculum ==================== ##
    warmup = 0.2
    endup = 0.7
    static_friction_start = (0.5, 0.6)
    static_friction_end = (0.3, 0.8)
    dynamic_friction_start = (0.4, 0.5)
    dynamic_friction_end = (0.2, 0.7)
    obs_noise_start = 0.05
    obs_noise_end = 0.1
    act_noise_start = 0.02
    act_noise_end = 0.075
    rfi_start = [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.08, 0.08]
    rfi_end = [0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.1, 0.1]
    rao_start = [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.08, 0.08]
    rao_end = [0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.1, 0.1]
    stop_ratio_start = 0.7
    stop_ratio_end = 0.3

    curriculum = CurriculumManagerCfg(
        warmup=warmup,
        endup=endup,
        params=[
            CurriculumParamCfg(
                name="static_friction_coefficient",
                attr_path="event_manager/cfg/wheel_physics_material/params/static_friction_range",
                start_value= static_friction_start,
                end_value=static_friction_end,
                schedule="linear",
            ),
            CurriculumParamCfg(
                name="dynamic_friction_coefficient",
                attr_path="event_manager/cfg/wheel_physics_material/params/dynamic_friction_range",
                start_value= dynamic_friction_start,
                end_value=dynamic_friction_end,
                schedule="linear",
            ),
            CurriculumParamCfg(
                name="observation_noise",
                attr_path="cfg/observation_noise_params/std",
                start_value=obs_noise_start,
                end_value=obs_noise_end,
                schedule="linear",
            ),
            CurriculumParamCfg(
                name="action_noise",
                attr_path="cfg/action_noise_params/std",
                start_value=act_noise_start,
                end_value=act_noise_end,
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
            CurriculumParamCfg(
                name="stop_command_ratio",
                attr_path="cfg/commands/prob_standing_envs",
                start_value=stop_ratio_start,
                end_value=stop_ratio_end,
                schedule="linear"
            ),
        ]
    )
    
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
        "right_wheel_velocity (deg/s)": 0.0}

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

    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*wheel.*",
        history_length=0,
        update_period=0.0                                           # Update every period
    )

    # Noise Model
    action_noise_type: str = "gaussian" # [gaussian, uniform, constant]
    action_noise_params: dict = {
        "mean": 1.0,
        "std": 0.075,
        "operation": "scale",
    }
    observation_noise_type: str = "gaussian" # [gaussian, uniform, constant]
    observation_noise_params: dict = {
        "mean": 1.0,
        "std": 0.1,
        "operation": "scale",
    }

    # Visualization
    root_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Root"
    )

    root_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)


@configclass
class GOATStandDRPPPlayEnvCfg(GOATStandDRPPEnvCfg):
    curriculum = None

    # ================ Set max difficulty to curriculum variables ====================
    events = EventPlayCfg()
    events.wheel_physics_material.params["static_friction_range"] = GOATStandDRPPEnvCfg.static_friction_end
    events.wheel_physics_material.params["dynamic_friction_range"] = GOATStandDRPPEnvCfg.dynamic_friction_end

    action_noise_params: dict = {
        "mean": 1.0,
        "std": GOATStandDRPPEnvCfg.act_noise_end,
        "operation": "scale",
    }
    observation_noise_params: dict = {
        "mean": 1.0,
        "std": GOATStandDRPPEnvCfg.obs_noise_end,
        "operation": "scale",
    }

    rfi_torque_limit: float = GOATStandDRPPEnvCfg.rfi_end  # N·m 
    rao_torque_limit: float = GOATStandDRPPEnvCfg.rao_end  # N·m

    commands: UniformVelocityCommandCfg = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 5.0),
        prob_standing_envs=GOATStandDRPPEnvCfg.stop_ratio_end,
        prob_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.1, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5), heading=(0.0, 0.0)
        ),
    )

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
    plotter = None