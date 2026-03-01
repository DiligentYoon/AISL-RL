import isaaclab.sim as sim_utils
import gymnasium
import torch

from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import DCMotorCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from lib.env.GOAT.base.GOAT_base_env_cfg import GOATBaseEnvCfg

from lib.domain_randomizer import randomizer
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

@configclass
class EventCfg:
    """Configuration for domain-randomization events."""

    reset_joint = EventTerm(
        func=randomizer.reset_joints_by_offset,
        mode='reset',
        params={      
            "position_range": (0, 0),
            "velocity_range": (0, 0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_body = EventTerm(
        func=randomizer.reset_root_state_uniform,
        mode='reset',
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.63, 0.63), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0)},
        },
    )

    wheel_physics_material = EventTerm(
      func=randomizer.randomize_rigid_body_material,
      mode='reset',
      params={
          "asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"),
          "static_friction_range": (0.6, 1.3),
          "dynamic_friction_range": (0.6, 1.3),
          "restitution_range": (1.0, 1.0),
          "num_buckets": 500,
          "make_consistent": True,
      },
  )

@configclass
class GOATStandDRPPEnvCfg(GOATBaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 5.0
    sim_dt = 0.005                              # 200Hz torque controller
    decimation = 2                              # 100Hz policy
    action_space = 8                            # [L + R, joint pos + wheel velocity]
    observation_space = 24                      # Observation space
    state_space = 39                            # State space including privilege information
    max_episode_length = episode_length_s / (sim_dt * decimation) 

    ## ==================== Controller gain ==================== ##
    joint_kp = torch.tensor([[0.330, 4.270, 0.40]])
    joint_kd = torch.tensor([[0.015, 0.010, 0.018]])
    wheel_kp = torch.tensor([[0.3]])
    wheel_ki = torch.tensor([[0.3]])
    PD_LPF_gain = 0.049
    PI_LPF_gain = 0.049
    action_scale_factor = {"joint" : [1.0, ()],
                           "wheel" : [1.0, ()]}
    pos_margin_factor = 1.2
    
    ## ==================== Robot configuration ==================== ##
    leg_dof = 3                                 # Hip, Thigh, Knee
    num_leg = 2                                 # Bipedal
    n_leg_j = leg_dof * num_leg
    num_total_joints = n_leg_j + num_leg        # Whee per legs
    torque_limits = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    
    ## ==================== Curriculum parameters ==================== ##
    total_DR_curriculum_level = 5               # Domain Randomization curriculum level
    total_task_curriculum_level = ["balancing", "recovery"]
    success_rate_buffer_len = 500

    max_base_acceleration_noise_per = 10        # Noise percentage (%)
    max_base_angular_vel_noise_per = 20
    max_gravity_vector_noise_per = 5
    max_base_quaternion_noise_per = 5
    max_joint_pos_noise_per = 3
    max_joint_vel_noise_per = 150

    max_terrain_friction_random_per = 50        # Friction randomization (%)
    max_terrain_restitution_random_per = 50     # Restitution randomization (%)

    default_terrain_static_friction = 0.7       # Default terrain configuration
    default_terrain_dynamic_friction = 0.5
    default_terrain_restitution = 0.4

    ## ==================== Terminal condition ==================== ##
    height_reset_condition = 0.4                # meter (m)
    base_tilt_reset_condition = 28              # degree

    ## ==================== Reward Shaping ==================== ##
    target_height = 0.45                        # meter (m)
    upright_threshold = 5                       # degree
    height_threshold = 0.1                      # meter (m)
    curriculum_level_up_threshold = 0.8         # success rate
    curriculum_level_down_threshold = 0.2
    soft_torque_limit = 0.8

    r_upright_weight = 3.0
    r_height_weight = 2.0
    r_alive_weight = 2.0

    p_lin_vel_weight = 0.01
    p_ang_vel_weight = 0.01
    p_joint_limit_weight = 5.0
    p_all_torque_limit_weight = 0.5
    p_all_torque_weight = 0.1
    p_joint_velocity_weight = 0.01
    p_action_rate_weight = 0.05
    p_terminated_weight = 200.0

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
        "left_hip_target (deg)": 0.0,
        "right_hip_target (deg)": 0.0,
        "left_thigh_target (deg)": 0.0,
        "right_thigh_target (deg)": 0.0,
        "left_knee_target (deg)": 0.0,
        "right_knee_target (deg)": 0.0,
        "left_wheel_target (rpm)": 0.0,
        "right_wheel_target (rpm)": 0.0,}

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
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

    
    # Domain Randomization
    events = EventCfg()

    # Noise Model
    action_noise_type: str = "gaussian" # [gaussian, uniform, constant]
    action_noise_params: dict = {
        "mean": 0.0,
        "std": 0.05,
        "operation": "add",
    }
    observation_noise_type: str = "gaussian" # [gaussian, uniform, constant]
    observation_noise_params: dict = {
        "mean": 0.0,
        "std": 0.08,
        "operation": "add",
    }