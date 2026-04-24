import isaaclab.sim as sim_utils
import gymnasium
import torch

from isaaclab.utils import configclass
from lib.env.GOAT.base.GOAT_base_env_cfg import GOATBaseEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from lib.domain_randomizer.randomizer import reset_robot_and_object_root_state_uniform
from lib.domain_randomizer.noise_model import build_noise_std_vector
from lib.domain_randomizer.commander import UniformPositionCommandCfg
from lib.utils.plot_utils import PNGSavePlotter
from lib.curriculum.curriculum_cfg import CurriculumManagerCfg, CurriculumParamCfg
from lib.assets.Jig.object import JIGCFG

@configclass
class GOATStandEnvCfg(GOATBaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 10.0
    sim_dt = 0.005                              # 200Hz torque controller
    decimation = 2                              # 50Hz policy
    action_space = 8                            # [L + R, joint pos + wheel velocity]
    observation_space = 29                      # Observation space
    state_space = 35                            # State space including privilege information
    max_episode_length = episode_length_s / (sim_dt * decimation) 

    ## ======================== Controller gain ======================= ##
    action_scale_factor = {"joint" : [1.0, ()],
                           "wheel" : [1.0, ()]}
    
    train_action_scale_factor = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0] # NOTE: Temporary
    torque_limits = [4.5, 4.5, 4.5, 4.5, 9.0, 9.0, 2.5, 2.5]

    ## ==================== Terminal condition ===================== ##
    height_reset_condition = 0.2 # meter (m)

    ## ======================= Reward Shaping ====================== ##
    soft_torque_limit = 0.7
    joint_vel_limit = 1.0 # rad/s
    target_height = 0.523

    r_upright_weight = 1.0
    r_height_weight = 3.0

    p_illegal_contact_weight = 0.1
    p_joint_deviation_weight = 1.0
    p_lin_vel_weight = 3.0
    p_ang_vel_weight = 0.5
    p_joint_limit_weight = 10.0
    p_all_torque_limit_weight = 2.0
    p_all_torque_weight = 0.01
    p_joint_vel_limit_weight = 1.0
    p_joint_velocity_weight = 0.05
    p_joint_accel_weight = 5.0e-7
    p_action_rate_weight = 0.02
    p_terminated_weight = 100.0

    ## ==================== ERFI Configuration ==================== ##
    erfi_enabled: bool = False
    vel_hist_length: int = 4

    ## ======================== Curriculum ======================= ##
    warmup = 0.2
    endup = 0.6
    static_friction_start: tuple[float, float] = (0.6, 1.2)
    static_friction_end: tuple[float, float] = (0.3, 1.2)
    dynamic_friction_start: tuple[float, float] = (0.6, 0.9)
    dynamic_friction_end: tuple[float, float] = (0.3, 0.9)

    # Per-axis observation noise groups (must match _get_observations concat order)
    # std=0.0 for internal values that require no sensor noise injection
    obs_noise_groups_start = {
        "base_ang_vel":      {"dim": 3,  "std": 0.1},   # IMU gyroscope
        "base_rot_w":        {"dim": 4,  "std": 0.01},  # Quaternion (normalized, sensitive)
        "joint_pos":         {"dim": 6,  "std": 0.005}, # Joint encoder
        "joint_vel":         {"dim": 8,  "std": 0.5},   # Encoder derivative (noisy)
        "previous_actions":  {"dim": 8,  "std": 0.0},   # Internal action buffer (no noise)
    }
    obs_noise_groups_end = {
        "base_ang_vel":      {"dim": 3,  "std": 0.2},
        "base_rot_w":        {"dim": 4,  "std": 0.03},
        "command_inputs_b":  {"dim": 3,  "std": 0.00},  # Command
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
        ]
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

    ## ===================== Jig Object ======================= ##
    jig = JIGCFG.replace(prim_path="/World/envs/env_.*/Jig")

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
        }

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = self.sim_dt
        self.sim.render_interval = self.decimation

        # Interactive Scene for DR : replicate_physics parameter shoule be 'False' for USD-level randomization
        self.scene.replicate_physics = False

        self.GOAT_cfg.init_state.pos = (0.0, 0.0, 0.4605)
        # self.GOAT_cfg.init_state.pos = (0.0, 0.0, 0.523) # Target Height
        self.GOAT_cfg.init_state.joint_pos = {"hip_L_Joint": 0.0,
                                              "hip_R_Joint": 0.0,
                                              "thigh_L_Joint": 0.738,
                                              "thigh_R_Joint": -0.738,
                                              "knee_L_Joint": 1.462,
                                              "knee_R_Joint": -1.462,
                                              "wheel_L_Joint": 0.0,
                                              "wheel_R_Joint": 0.0,}
        
        # robot
        self.GOAT_cfg.actuators["hip"].max_delay = 4
        self.GOAT_cfg.actuators["thigh"].max_delay = 4
        self.GOAT_cfg.actuators["knee"].max_delay = 4
        self.GOAT_cfg.actuators["wheel"].max_delay = 4
        
        # self.GOAT_cfg.actuators["hip"].stiffness = 0.0
        # self.GOAT_cfg.actuators["hip"].damping = 0.0
        # self.GOAT_cfg.actuators["thigh"].stiffness = 0.0
        # self.GOAT_cfg.actuators["thigh"].damping = 0.0
        # self.GOAT_cfg.actuators["knee"].stiffness = 0.0
        # self.GOAT_cfg.actuators["knee"].damping = 0.0
        # self.GOAT_cfg.actuators["wheel"].stiffness = 0.0
        # self.GOAT_cfg.actuators["wheel"].damping = 0.0

        # event
        self.events.robot_leg_physics_material.params["static_friction_range"] = self.static_friction_end
        self.events.robot_leg_physics_material.params["dynamic_friction_range"] = self.dynamic_friction_end
        self.events.robot_wheel_physics_material.params["static_friction_range"] = self.static_friction_end
        self.events.robot_wheel_physics_material.params["dynamic_friction_range"] = self.dynamic_friction_end
        self.events.reset_robot_joints.params["bias"] = (0.0, 0.0, 0.3563, -0.3563, 0.4232, -0.4232, 0.0, 0.0)
        self.events.reset_robot_joints.params["position_range"] = (-0.0, 0.0)

        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 0.5)
        self.events.add_link_mass.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.rigid_body_mass_inertia.params["mass_inertia_distribution_params"] = (0.8, 1.2)
        self.events.robot_center_of_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["^(?!wheel_).*$"]) # exclude wheel
        self.events.robot_center_of_mass.params["com_distribution_params"] = ((-0.025, 0.025), (-0.025, 0.025), (-0.025, 0.025))

        # robot must be syncronized with jig object. 
        self.events.reset_body = EventTerm(
            func=reset_robot_and_object_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.0, 0.0),
                    "y": (-0.0, 0.0),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {
                    "x": (-0.0, 0.0),
                    "y": (-0.0, 0.0),
                    "z": (-0.0, 0.0),
                    "roll": (-0.0, 0.0),
                    "pitch": (-0.0, 0.0),
                    "yaw": (-0.0, 0.0),
                },
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("jig"),
                "object_relative_pos": (0.0, 0.0, 0.0),  
                "object_relative_yaw": 0.0,          
            }
        )

        # self.events.add_base_mass = None
        # self.events.robot_center_of_mass = None
        # self.events.rigid_body_mass_inertia = None
        # self.events.robot_leg_actuator_gain = None
        # self.events.robot_wheel_actuator_gain = None
        # self.events.robot_center_of_mass = None

        self.events.push_robot = None

@configclass
class GOATStandPlayEnvCfg(GOATStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum = None

        # viewer
        self.viewer = ViewerCfg(
            origin_type="asset_root",
            asset_name="robot",
            env_index=0,
            eye=(0.0, 3.0, 0.5),
            lookat=(0.0, 0.0, 0.0)
        )

        # disable randomization
        self.events.add_base_mass = None
        self.events.add_link_mass = None
        self.events.rigid_body_mass_inertia = None
        self.events.robot_leg_actuator_gain = None
        self.events.robot_wheel_actuator_gain = None
        self.events.robot_center_of_mass = None
        self.events.reset_body.params["pose_range"]["yaw"] = (-0.0, 0.0)
        self.events.reset_robot_joints.params["position_range"] = (-0.0, 0.0)
        # disable noise
        self.observation_noise_type = None
        self.observation_noise_params = None

        # robot
        self.GOAT_cfg.actuators["hip"].max_delay = 0
        self.GOAT_cfg.actuators["thigh"].max_delay = 0
        self.GOAT_cfg.actuators["knee"].max_delay = 0
        self.GOAT_cfg.actuators["wheel"].max_delay = 0

        # plot
        self.plotter = PNGSavePlotter