
import isaaclab.sim as sim_utils

from isaaclab.utils import configclass
from lib.env.GOAT.base.GOAT_base_env_cfg import GOATBaseEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

from lib.domain_randomizer.noise_model import build_noise_std_vector
from lib.utils.plot_utils import PNGSavePlotter
from lib.curriculum.curriculum_cfg import CurriculumManagerCfg, CurriculumParamCfg
from lib.assets.objects.Jig.object import JIGCFG
from lib.domain_randomizer.commander import UniformVelocityCommandCfg

from lib.env.GOAT.stand.mdp.randomizer import reset_robot_and_object_root_state_uniform, reset_joint_state_from_buffer

@configclass
class GOATStandEnvCfg(GOATBaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 10.0
    sim_dt = 0.005                              # 200Hz torque controller
    decimation = 2                              # 50Hz policy
    action_space = 8                            # [L + R, joint pos + wheel velocity]
    observation_space = 32                      # Observation space
    state_space = 38                            # State space including privilege information
    max_episode_length = episode_length_s / (sim_dt * decimation) 

    ## ======================== Controller gain ======================= ##
    action_scale_factor = {"joint" : [0.5, ()],
                           "wheel" : [5.0, ()]}

    torque_limits = [4.5, 4.5, 4.5, 4.5, 9.0, 9.0, 2.5, 2.5]

    ## ======================= Reward Shaping ====================== ##
    soft_torque_limit = 0.7
    joint_vel_limit = 2.0 # rad/s
    terminated_tilt = 0.6
    terminated_joint_vel_limit = 2.0 * joint_vel_limit  # rad/s

    terminated_lin_vel_limit_z_start = 0.2
    terminated_lin_vel_limit_z_end = 0.15
    terminated_lin_vel_limit_z = 0.2

    height_reset_condition = 0.2 # meter (m)
    target_height = 0.523

    r_height_weight = 6.0
    r_upright_weight = 4.0
    r_joint_tracking_weight = 5.0
    r_velocity_tracking_weight = 4.0

    p_illegal_contact_weight = 1.0
    p_joint_deviation_lr_weight = 1.0
    p_ang_vel_weight = 0.1

    p_all_torque_limit_weight = 0.0
    p_all_torque_weight = 0.001
    p_joint_vel_limit_weight = 2.0
    p_joint_velocity_weight = 0.01
    p_joint_accel_weight = 5.0e-5
    p_action_rate_weight = 0.1
    p_terminated_weight = 200.0

    # Per-axis observation noise groups
    obs_noise_groups_end = {
        "base_ang_vel":      {"dim": 3,  "std": 0.1},
        "base_rot_w":        {"dim": 4,  "std": 0.01},
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
    
    ## ======================== Curriculum ======================= ##
    # warmup = 0.1
    # endup = 0.25

    # curriculum = CurriculumManagerCfg(
    #     warmup=warmup,
    #     endup=endup,
    #     params=[
    #         CurriculumParamCfg(
    #             name="lin_vel_limit",
    #             attr_path="cfg/terminated_lin_vel_limit_z",
    #             start_value=terminated_lin_vel_limit_z_start,
    #             end_value=terminated_lin_vel_limit_z_end,
    #             schedule="linear",
    #         ),
    #     ]
    # )

    ## ======================== Command ======================= ##
    commands: UniformVelocityCommandCfg = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 5.0),
        prob_standing_envs=0.0,
        prob_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.0, 0.0), heading=(0.0, 0.0)
        ),
    )

    ## ===================== Jig Object ======================= ##
    jig = JIGCFG.replace(prim_path="/World/envs/env_.*/Jig")

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
        
        # event
        self.events.robot_leg_physics_material.params["static_friction_range"] = (0.3, 0.9)
        self.events.robot_leg_physics_material.params["dynamic_friction_range"] = (0.3, 0.7)
        self.events.robot_wheel_physics_material.params["static_friction_range"] = (0.3, 0.9)
        self.events.robot_wheel_physics_material.params["dynamic_friction_range"] = (0.3, 0.7)

        self.events.robot_hip_actuator_gain.params["stiffness_distribution_params"] = (0.7, 1.05)
        self.events.robot_hip_actuator_gain.params["damping_distribution_params"] = (0.7, 1.05)
        self.events.robot_thigh_actuator_gain.params["stiffness_distribution_params"] = (0.7, 1.05)
        self.events.robot_thigh_actuator_gain.params["damping_distribution_params"] = (0.7, 1.05)
        self.events.robot_knee_actuator_gain.params["stiffness_distribution_params"] = (0.7, 1.05)
        self.events.robot_knee_actuator_gain.params["damping_distribution_params"] = (0.7, 1.05)
        self.events.robot_wheel_actuator_gain.params["stiffness_distribution_params"] = (0.7, 1.05)
        self.events.robot_wheel_actuator_gain.params["damping_distribution_params"] = (0.7, 1.05)

        self.events.add_base_mass.params["mass_distribution_params"] = (0.9, 1.05)
        self.events.add_link_mass.params["mass_distribution_params"] = (0.9, 1.05)
        self.events.robot_center_of_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["^(?!wheel_).*$"]) 
        self.events.robot_center_of_mass.params["com_distribution_params"] = ((-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01))

        # robot's joints should be reset by dataset
        self.events.reset_robot_joints = EventTerm(
            func=reset_joint_state_from_buffer,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "dataset_path":"logs/GOAT_stand/joint_buffer/random_joint_pos_3.pt",
            }
        )

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

        self.events.push_robot = None

@configclass
class GOATStandPlayEnvCfg(GOATStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum = None

        # disable randomization
        self.events.add_base_mass = None
        self.events.add_link_mass = None
        self.events.robot_center_of_mass = None
        self.events.robot_leg_physics_material = None
        self.events.robot_wheel_physics_material = None
        self.events.robot_hip_actuator_gain = None
        self.events.robot_thigh_actuator_gain = None
        self.events.robot_knee_actuator_gain = None
        self.events.robot_wheel_actuator_gain = None
        # self.events.reset_body.params["pose_range"]["yaw"] = (-0.0, 0.0)

        # disable noise
        self.observation_noise_type = None
        self.observation_noise_params = None

        ## ==================== Plot variables ==================== ##
        self.viz_data: dict = {
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
            "left_hip_action": 0.0,
            "right_hip_action": 0.0,
            "left_thigh_action": 0.0,
            "right_thigh_action": 0.0,
            "left_knee_action": 0.0,
            "right_knee_action": 0.0,
            "left_wheel_action": 0.0,
            "right_wheel_action": 0.0,
            "base_lin_x_velocity (m/s)": 0.0,
            "base_lin_y_velocity (m/s)": 0.0,
            "base_lin_z_velocity (m/s)": 0.0,
            "base_ang_velocity (deg/s)": 0.0,
            "effective_contact_force (N)": 0.0
            }

        # visualization
        self.goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Command/velocity_goal"
        )

        self.current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Command/velocity_current"
        )
        self.goal_vel_visualizer_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
        self.current_vel_visualizer_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)

        # viewer
        self.viewer = ViewerCfg(
            origin_type="asset_root",
            asset_name="robot",
            env_index=0,
            eye=(0.0, 3.0, 0.5),
            lookat=(0.0, 0.0, 0.0)
        )

        # plot
        self.plotter = PNGSavePlotter