import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.markers import VisualizationMarkersCfg

from lib.env.GOAT.base.GOAT_base_env_cfg import GOATBaseEnvCfg
from lib.env.GOAT.track.mdp.commander import UniformJointPositionCommandCfg
from lib.domain_randomizer.noise_model import build_noise_std_vector
from lib.domain_randomizer.randomizer import reset_joints_by_offset
from lib.utils.plot_utils import PNGSavePlotter


@configclass
class GOATTrackFixedEnvCfg(GOATBaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 10.0
    sim_dt = 0.005                              # 200Hz torque controller
    decimation = 2                              # 100Hz policy
    action_space = 6                            # [L + R, joint pos]
    observation_space = 24                      # Observation space
    max_episode_length = episode_length_s / (sim_dt * decimation)

    ## ======================== Controller gain ======================= ##
    action_scale_factor = {"joint" : [1.0, ()],
                           "wheel" : [5.0, ()]}
    
    torque_limits = [4.5, 4.5, 4.5, 4.5, 9.0, 9.0, 2.5, 2.5]

    ## ======================= Reward Shaping ====================== ##
    soft_torque_limit = 0.7
    joint_vel_limit = 1.0 # rad/s

    r_joint_tracking_weight = 1.0
    p_joint_limit_weight = 0.0
    p_all_torque_limit_weight = 0.0
    p_all_torque_weight = 0.0
    p_joint_vel_limit_weight = 0.0
    p_joint_velocity_weight = 0.0
    p_joint_accel_weight = 0.0
    p_action_rate_weight = 0.0
    # p_joint_limit_weight = 15.0
    # p_all_torque_limit_weight = 2.0
    # p_all_torque_weight = 0.005
    # p_joint_vel_limit_weight = 1.0
    # p_joint_velocity_weight = 0.05
    # p_joint_accel_weight = 5.0e-6
    # p_action_rate_weight = 0.2

    ## ======================== Curriculum ======================= ##

    # Per-axis observation noise groups (must match _get_observations concat order)
    # std=0.0 for internal values that require no sensor noise injection
    obs_noise_groups_end = {
        "joint_pos":         {"dim": 6,  "std": 0.03},
        "joint_vel":         {"dim": 6,  "std": 1.5},
        "previous_actions":  {"dim": 6,  "std": 0.0},
        "command":           {"dim": 6,  "std": 0.0},   # internal target command (no noise)
    }
    obs_noise_end   = build_noise_std_vector(obs_noise_groups_end)    # list

    # Noise Model — std is a list for per-axis control; initialized to max difficulty
    # and overridden to start values by CurriculumManager on env init.
    observation_noise_type: str = "gaussian" # [gaussian, uniform, constant]
    observation_noise_params: dict = {
        "mean": 0.0,
        "std": obs_noise_end,
        "operation": "add",
    }

    ## ======================== Command ========================== ##
    commands: UniformJointPositionCommandCfg = UniformJointPositionCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 4.0),
    )

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = self.sim_dt
        self.sim.render_interval = self.decimation

        # Interactive Scene for DR : replicate_physics parameter shoule be 'False' for USD-level randomization
        self.scene.replicate_physics = False
        
        # Initial condition
        self.GOAT_cfg.spawn.articulation_props.fix_root_link = True
        self.GOAT_cfg.init_state.pos = (0.0, 0.0, 1.0)
        
        # disable wheel controller
        self.GOAT_cfg.actuators["wheel"].stiffness = {"wheel_L_Joint": 0.0, "wheel_R_Joint": 0.0}
        self.GOAT_cfg.actuators["wheel"].damping = {"wheel_L_Joint": 0.0, "wheel_R_Joint": 0.0}

        # event
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.add_link_mass = None
        self.events.rigid_body_mass_inertia = None
        self.events.robot_center_of_mass = None
        self.events.robot_leg_physics_material = None
        self.events.robot_wheel_physics_material = None
        self.events.robot_leg_actuator_gain.params["stiffness_distribution_params"] = (0.5, 1.3)
        self.events.robot_leg_actuator_gain.params["damping_distribution_params"] = (0.5, 1.3)

        self.events.reset_body.params["pose_range"] = {"yaw": (-3.14, 3.14)}
        self.events.reset_robot_joints = EventTerm(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.3, 0.3),
                "velocity_range": (0.0, 0.0) 
            }
        )


@configclass
class GOATTrackFixedPlayEnvCfg(GOATTrackFixedEnvCfg):
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
    
        # visualization
        self.viz_data = {
            "left_hip_torque (Nm)": 0.0,
            "right_hip_torque (Nm)": 0.0,
            "left_thigh_torque (Nm)": 0.0,
            "right_thigh_torque (Nm)": 0.0,
            "left_knee_torque (Nm)": 0.0,
            "right_knee_torque (Nm)": 0.0,
            "left_hip_velocity (deg/s)": 0.0,
            "right_hip_velocity (deg/s)": 0.0,
            "left_thigh_velocity (deg/s)": 0.0,
            "right_thigh_velocity (deg/s)": 0.0,
            "left_knee_velocity (deg/s)": 0.0,
            "right_knee_velocity (deg/s)": 0.0,
        }

        # disable randomization
        self.events.add_base_mass = None
        self.events.add_link_mass = None
        self.events.rigid_body_mass_inertia = None
        self.events.robot_leg_actuator_gain = None
        self.events.robot_wheel_actuator_gain = None
        self.events.robot_center_of_mass = None
        # disable noise
        self.observation_noise_type = None
        self.observation_noise_params = None

        # Target wheel-center markers 
        self.target_wheel_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/target_wheel",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )

        # plot
        # self.plotter = PNGSavePlotter