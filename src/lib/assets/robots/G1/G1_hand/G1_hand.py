import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# Robot asset paths
current_dir = os.path.dirname(__file__)
G1_hand_ASSET = {
    "urdf_path": os.path.join(current_dir, "urdf/G1_hand.urdf"),
    "usd_path": os.path.join(current_dir, "usd/G1_hand.usd"),
    "usd_place": os.path.join(current_dir, "usd/"),
    "usd_filename": "G1_hand.usd"
}

# URDF to USD conversion
urdf_cfg: sim_utils.UrdfConverterCfg = sim_utils.UrdfConverterCfg(
    root_link_name = "pelvis",
    asset_path = G1_hand_ASSET["urdf_path"],
    usd_dir = G1_hand_ASSET["usd_place"],
    usd_file_name = G1_hand_ASSET["usd_filename"],
    fix_base=False,
    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
    ),
)
urdf_converter = sim_utils.UrdfConverter(cfg = urdf_cfg)

# URDF conversion check
if urdf_converter.usd_path == G1_hand_ASSET["usd_path"]:
    print("urdf conversion success!")
else:
    print("urdf conversion failed!")



G1CFG: ArticulationCfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=urdf_converter.usd_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            fix_root_link=False,
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),    # knee = 0.8 rad
        joint_pos={
            ".*_hip_pitch_joint": -0.4,
            ".*_knee_joint": 0.8,
            ".*_ankle_pitch_joint": -0.4,
            
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=300,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_.*_joint",
            ],
            effort_limit={
                "waist_yaw_joint": 300.0,
                "waist_roll_joint": 300.0,
                "waist_pitch_joint": 300.0,
            },
            stiffness={
                "waist_yaw_joint": 200.0,
                "waist_roll_joint": 200.0,
                "waist_pitch_joint": 200.0,
            },
            damping={
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
            },
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit=300,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_wrist_.*_joint": 0.001,
            },
        ),
        # "hands": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         ".*_index_.*",
        #         ".*_middle_.*",
        #         ".*_pinky_.*",
        #         ".*_ring_.*",
        #         ".*_thumb_.*",
        #     ],
        #     effort_limit=300,
        #     velocity_limit=1.0,
        #     stiffness=10000.0,
        #     damping=200.0,
        #     armature=0.001,
        # ),
    },
)