from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


# -----------------------------------------------------------------------------
# Asset paths
# -----------------------------------------------------------------------------

_CURRENT_DIR = Path(__file__).resolve().parent

R1_ASSET = {
    "urdf_path": str(_CURRENT_DIR / "urdf" / "R1.urdf"),
    "usd_path": str(_CURRENT_DIR / "usd" / "R1.usd"),
    "usd_dir": str(_CURRENT_DIR / "usd"),
    "usd_filename": "R1.usd",
}

Path(R1_ASSET["usd_dir"]).mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# URDF -> USD conversion
# -----------------------------------------------------------------------------

urdf_cfg = sim_utils.UrdfConverterCfg(
    asset_path=R1_ASSET["urdf_path"],
    usd_dir=R1_ASSET["usd_dir"],
    usd_file_name=R1_ASSET["usd_filename"],
    fix_base=False,
    merge_fixed_joints=True,
    make_instanceable=True,
    force_usd_conversion=False,
    collision_from_visuals=False,
    self_collision=False,
    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=0.0,
            damping=0.0,
        ),
    ),
)

urdf_converter = sim_utils.UrdfConverter(cfg=urdf_cfg)


# -----------------------------------------------------------------------------
# Conversion check
# -----------------------------------------------------------------------------

converted_usd_path = Path(urdf_converter.usd_path).resolve()
expected_usd_path = Path(R1_ASSET["usd_path"]).resolve()

if not converted_usd_path.is_file():
    raise RuntimeError(
        f"R1 URDF -> USD conversion failed.\n"
        f"Expected USD file: {converted_usd_path}"
    )

if converted_usd_path == expected_usd_path:
    print(f"[R1] URDF conversion success: {converted_usd_path}")
else:
    print(
        "[R1] URDF conversion succeeded, but output path differs.\n"
        f"  converted: {converted_usd_path}\n"
        f"  expected : {expected_usd_path}"
    )


# -----------------------------------------------------------------------------
# R1 Articulation Configuration
# -----------------------------------------------------------------------------

R1_CFG: ArticulationCfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",

    spawn=sim_utils.UsdFileCfg(
        usd_path=str(converted_usd_path),

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
            enabled_self_collisions=False,
            fix_root_link=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),

    # -------------------------------------------------------------------------
    # Initial state
    # -------------------------------------------------------------------------
    init_state=ArticulationCfg.InitialStateCfg(
        # Spawn height
        pos=(0.0, 0.0, 0.73),

        joint_pos={
            # -----------------------------------------------------------------
            # Legs
            # -----------------------------------------------------------------
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.1745,
            ".*_knee_joint": 0.4363,
            ".*_ankle_pitch_joint": -0.2793,
            ".*_ankle_roll_joint": 0.0,
            # -----------------------------------------------------------------
            # Waist
            # -----------------------------------------------------------------
            "waist_.*_joint": 0.0,
            # -----------------------------------------------------------------
            # Arms
            # -----------------------------------------------------------------
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.2182,
            "right_shoulder_roll_joint": -0.2182,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 1.5010,
            ".*_wrist_roll_joint": 0.0,
            # -----------------------------------------------------------------
            # Head
            # -----------------------------------------------------------------
            "head_.*_joint": 0.0,
        },

        joint_vel={
            ".*": 0.0,
        },
    ),

    soft_joint_pos_limit_factor=0.9,

    actuators={
        # ---------------------------------------------------------------------
        # Legs: hip + knee
        # ---------------------------------------------------------------------
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],

            stiffness={
                ".*_hip_yaw_joint": 120.0,
                ".*_hip_roll_joint": 120.0,
                ".*_hip_pitch_joint": 160.0,
                ".*_knee_joint": 160.0,
            },

            damping={
                ".*_hip_yaw_joint": 4.0,
                ".*_hip_roll_joint": 4.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
            },
        ),

        # ---------------------------------------------------------------------
        # Ankles
        # ---------------------------------------------------------------------
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],

            stiffness={
                ".*_ankle_pitch_joint": 30.0,
                ".*_ankle_roll_joint": 30.0,
            },

            damping={
                ".*_ankle_pitch_joint": 2.0,
                ".*_ankle_roll_joint": 2.0,
            },
        ),

        # ---------------------------------------------------------------------
        # Waist
        # R1: roll + yaw only
        # ---------------------------------------------------------------------
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_roll_joint",
                "waist_yaw_joint",
            ],

            stiffness={
                "waist_roll_joint": 100.0,
                "waist_yaw_joint": 100.0,
            },

            damping={
                "waist_roll_joint": 4.0,
                "waist_yaw_joint": 4.0,
            },
        ),

        # ---------------------------------------------------------------------
        # Arms
        # R1: shoulder pitch/roll/yaw + elbow + wrist roll
        # ---------------------------------------------------------------------
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],

            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 30.0,
                ".*_elbow_joint": 30.0,
                ".*_wrist_roll_joint": 20.0,
            },

            damping={
                ".*_shoulder_pitch_joint": 4.0,
                ".*_shoulder_roll_joint": 4.0,
                ".*_shoulder_yaw_joint": 3.0,
                ".*_elbow_joint": 3.0,
                ".*_wrist_roll_joint": 2.0,
            },
        ),

        # ---------------------------------------------------------------------
        # Head
        # ---------------------------------------------------------------------
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_pitch_joint",
                "head_yaw_joint",
            ],

            stiffness={
                "head_pitch_joint": 20.0,
                "head_yaw_joint": 20.0,
            },

            damping={
                "head_pitch_joint": 2.0,
                "head_yaw_joint": 2.0,
            },
        ),
    },
)