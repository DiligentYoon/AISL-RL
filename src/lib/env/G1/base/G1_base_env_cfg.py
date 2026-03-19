# Reference (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg

from lib.env.env_cfg import EnvCfg


# Robot asset paths
current_dir = os.path.dirname(__file__)
G1_hand_ASSET = {
    "urdf_path": os.path.join(current_dir, "../../../assets/G1/G1_hand/urdf/G1_hand.urdf"),
    "usd_path": os.path.join(current_dir, "../../../assets/G1/G1_hand/usd/G1_hand.usd"),
    "usd_place": os.path.join(current_dir, "../../../assets/G1/G1_hand/usd/"),
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


@configclass
class G1BaseEnvCfg(EnvCfg):
    # env
    episode_length_s = 20.0
    action_scale = 1.0
    sim_dt = 0.005
    decimation = 4
    action_space = 0
    observation_space = 0 
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=sim_dt, render_interval=decimation)
    terrain = TerrainImporterCfg(
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # sensor
    contact_forces = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", 
                                history_length=3, 
                                track_air_time=True)

    # robot
    robot: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
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
            fix_root_link=False,  # Configurable - can be set to True for fixed base
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.0, 0.0, 0.78),  # knee = 0.3rad
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
        "legs": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 88.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 32.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.5,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.5,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.03,
                ".*_knee_joint": 0.03,
            },
            saturation_effort=180.0,
        ),
        "feet": DCMotorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.2,
                ".*_ankle_roll_joint": 0.1,
            },
            effort_limit={
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
            },
            velocity_limit={
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint": 37.0,
            },
            armature=0.03,
            saturation_effort=80.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_.*_joint",
            ],
            effort_limit={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            velocity_limit={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness={
                "waist_yaw_joint": 5000.0,
                "waist_roll_joint": 5000.0,
                "waist_pitch_joint": 5000.0,
            },
            damping={
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
            },
            armature=0.001,
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
            velocity_limit=100,
            stiffness=3000.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.001,
                ".*_elbow_.*": 0.001,
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