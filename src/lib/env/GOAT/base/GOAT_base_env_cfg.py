# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import isaaclab.sim as sim_utils

from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from lib.env.env_cfg import EnvCfg
from lib.domain_randomizer import randomizer
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from lib.assets.actuators.actuator_cfg import GearDelayedPDActuatorCfg


# Robot asset paths
current_dir = os.path.dirname(__file__)
WF_GOAT_ASSET = {
    "urdf_path": os.path.join(current_dir, "../../../assets/robots/GOAT/WF_GOAT/urdf/WF_GOAT.urdf"),
    "usd_path": os.path.join(current_dir, "../../../assets/robots/GOAT/WF_GOAT/usd/WF_GOAT.usd"),
    "usd_place": os.path.join(current_dir, "../../../assets/robots/GOAT/WF_GOAT/usd/"),
    "usd_filename": "WF_GOAT.usd"
}

# URDF to USD conversion
urdf_cfg: sim_utils.UrdfConverterCfg = sim_utils.UrdfConverterCfg(
    root_link_name = "base_Link",
    asset_path = WF_GOAT_ASSET["urdf_path"],
    usd_dir = WF_GOAT_ASSET["usd_place"],
    usd_file_name = WF_GOAT_ASSET["usd_filename"],
    fix_base=False,
    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
    ),
)
urdf_converter = sim_utils.UrdfConverter(cfg = urdf_cfg)

# URDF conversion check
if urdf_converter.usd_path == WF_GOAT_ASSET["usd_path"]:
    print("urdf conversion success!")
else:
    print("urdf conversion failed!")

GOAT_Cfg: ArticulationCfg = ArticulationCfg(
    # prim_path="{ENV_REGEX_NS}/Robot",               # Path for Interactivescene's clone_environemnts
    prim_path="/World/envs/env_.*/Robot",             # Path for DirectRLEnv
    soft_joint_pos_limit_factor=0.9,
    spawn=sim_utils.UsdFileCfg(
        usd_path=urdf_converter.usd_path,
        scale=(1.0, 1.0, 1.0),
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
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=False
        ),
    ),

    # Link, Joint list in Isaac sim
    # Link = ['base_Link', 'hip_L_Link', 'hip_R_Link', 'thigh_L_Link', 'thigh_R_Link', 'calf_L_Link', 'calf_R_Link', 'wheel_L_Link', 'wheel_R_Link']
    # Joint = ['hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint', 'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint']
    
    # Initial Joint pos and vel
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.594),
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

    # Actuators cfg
    actuators={
        "hip": DelayedPDActuatorCfg(
            joint_names_expr=["hip_.*",],
            effort_limit=4.5,
            velocity_limit=15.0,
            min_delay=0,
            max_delay=4,
            stiffness={
                "hip_L_Joint": 5.0,
                "hip_R_Joint": 5.0,
            },                                      
            damping={
                "hip_L_Joint": 0.1,
                "hip_R_Joint": 0.1,
            },
            friction={
                "hip_L_Joint": 0.0,
                "hip_R_Joint": 0.0,
            },                    
            armature={
                "hip_L_Joint": 0.01,
                "hip_R_Joint": 0.01,
            }             
        ),

        "thigh": DelayedPDActuatorCfg(
            joint_names_expr=["thigh_.*",],
            effort_limit=4.5,
            velocity_limit=15.0,
            min_delay=0,
            max_delay=4,
            stiffness={
                "thigh_L_Joint": 5.0,
                "thigh_R_Joint": 5.0,
            },                                      
            damping={
                "thigh_L_Joint": 0.1,
                "thigh_R_Joint": 0.1,
            },
            friction={
                "thigh_L_Joint": 0.0,
                "thigh_R_Joint": 0.0,
            },                    
            armature={
                "thigh_L_Joint": 0.01,
                "thigh_R_Joint": 0.01,
            }             
        ),

        "knee": GearDelayedPDActuatorCfg(
            joint_names_expr=["knee_.*",],
            effort_limit=9.0,
            velocity_limit=7.5,
            gamma=1.0,
            gear_ratio=2.0,
            min_delay=0,
            max_delay=4,
            stiffness={
                "knee_L_Joint": 5.0,
                "knee_R_Joint": 5.0,
            },                                      
            damping={
                "knee_L_Joint": 0.1,
                "knee_R_Joint": 0.1,
            },
            friction={
                "knee_L_Joint": 0.0,
                "knee_R_Joint": 0.0,
            },                    
            armature={
                "knee_L_Joint": 0.01,
                "knee_R_Joint": 0.01,
            }             
        ),
        
        "wheel": DelayedPDActuatorCfg(
            joint_names_expr=["wheel_.*",],
            effort_limit=2.5,
            velocity_limit=15.0,
            min_delay=0,
            max_delay=4,
            stiffness={
                "wheel_L_Joint": 0.0,
                "wheel_R_Joint": 0.0,
            },                                      
            damping={
                "wheel_L_Joint": 0.02,
                "wheel_R_Joint": 0.02,
            },
            friction={
                "wheel_L_Joint": 0.0,
                "wheel_R_Joint": 0.0,
            },                    
            armature={
                "wheel_L_Joint": 0.01,
                "wheel_R_Joint": 0.01,
            }             
        ),
    }


    # actuators={
    #     "hip": DCMotorCfg(
    #         joint_names_expr=["hip_.*",],
    #         effort_limit=4.5,
    #         saturation_effort=4.5,
    #         velocity_limit=15.0,
    #         stiffness=0.0,                                      # Internal PD controller not used
    #         damping=0.0,                                        # Internal PD controller not used
    #         friction=0.12,                                      # Static friction coefficient
    #         dynamic_friction=5.646268e-02,                      # Dynamic friction coefficient 
    #         viscous_friction=3.190248e-01,                      # Viscous friction coefficient
    #     ),

    #     "thigh": DCMotorCfg(
    #         joint_names_expr=["thigh_.*",],
    #         effort_limit=4.5,
    #         saturation_effort=4.5,
    #         velocity_limit=15.0,
    #         stiffness=0.0,
    #         damping=0.0,
    #         friction=0.12,
    #         dynamic_friction=5.646268e-02,
    #         viscous_friction=3.190248e-01,
    #     ),

    #     "knee": DCMotorCfg(
    #         joint_names_expr=["knee_.*",],
    #         effort_limit=4.5,
    #         saturation_effort=4.5,
    #         velocity_limit=15.0,
    #         stiffness=0.0,
    #         damping=0.0,
    #         friction=0.12,
    #         dynamic_friction=5.373143e-02,
    #         viscous_friction=8.441387e-02,
    #     ),
        
    #     "wheel": DCMotorCfg(
    #         joint_names_expr=["wheel_.*"],
    #         effort_limit=2.5,
    #         saturation_effort=2.5,
    #         velocity_limit=15.0,
    #         stiffness=0.0,
    #         damping=0.0,
    #         friction=0.07,
    #         dynamic_friction=3.218126e-02,
    #         viscous_friction=1.715931e-02,
    #     )
    # }
)

@configclass
class EventCfg:
    """Configuration for domain-randomization events."""
    # startup
    add_base_mass = EventTerm(
        func=randomizer.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg":SceneEntityCfg("robot", body_names="base_Link"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        }
    )
    add_link_mass = EventTerm(
        func=randomizer.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_[LR]_Link"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    rigid_body_mass_inertia = EventTerm(
        func=randomizer.randomize_rigid_body_mass_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    robot_leg_physics_material = EventTerm(
        func=randomizer.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["^(?!wheel_).*$"]),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.7, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1000,
            "make_consistent": True,
        },
    )
    robot_wheel_physics_material = EventTerm(
      func=randomizer.randomize_rigid_body_material_shared,
      mode='startup',
      params={
          "asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"),
          "static_friction_range": (0.4, 1.2),
          "dynamic_friction_range": (0.7, 0.9),
          "restitution_range": (0.0, 0.0),
          "num_buckets": 1000,
          "make_consistent": True,
        },
    )
    robot_leg_actuator_gain = EventTerm(
        func=randomizer.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["^(?!wheel_).*$"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    robot_wheel_actuator_gain = EventTerm(
        func=randomizer.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="wheel_.*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    robot_center_of_mass = EventTerm(
        func=randomizer.randomize_rigid_body_coms,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_distribution_params": ((-0.075, 0.075), (-0.075, 0.075), (-0.075, 0.075)),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # reset
    reset_body = EventTerm(
        func=randomizer.reset_root_state_uniform,
        mode='reset',
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0)},
        },
    )
    reset_robot_joints = EventTerm(
        func=randomizer.reset_joints_by_offset_and_bias,
        mode="reset",
        params={
            "bias": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "position_range": (-0.03, 0.03),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=randomizer.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "roll": (-1.0, 1.0), "pitch": (-1.0, 1.0)}},
    )



@configclass
class GOATBaseEnvCfg(EnvCfg):
    # Env
    episode_length_s: int = 10       # Episode length in seconds
    sim_dt: float = 0.01             # Simulation(low-level controller) frequency
    decimation: int = 3              # Policy frequency = sim_freq / decimation
    action_space: int = 0            # Dimension of action space vector
    observation_space: int = 0       # Dimension of observation space vector
    state_space: int = 0             # Dimension of state space vector for privileged RL

    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        env_spacing=3.0,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0                 # Collision
        ),
        debug_vis=False
    )

    # Light
    dome_light_cfg = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # event
    events: EventCfg = EventCfg()

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # sensor
    contact_sensors = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*")

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

    # Visualization
    root_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Root")
    root_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    # GOAT cfg
    GOAT_cfg: ArticulationCfg = GOAT_Cfg

    # Torque limit
    torque_limits: list[float] = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]