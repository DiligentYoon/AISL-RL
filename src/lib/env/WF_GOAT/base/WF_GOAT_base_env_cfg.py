# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import isaaclab.sim as sim_utils

from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from lib.env.env_cfg import EnvCfg
from lib.domain_randomizer import randomizer
from lib.assets.robots.GOAT.WF_GOAT.WF_GOAT import GOAT_CFG

@configclass
class EventCfg:
    """Configuration for domain-randomization events."""
    # startup
    add_base_mass = EventTerm(
        func=randomizer.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg":SceneEntityCfg("robot", body_names="base_Link"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        }
    )
    add_link_mass = EventTerm(
        func=randomizer.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_[LR]_Link"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    robot_leg_physics_material = EventTerm(
        func=randomizer.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["^(?!wheel_).*$"]),
            "static_friction_range": (0.7, 1.1),
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
          "static_friction_range": (0.3, 0.8),
          "dynamic_friction_range": (0.3, 0.7),
          "restitution_range": (0.0, 0.0),
          "num_buckets": 1000,
          "make_consistent": True,
        },
    )
    
    robot_base_center_of_mass = EventTerm(
        func=randomizer.randomize_rigid_body_coms,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "com_distribution_params": ((-0.04, 0.02), (-0.02, 0.02), (-0.02, 0.02)),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    robot_link_center_of_mass = EventTerm(
        func=randomizer.randomize_rigid_body_coms,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=r"(?!wheel_).*_[LR]_Link"),
            "com_distribution_params": ((-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    
    # reset
    robot_hip_actuator_gain = EventTerm(
        func=randomizer.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="hip_.*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    robot_thigh_actuator_gain = EventTerm(
        func=randomizer.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="thigh_.*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    robot_knee_actuator_gain = EventTerm(
        func=randomizer.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="knee_.*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    robot_wheel_actuator_gain = EventTerm(
        func=randomizer.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="wheel_.*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    robot_hip_joint_friction = EventTerm(
        func=randomizer.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="hip_.*"),
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        }      
    )
    robot_thigh_joint_friction = EventTerm(
        func=randomizer.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="thigh_.*"),
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        }      
    )
    robot_knee_joint_friction = EventTerm(
        func=randomizer.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="knee_.*"),
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        }      
    )
    robot_wheel_joint_friction = EventTerm(
        func=randomizer.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="wheel_.*"),
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        }
    )

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
        func=randomizer.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
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
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )



@configclass
class WFGOATBaseEnvCfg(EnvCfg):
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
    GOAT_cfg: ArticulationCfg = GOAT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Torque limit
    torque_limits: list[float] = [4.5, 4.5, 4.5, 4.5, 9.0, 9.0, 2.5, 2.5]