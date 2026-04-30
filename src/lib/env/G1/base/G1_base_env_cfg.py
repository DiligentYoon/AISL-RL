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

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from lib.env.env_cfg import EnvCfg
from lib.domain_randomizer import randomizer
from lib.assets.robots.G1.G1_hand.G1_hand import G1CFG
from lib.assets.robots.G1.G1_hand.G1_hand_box_foot import G1_BOX_FOOT_CFG

@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_base = EventTerm(
        func=randomizer.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
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
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "roll": (-1.0, 1.0), "pitch": (-1.0, 1.0)}},
    )


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
    soft_torque_limit = 0.8

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=sim_dt, render_interval=decimation)

    # terrain
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

    # event
    events: EventCfg = EventCfg()

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # sensor
    contact_forces = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", 
                                history_length=3, 
                                track_air_time=True)

    # robot
    robot: ArticulationCfg = G1CFG.replace(prim_path="/World/envs/env_.*/Robot")


    # visualization
    torso_rotation_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Torso_rotation"
    )

    torso_rotation_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)