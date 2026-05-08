
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm

from lib.env.G1.base.G1_base_env_cfg import G1BaseEnvCfg
from lib.domain_randomizer.commander import UniformVelocityCommandCfg
from lib.domain_randomizer import randomizer


@configclass
class G1SafeEnvCfg(G1BaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 3.0
    sim_dt = 1/200
    decimation = 4          

    ## ========== Nominal policy Setting =========== ##
    possible_agents = ["arm", "leg"]
    action_space = {"arm": 17, "leg": 12}                         
    observation_space = {"arm": 60, "leg": 45}                
    state_space = {"arm": 97, "leg": 97}
    ra_state_space = 40
    num_agents = 2
    action_scale_factor = {"arm": [0.5, ()], 
                           "leg": [0.5, ()]}
    
    ## ========== Safety policy setting ========== ##
    safe_action_space = action_space
    safe_observation_space = {"arm": 44, "leg": 34}
    safe_state_space = {"arm": 68, "leg": 68}

    ## ========== Single Agent Setting ========== ##  
    # action_space = 37                     
    # observation_space = 106                  
    # state_space = 0
    # num_agents = 1
    # action_scale_factor = 0.5

    ## ==================== Reward Shaping ==================== ##
    w_alive:              float = 0.0

    w_limits:             float = 10.0
    w_joint_torque:       float = 1.0e-5
    w_joint_torque_limit: float = 0.0
    w_joint_vel:          float = 5.0e-3

    w_deviation_arm:        float = 0.003
    w_deviation_leg:        float = 0.005
    w_action_rate:          float = 0.01
    
    w_max_collision:        float = 0.3
    w_prefer_collision:     float = 0.001
    w_not_prefer_collision: float = 0.01

    w_termination: float = 200

    # ===== Gait guidance ===== #
    time_period = 0.35
    z_c = 0.75

    # Commander
    commands: UniformVelocityCommandCfg = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 5.0),
        prob_standing_envs=0.0,
        prob_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-0.0, 0.0), heading=(0.0, 0.0)
        ),
    )

    ## ========== Risk-bucket reset ========== ##
    # Sampling weights over {low, mid, high} buckets produced by collect.py.
    # Replace the dict (do not mutate in place) to update at runtime.
    bucket_weights: dict[str, float] = {"low": 0.0, "mid": 0.0, "high": 1.0}

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation

        # Disable inherited reset / push events; dataset reset takes over.
        self.events.push_robot = None
        self.events.reset_base = None
        self.events.reset_robot_joints = None

        # Single dataset-driven reset event. dataset_dir is injected by the
        # train script before the first env.reset().
        self.events.reset_state_from_dataset = EventTerm(
            func=randomizer.reset_state_from_dataset,
            mode="reset",
            params={
                "dataset_dir": "",
                "bucket_weights": self.bucket_weights,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

@configclass
class G1SafePlayEnvCfg(G1SafeEnvCfg):
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