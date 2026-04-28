
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from lib.env.G1.base.G1_base_env_cfg import G1BaseEnvCfg
from lib.domain_randomizer import randomizer


@configclass
class G1SafeEnvCfg(G1BaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 3.0
    sim_dt = 1/200
    decimation = 4          

    ## ========== Multi Agent Setting =========== ##
    possible_agents = ["arm", "leg"]
    action_space = {"arm": 17, "leg": 12}                         
    observation_space = {"arm": 66, "leg": 51}                
    state_space = {"arm": 103, "leg": 103}
    ra_state_space = 11
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
    w_vel_limits:         float = 5.0
    w_joint_torque:       float = 1.0e-5
    w_joint_torque_limit: float = 5.0e-5
    w_joint_acc:          float = 5.0e-6
    w_joint_vel:          float = 5.0e-3

    w_deviation_hip:        float = 0.2
    w_deviation_torso:      float = 0.0
    w_deviation_arm:        float = 1.0
    w_action_rate:          float = 0.05
    w_prefer_collision:     float = 0.001
    w_not_prefer_collision: float = 0.01

    w_termination: float = 200

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation
        
        # Event
        self.events.push_robot = None
        self.events.reset_base = EventTerm(
            func=randomizer.reset_root_state_orientation_biased_uniform,
            mode="reset",
            params={
                "pose_range": {"roll": (-3.14/3, 3.14/3), "pitch": (-3.14/3, 3.14/3), "yaw": (-3.14, 3.14)},
                "velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5), "z": (-0.0, 0.0),
                                   "roll": (-2.0, 2.0), "pitch": (-2.0, 2.0), "yaw": (-1.0, 1.0)},
                "bias": 3.14/6
            }
        )

        self.events.reset_robot_joints = EventTerm(
            func=randomizer.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": {-0.3, 0.3},
                "velocity_range": {-2.0, 2.0}
            }
        )
