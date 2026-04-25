
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from lib.env.G1.base.G1_base_env_cfg import G1BaseEnvCfg
from lib.utils.plot_utils import CapturabilityPlotter
from isaaclab.managers import SceneEntityCfg


@configclass
class G1SafeEnvCfg(G1BaseEnvCfg):
    ## ==================== Environment parameters ==================== ##
    episode_length_s = 3.0
    sim_dt = 1/200
    decimation = 4          

    ## ========== Multi Agent Setting =========== ##
    possible_agents = ["arm", "leg"]
    action_space = {"arm": 17, "leg": 12}                         
    observation_space = {"arm": 44, "leg": 34}                
    state_space = {"arm": 68, "leg": 68}
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
    w_alive:          float = 0.0

    w_ang_vel_xy:         float = 0.05
    w_joint_torque:       float = 1.0e-4
    w_joint_torque_limit: float = 1.0e-4
    w_joint_acc:          float = 1.0e-6
    w_joint_vel:          float = 5.0e-3

    w_limits:               float = 10.0
    w_deviation_hip:        float = 0.2
    w_deviation_torso:      float = 0.0
    w_deviation_arm:        float = 1.0
    w_action_rate:          float = 2.0
    w_prefer_collision:     float = 0.001
    w_not_prefer_collision: float = 0.01

    w_termination: float = 300

    soft_torque_limit: float = 0.8

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation
