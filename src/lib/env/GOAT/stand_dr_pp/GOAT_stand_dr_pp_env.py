from __future__ import annotations

import torch
import copy
import numpy as np

from isaaclab.utils.math import normalize, quat_from_angle_axis, quat_from_euler_xyz
from isaaclab.terrains import TerrainImporter
from isaaclab.markers import VisualizationMarkers 
from isaaclab.sensors import ContactSensor
from isaacsim.core.utils import bounds
from isaacsim.core.utils import prims
from lib.env.GOAT.stand_dr_pp.GOAT_stand_dr_pp_env_cfg import GOATStandDRPPEnvCfg
from lib.env.GOAT.base.GOAT_base_env import GOATBaseEnv
from lib.controller.PD_controller import PD_Controller
from lib.controller.PI_controller import PI_Controller

csv_path = "initial_pose_data.csv"              # Path to csv file

class GOATStandDRPPEnv(GOATBaseEnv):
    cfg: GOATStandDRPPEnvCfg

    def __init__(self, cfg: GOATStandDRPPEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Config
        self.cfg = cfg
        self._contact_sensor =  self.scene.sensors["contact_sensor"]
        self.env_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # Torque controller initialization
        self.zero_joint_efforts = torch.zeros(self.num_envs, cfg.num_total_joints, device=self.device)
        self.leg_controller = PD_Controller(kp=self.cfg.joint_kp,
                                            kd=self.cfg.joint_kd,
                                            alpha=0.3,
                                            pos_margin_factor=self.cfg.pos_margin_factor,
                                            num_envs=self.num_envs,
                                            num_dof=self.cfg.leg_dof,
                                            num_leg=self.cfg.num_leg,
                                            device=self.device,
                                            dt=self.cfg.sim_dt,
                                            pos_limits=self._robot.data.joint_limits,
                                            torque_limits=self.torque_limits,
                                            default_joint_pos=self._robot.data.default_joint_pos)
        
        self.wheel_controller = PI_Controller(kp=self.cfg.wheel_kp,
                                              ki=self.cfg.wheel_ki,
                                              alpha=0.3,
                                              num_envs=self.num_envs,
                                              num_dof=1,                        # One wheel per legs
                                              num_leg=self.cfg.num_leg,
                                              device=self.device,
                                              dt=self.cfg.sim_dt,
                                              joint_vel_limits=self.joint_vel_limits,
                                              torque_limits=self.torque_limits)

        # Joint ids
        self.hip_joint_ids, _  = self._robot.find_joints(["hip_.*"])

        # Curriculum Info
        self.extras["Curriculum"] = {}
        self.extras["Curriculum"]["step_progress"] = 0
    
        # Index Mapping for external action scaling
        self.cfg.action_scale_factor["joint"][1] = self.joint_ids
        self.cfg.action_scale_factor["wheel"][1] = self.wheel_ids

        # Previous action
        self.previous_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # Plotting boolean
        debug_vis = self.num_envs <= 32
        self.set_debug_vis(debug_vis)
        self.is_plot = (self.num_envs == 1)
    

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "root_visualizer"):
                self.root_visualizer = VisualizationMarkers(self.cfg.root_visualizer_cfg)
            self.root_visualizer.set_visibility(True)
        else:
            if hasattr(self, "root_visualizer"):
                self.root_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self._robot.is_initialized:
            return
        
        root_pos = self.base_pos_w
        root_rot = self.base_rot_w

        self.root_visualizer.visualize(root_pos, root_rot)


    def _setup_scene(self):
        super()._setup_scene()

        self.terrain = TerrainImporter(self.cfg.terrain_importer_cfg)
        self.cfg.dome_light_cfg.spawn.func(self.cfg.dome_light_cfg.prim_path,
                                           self.cfg.dome_light_cfg.spawn)

        # Spawn contact sensor
        contact_sensor = ContactSensor(cfg=self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = contact_sensor

    def _reset_idx(self, env_ids: torch.Tensor):

        # NOTE: Initializations (joint states, material properties, etc..) are implemented by domain randomizer
        # Adjustment the Domain Randomization Parameters
        # self.set_curriculum()
        super()._reset_idx(env_ids)
        # Reset previous action observation
        self.previous_actions[env_ids] = torch.zeros_like(self.actions[env_ids], device=self.device)
        


        # Update planning state
        self._compute_intermediate_values()
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Preprocessor that helps applying policy's action to simulation

        Args:
            actions (torch.Tensor): Joint pos command (angle), wheel's velocity for each legs in shape (num_envs, 2, 4)
        """
        
        # Refine command
        self.actions = actions.clone()
        self.joint_pos_delta_cmd = self.actions[:, self.joint_ids]
        self.wheel_vel_cmd = self.actions[:, self.wheel_ids]
        
    def _apply_action(self):                    # Since it's inside the decimation loop, the low-level controller has to be located here
        # Current state
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel

        self.joint_torque_cmd = self.leg_controller.compute_torque(joint_pos=joint_pos,
                                                                   joint_vel=joint_vel,
                                                                   joint_pos_cmd=self.joint_pos_delta_cmd)
        
        self.wheel_torque_cmd = self.wheel_controller.compute_torque(joint_vel=joint_vel,
                                                                     joint_vel_cmd=self.wheel_vel_cmd)
        # Combine torque commands
        self.torque_cmd = torch.cat((self.joint_torque_cmd, self.wheel_torque_cmd), dim=1)
        
        # Load to sim buffer
        self._robot.set_joint_effort_target(self.torque_cmd)
        # self._robot.write_joint_state_to_sim(self._robot.data.default_joint_pos, self._robot.data.default_joint_vel)

    def _get_observations(self) -> torch.Tensor:
        """
        Get sensor data without curriculum Gaussian noise

        Returns:
            Observation space
        """
        # TODO: base의 높이를 알 수 있는 방법이 있는지 check
        observation = torch.cat((self.base_ang_acceleration,
                                 self.base_ang_vel,
                                 self.base_rot_w,
                                 self.joint_pos[:, self.joint_ids],
                                 self.joint_vel), dim=1)

        return observation
    
    def _get_states(self) -> torch.Tensor:
        """"
        Get State space using previleged information

        Returns
            State space
        """
        observation = torch.cat((self.base_ang_acceleration,
                                 self.base_ang_vel,
                                 self.gravity_vector,
                                 self.base_rot_w,
                                 self.joint_pos[:, self.joint_ids],
                                 self.joint_vel), dim=1)
        
        privileged_info = torch.cat((self.base_lin_vel,
                                     self.base_height,
                                     self.contact_force,
                                     self.friction_coefficient), dim=1)
        
        state = torch.cat((observation, privileged_info), dim=1)

        return state
    
    def _get_rewards(self) -> torch.Tensor:
        # ======================= Scheduler ======================= #
        current_time = self.episode_length_buf.float()
        # Target gravity in base frame (Upright state = [0, 0, -1])
        target_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        
        # Upright_rate (1.0: upright properly, -1.0: upside down)
        upright_rate = torch.sum(self.gravity_vector * target_gravity, dim=1)       # Dot product
        
        # boolean for success measure
        is_upright = upright_rate > (self.cfg.upright_threshold * torch.pi / 180)
        is_height_reached = torch.abs(self.base_height - self.cfg.target_height) < self.cfg.height_threshold
        
        # Velocity criteria (Only strict for balancing)
        lin_vel_norm = torch.norm(self.base_lin_vel, dim=1)                             # L2 norm 
        ang_vel_norm = torch.norm(self.base_ang_vel, dim=1)
        is_stable = (lin_vel_norm < 0.5) & (ang_vel_norm < 1.0)

        is_upright = is_upright.view(-1)
        is_height_reached = is_height_reached.view(-1)
        is_stable = is_stable.view(-1)
        
        # ======================= Reward ======================= #
        # Orientation Reward (Projected Gravity Alignment) [Highest Priority]
        upright_error = torch.sum(torch.square(self.gravity_vector - target_gravity), dim=1)
        r_upright = torch.exp(-upright_error / 0.5**2)                                       # Raidial Basis FUnction (RBF)

        # Base Height Reward
        height_error = torch.square(self.base_height - self.cfg.target_height).squeeze(-1)
        r_height = torch.exp(-height_error / 0.5**2)

        # Alive Reward
        r_alive = self.cfg.r_alive_weight * (current_time * self.cfg.decimation) / (self.cfg.max_episode_length)
 
        # Regularization Penalty
        p_lin_vel           = -torch.sum(torch.square(self.base_lin_vel), dim=1)
        p_ang_vel           = -torch.sum(torch.square(self.base_ang_vel), dim=1)
        p_joint_deviation   = -torch.sum(torch.square(self.joint_deviation[:, self.joint_ids]), dim=1) # wheel is not included
        # p_joint_deviation_hip = -torch.sum(torch.abs(self.joint_deviation[:, self.hip_joint_ids]), dim=-1)
        p_joint_limit       = -torch.sum(self.out_of_limits_joint[:, self.joint_ids], dim=1) # wheel is not included
        p_all_torque_limit  = -torch.sum(self.out_of_limits_torque, dim=1)
        p_all_torque        = -torch.sum(torch.square(self.applied_torque), dim=1)
        p_joint_velocity    = -torch.sum(torch.square(self.joint_vel), dim=1)
        p_action_rate       = -torch.sum(torch.square(self.actions - self.previous_actions), dim=1)
        p_terminated        = -self.reset_terminated.float()

        # Total Reward Summation
        total_reward = (
            self.cfg.r_upright_weight * r_upright                   +
            self.cfg.r_height_weight * r_height                     +
            self.cfg.r_alive_weight * r_alive                       +
            self.cfg.p_lin_vel_weight * p_lin_vel                   +
            self.cfg.p_ang_vel_weight * p_ang_vel                   +
            self.cfg.p_joint_deviation * p_joint_deviation          +
            self.cfg.p_joint_limit_weight * p_joint_limit           +
            self.cfg.p_all_torque_limit_weight * p_all_torque_limit +
            self.cfg.p_all_torque_weight * p_all_torque             +
            self.cfg.p_joint_velocity_weight * p_joint_velocity     +
            self.cfg.p_action_rate_weight * p_action_rate           +
            self.cfg.p_terminated_weight * p_terminated
        )

        self.extras["reward"] = {
            # ==========================================
            # Task Reward (+)
            # ==========================================
            "Task Reward / Upright"          : self.cfg.r_upright_weight * r_upright,
            "Task Reward / Height"           : self.cfg.r_height_weight * r_height,
            "Task Reward / Alive"            : self.cfg.r_alive_weight * r_alive,
            # ==========================================
            # Task Penalty (-)
            # ==========================================
            "Task Penalty / Lin_Vel"         : self.cfg.p_lin_vel_weight * p_lin_vel,
            "Task Penalty / Ang_Vel"         : self.cfg.p_ang_vel_weight * p_ang_vel,
            "Task Penalty / Joint_Deviation" : self.cfg.p_joint_deviation * p_joint_deviation, 
            "Task Penalty / Joint_Limit"     : self.cfg.p_joint_limit_weight * p_joint_limit,
            "Task Penalty / Torque_Limit"    : self.cfg.p_all_torque_limit_weight * p_all_torque_limit,
            "Task Penalty / Torque"          : self.cfg.p_all_torque_weight * p_all_torque,
            "Task Penalty / Joint_Vel"       : self.cfg.p_joint_velocity_weight * p_joint_velocity,
            "Task Penalty / Action_Rate"     : self.cfg.p_action_rate_weight * p_action_rate,
        }

        self.previous_actions = self.actions.clone()

        return total_reward
    
    def _get_dones(self):
        self._compute_intermediate_values() # planning state calculation
        
        # tilt_threshold_rad = torch.tensor(self.cfg.base_tilt_reset_condition, device=self.device) * torch.pi / 180.0
        # cos_threshold = torch.cos(tilt_threshold_rad)

        # target_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        # base_tilt = torch.sum(self.gravity_vector * target_gravity, dim=1)

        projected_gravity_x = self.gravity_vector[:, 0]
        projected_gravity_y = self.gravity_vector[:, 1]
        died_fall   = (self.base_height <= self.cfg.height_reset_condition).squeeze(-1)
        died_fall_2 = torch.logical_or(torch.abs(projected_gravity_x) >= self.cfg.termination_gravity,
                                       torch.abs(projected_gravity_y) >= self.cfg.termination_gravity)
        
        terminated = died_fall | died_fall_2
        truncated = self.episode_length_buf >= (self.cfg.max_episode_length - 1)

        return terminated, truncated

    def _compute_intermediate_values(self):
        # Observation data
        self.base_pos_w = self._robot.data.root_pos_w
        self.base_rot_w = self._robot.data.root_quat_w
        self.base_ang_acceleration = self._robot.data.body_acc_w[:, 0, 3:]
        self.base_ang_vel = self._robot.data.body_ang_vel_w[:, 0, :3]
        self.gravity_vector = self._robot.data.projected_gravity_b                     
        self.joint_pos = self._robot.data.joint_pos
        self.joint_vel = self._robot.data.joint_vel

        # State(privileged) data
        self.base_lin_vel = self._robot.data.root_vel_w[:, :3]
        self.base_height = self._robot.data.root_pos_w[:, 2].unsqueeze(-1)
        self.contact_force = self._contact_sensor.data.net_forces_w.view(self.num_envs, -1)
        material_property = self._robot.root_physx_view.get_material_properties()                   # device is "cpu" not "cuda" 
        self.friction_coefficient = torch.stack([material_property[:, 0, 0], material_property[:, 0, 1]], dim=-1).to(self.device)

        # Action regularization
        self.out_of_limits_joint = -(self.joint_pos - self._robot.data.soft_joint_pos_limits[:, :, 0]).clip(max=0.0) + \
                                    (self.joint_pos - self._robot.data.soft_joint_pos_limits[:, :, 1]).clip(min=0.0)
        self.out_of_limits_torque = (torch.abs(self._robot.data.applied_torque) - self.torque_limits * self.cfg.soft_torque_limit).clip(min=0.0)
        self.applied_torque = self._robot.data.applied_torque
        self.joint_deviation = self.joint_pos - self._robot.data.default_joint_pos
        

        # Extra Information data
        self.extras["Curriculum"]["step_progress"] = self.common_step_counter

    def _update_viz_data(self):
        joint_cmd_rad = self.joint_pos_delta_cmd
        wheel_vel_cmd_rad = self.wheel_vel_cmd

        joint_cmd_deg = joint_cmd_rad * (180 / torch.pi)
        wheel_vel_cmd_rpm = wheel_vel_cmd_rad * (30 / torch.pi)

        applied_target = torch.cat([joint_cmd_deg, wheel_vel_cmd_rpm], dim=-1)
        applied_torque = self._robot.data.applied_torque
        
        extras = copy.deepcopy(self.extras)
        extras["viz_data"]["left_hip_torque (Nm)"]    = applied_torque[:, 0]
        extras["viz_data"]["right_hip_torque (Nm)"]   = applied_torque[:, 1]
        extras["viz_data"]["left_thigh_torque (Nm)"]  = applied_torque[:, 2]
        extras["viz_data"]["right_thigh_torque (Nm)"] = applied_torque[:, 3]
        extras["viz_data"]["left_knee_torque (Nm)"]   = applied_torque[:, 4]
        extras["viz_data"]["right_knee_torque (Nm)"]  = applied_torque[:, 5]
        extras["viz_data"]["left_wheel_torque (Nm)"]  = applied_torque[:, 6]
        extras["viz_data"]["right_wheel_torque (Nm)"] = applied_torque[:, 7]

        extras["viz_data"]["left_hip_target (deg)"]    = applied_target[:, 0]
        extras["viz_data"]["right_hip_target (deg)"]   = applied_target[:, 1]
        extras["viz_data"]["left_thigh_target (deg)"]  = applied_target[:, 2]
        extras["viz_data"]["right_thigh_target (deg)"] = applied_target[:, 3]
        extras["viz_data"]["left_knee_target (deg)"]   = applied_target[:, 4]
        extras["viz_data"]["right_knee_target (deg)"]  = applied_target[:, 5]
        extras["viz_data"]["left_wheel_target (rpm)"]  = applied_target[:, 6]
        extras["viz_data"]["right_wheel_target (rpm)"] = applied_target[:, 7]

        return extras

    # ============== Auxilary Functions ================
    def set_curriculum(self):
        """
        Curriculum Learning for easy-to-hard task learning 
        """
        if self.extras["Curriculum"]["step_progress"] % 1e4 == 0:
            # Observation & Action Noises Control (Only Gaussian Noise)
            if self.cfg.action_noise_type:
                self.cfg.action_noise_params["mean"] = 0.0
                self.cfg.action_noise_params["std"] = 0.08

            if self.cfg.observation_noise_type:
                self.cfg.observation_noise_params["mean"] = 0.0
                self.cfg.observation_noise_params["std"] = 0.08

            # Environment Parameters Control
            self.event_manager.cfg.wheel_physics_material.params["static_friction_range"] = (0.5, 1.3) 
            self.event_manager.cfg.wheel_physics_material.params["dynamic_friction_range"] = (0.5, 1.3)