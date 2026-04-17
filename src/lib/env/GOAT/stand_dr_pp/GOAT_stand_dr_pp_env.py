from __future__ import annotations

import torch
import copy
import numpy as np

from isaaclab.terrains import TerrainImporter
from isaaclab.markers import VisualizationMarkers 
from isaaclab.sensors import ContactSensor
from lib.env.GOAT.stand_dr_pp.GOAT_stand_dr_pp_env_cfg import GOATStandDRPPEnvCfg, GOATStandDRPPPlayEnvCfg
from lib.env.GOAT.base.GOAT_base_env import GOATBaseEnv
from lib.controller.PD_controller import PD_Controller
from lib.controller.PI_controller import PI_Controller
from lib.domain_randomizer.commander import UniformNonHolonomicCommand
from lib.domain_randomizer.randomizer import sample_rao_torque, sample_rfi_torque

class GOATStandDRPPEnv(GOATBaseEnv):
    cfg: GOATStandDRPPEnvCfg | GOATStandDRPPPlayEnvCfg

    def __init__(self, cfg: GOATStandDRPPEnvCfg | GOATStandDRPPPlayEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Config
        self.cfg = cfg
        self._contact_sensor =  self.scene.sensors["contact_sensor"]
        self.env_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # REFI
        if self.cfg.erfi_enabled:
            # RAO offset buffer
            self.rao_torque_offset = torch.zeros(
                self.num_envs, self.cfg.num_total_joints, device=self.device
            )
            # Front 50% = RFI, Next 50% = RAO
            n_rfi = self.num_envs // 2
            self.rfi_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.rfi_env_mask[:n_rfi] = True
            self.rao_env_mask = ~self.rfi_env_mask


        # Commands for reference generator
        self.commands = UniformNonHolonomicCommand(self.cfg.commands, self._robot, self.device)

        # Torque controller initialization
        self.leg_controller = PD_Controller(kp=self.cfg.joint_kp,
                                            kd=self.cfg.joint_kd,
                                            alpha=self.cfg.PD_LPF_gain,
                                            pos_margin_factor=self.cfg.pos_margin_factor,
                                            num_envs=self.num_envs,
                                            num_dof=self.cfg.leg_dof,
                                            num_leg=self.cfg.num_leg,
                                            min_delay=self.cfg.min_action_delay_steps,
                                            max_delay=self.cfg.max_action_delay_steps,
                                            device=self.device,
                                            dt=self.cfg.sim_dt,
                                            pos_limits=self._robot.data.joint_limits,
                                            torque_limits=self.torque_limits,
                                            default_joint_pos=self._robot.data.default_joint_pos)
        
        self.wheel_controller = PI_Controller(kp=self.cfg.wheel_kp,
                                              ki=self.cfg.wheel_ki,
                                              alpha=self.cfg.PI_LPF_gain,
                                              num_envs=self.num_envs,
                                              num_dof=1,                        
                                              num_leg=self.cfg.num_leg,
                                              min_delay=self.cfg.min_action_delay_steps,
                                              max_delay=self.cfg.max_action_delay_steps,
                                              device=self.device,
                                              dt=self.cfg.sim_dt,
                                              joint_vel_limits=self.joint_vel_limits,
                                              torque_limits=self.torque_limits)

        # Robot data
        self.base_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.base_rot_w = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.gravity_vector = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)                  
        self.joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.joint_vel_hist = torch.zeros((self.num_envs, self.cfg.vel_hist_length, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.hist_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Privileged data
        self.base_height = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.friction_coefficient = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        # Command
        self.command_inputs_b   = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.command_inputs_w   = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        # Action regularization
        self.out_of_limits_joint = -(self.joint_pos - self._robot.data.soft_joint_pos_limits[:, :, 0]).clip(max=0.0) + \
                                    (self.joint_pos - self._robot.data.soft_joint_pos_limits[:, :, 1]).clip(min=0.0)
        self.out_of_limits_torque = (torch.abs(self._robot.data.applied_torque) - self.torque_limits * self.cfg.soft_torque_limit).clip(min=0.0)
        self.applied_torque = self._robot.data.applied_torque
        self.joint_deviation = self.joint_pos - self._robot.data.default_joint_pos
        self.joint_acc = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)

        # Index Mapping for external action scaling
        self.cfg.action_scale_factor["joint"][1] = self.joint_ids
        self.cfg.action_scale_factor["wheel"][1] = self.wheel_ids
        self.action_scale_factor = torch.tensor(self.cfg.train_action_scale_factor, device=self.device).repeat((self.num_envs, 1))

        # Action Buffer for action delay
        self.action_buffer = torch.zeros(
            (self.num_envs, self.cfg.max_action_delay_steps + 1, self.cfg.action_space),
            device=self.device
        )
        
        # Delays for each environments
        self.delays_per_env = torch.zeros(
            self.num_envs, 
            dtype=torch.long, 
            device=self.device
        )
        
        # batch index
        self.batch_indices = torch.arange(self.num_envs, device=self.device)
        
        # Previous action
        self.previous_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.previous_joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)

        # Plotting boolean
        debug_vis = self.num_envs <= 32
        self.set_debug_vis(debug_vis)
        self.is_plot = (self.num_envs == 1)

        # Default config
        self.default_joint_pos = self._robot.data.default_joint_pos
    

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "root_visualizer"):
                self.root_visualizer = VisualizationMarkers(self.cfg.root_visualizer_cfg)
                self.root_visualizer.set_visibility(True)
            if not hasattr(self, "goal_vel_visualizer") and hasattr(self.cfg, "goal_vel_visualizer_cfg"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.goal_vel_visualizer.set_visibility(True)
            if not hasattr(self, "current_vel_visualizer") and hasattr(self.cfg, "current_vel_visualizer_cfg"):
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
                self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "root_visualizer"):
                self.root_visualizer.set_visibility(False)
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
            if hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self._robot.is_initialized:
            return

        # ============== Arrow ================ # 
        # Arrow: get marker location
        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # Arrow: resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self.commands._resolve_xy_velocity_to_arrow(scale=self.goal_vel_visualizer.cfg.markers["arrow"].scale,
                                                                                              xy_velocity=self.commands.command_b)
        vel_arrow_scale, vel_arrow_quat = self.commands._resolve_xy_velocity_to_arrow(scale=self.current_vel_visualizer.cfg.markers["arrow"].scale,
                                                                                      xy_velocity=self._robot.data.root_lin_vel_b[:, :2])
        root_pos = self.base_pos_w
        root_rot = self.base_rot_w

        if hasattr(self, "goal_vel_visualizer"):
            self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        if hasattr(self, "current_vel_visualizer"):   
            self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)
        self.root_visualizer.visualize(root_pos, root_rot)
    

    def _setup_scene(self):
        super()._setup_scene()

        self.terrain = TerrainImporter(self.cfg.terrain_importer_cfg)
        self.cfg.dome_light_cfg.spawn.func(self.cfg.dome_light_cfg.prim_path,
                                           self.cfg.dome_light_cfg.spawn)

        # Spawn contact sensor
        contact_sensor = ContactSensor(cfg=self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = contact_sensor
        # add commands cfg
        self.cfg.commands.num_envs = self.scene.num_envs
        self.cfg.commands.step_dt = self.step_dt

        # Collision filtering
        global_prim_paths = []
        if hasattr(self.cfg, "terrain") and hasattr(self.cfg.terrain, "prim_path"):
            global_prim_paths.append(self.cfg.terrain.prim_path)
        self.scene.filter_collisions(global_prim_paths=global_prim_paths)
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Preprocessor that helps applying policy's action to simulation

        Args:
            actions (torch.Tensor): Joint pos command (angle), wheel's velocity for each legs in shape (num_envs, 2, 4)
        """
        # Refine command
        # Pop
        # self.action_buffer = torch.roll(self.action_buffer, shifts=1, dims=1)
        
        # # Action update
        # self.action_buffer[:, 0, :] = actions.clone()
        # delayed_actions = self.action_buffer[self.batch_indices, self.delays_per_env, :]

        # self.joint_pos_delta_cmd = delayed_actions[:, self.joint_ids]
        # self.wheel_vel_cmd = delayed_actions[:, self.wheel_ids]

        self.actions = actions.clone()
        self.processed_actions = actions.clone() * self.action_scale_factor
        self.joint_pos_delta_cmd = self.processed_actions[:, self.joint_ids]
        self.wheel_vel_cmd = self.processed_actions[:, self.wheel_ids]
        
    def _apply_action(self):         
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
        
        if self.cfg.erfi_enabled:
            erfi_perturbation = torch.zeros_like(self.torque_cmd)
            # RFI Env : Random torque purterbation at each step
            erfi_perturbation[self.rfi_env_mask] = sample_rfi_torque(
                self.rfi_env_mask.sum(), self.cfg.num_total_joints,
                self.cfg.rfi_torque_limit, self.device
            )
            # RAO Env : Random constant torque offset
            erfi_perturbation[self.rao_env_mask] = self.rao_torque_offset[self.rao_env_mask]
            self.torque_cmd = self.torque_cmd + erfi_perturbation

        # Load to sim buffer
        self._robot.set_joint_effort_target(self.torque_cmd)

    def _get_observations(self) -> torch.Tensor:
        """
        Get sensor data without curriculum Gaussian noise

        Returns:
            Observation space
        """
        observation = torch.cat((self.base_ang_vel,                                      # [E, 3]
                                 self.base_rot_w,                                        # [E, 4]
                                 self.command_inputs_b,                                  # [E, 3]
                                 self.joint_pos[:, self.joint_ids],                      # [E, 6]
                                 self.joint_vel,                                         # [E, 8]
                                 self.previous_actions,                                  # [E, 8]
                                 self.joint_vel_hist.reshape(self.num_envs, -1),         # [E, 8*t]
                                ), dim=1) 

        return observation
    
    def _get_states(self) -> torch.Tensor:
        """"
        Get State space using previleged information

        Returns
            State space
        """
        observation = torch.cat((self.base_ang_vel,                                      # [E, 3]
                                 self.base_rot_w,                                        # [E, 4]
                                 self.command_inputs_b,                                  # [E, 3]
                                 self.joint_pos[:, self.joint_ids],                      # [E, 6]
                                 self.joint_vel,                                         # [E, 8]
                                 self.previous_actions,                                  # [E, 8]
                                 self.joint_vel_hist.reshape(self.num_envs, -1),         # [E, 8*t]
                                 ), dim=1)                             
        
        privileged_info = torch.cat((self.base_lin_vel,                                      # [E, 3]
                                     self.base_height,                                       # [E, 1]
                                     self.friction_coefficient), dim=1)                      # [E, 2]
        
        state = torch.cat([observation, privileged_info], dim=-1)

        return state
    
    def _get_rewards(self) -> torch.Tensor:
        # Orientation Reward (Projected Gravity Alignment)
        upright_error = torch.sum(torch.square(self.gravity_vector[:, :2]), dim=1)
        r_upright = torch.exp(-upright_error / 0.5**2)

        # Command Tracking Reward
        lin_vel_error = torch.sum(torch.square(self.command_inputs_b[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        r_lin_vel_tracking = torch.exp(-lin_vel_error / 0.75**2)

        ang_vel_error = torch.square(self.command_inputs_b[:, 2] - self.base_ang_vel[:, 2])
        r_ang_vel_tracking = torch.exp(-ang_vel_error / 0.5**2)

        # Regularization Penalty
        p_joint_deviation   = -torch.sum(torch.abs(self.joint_deviation[:, self.joint_ids]), dim=1) # wheel is not included
        p_ang_vel           = -torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) # Rolling & Pitching 
        p_joint_limit       = -torch.sum(self.out_of_limits_joint[:, self.joint_ids], dim=1) # wheel is not included
        p_all_torque_limit  = -torch.sum(self.out_of_limits_torque, dim=1)
        p_all_torque        = -torch.sum(torch.square(self.applied_torque), dim=1)
        p_joint_velocity    = -torch.sum(torch.square(self.joint_vel[:, self.joint_ids]), dim=1) # wheel is not included
        p_joint_accel       = -torch.sum(torch.square(self.joint_acc), dim=1) # NOTE: wheel is included
        p_action_rate       = -torch.sum(torch.square((self.actions - self.previous_actions)), dim=1)
        p_terminated        = -self.reset_terminated.float()

        # Total Reward Summation
        total_reward = (
            self.cfg.r_upright_weight * r_upright                           +
            self.cfg.r_lin_vel_tracking_weight * r_lin_vel_tracking         +
            self.cfg.r_ang_vel_tracking_weight * r_ang_vel_tracking         +
            self.cfg.p_joint_deviation_weight * p_joint_deviation           +
            self.cfg.p_ang_vel_weight * p_ang_vel                           +
            self.cfg.p_joint_limit_weight * p_joint_limit                   +
            self.cfg.p_all_torque_limit_weight * p_all_torque_limit         +
            self.cfg.p_all_torque_weight * p_all_torque                     +
            self.cfg.p_joint_velocity_weight * p_joint_velocity             +
            self.cfg.p_action_rate_weight * p_action_rate                   +
            self.cfg.p_terminated_weight * p_terminated
        )

        self.extras["reward"] = {
            # ==========================================
            # Task Reward (+)
            # ==========================================
            "Task Reward / Upright"             : self.cfg.r_upright_weight * r_upright,
            "Task Reward / Lin_Vel_Tracking"    : self.cfg.r_lin_vel_tracking_weight * r_lin_vel_tracking,
            "Task Reward / Ang_Vel_Tracking"    : self.cfg.r_ang_vel_tracking_weight * r_ang_vel_tracking,
            # ==========================================
            # Task Penalty (-)
            # ==========================================
            "Task Penalty / Joint_Deviation" : self.cfg.p_joint_deviation_weight * p_joint_deviation,
            "Task Penalty / Ang_Vel"         : self.cfg.p_ang_vel_weight * p_ang_vel,
            "Task Penalty / Joint_Limit"     : self.cfg.p_joint_limit_weight * p_joint_limit,
            "Task Penalty / Torque_Limit"    : self.cfg.p_all_torque_limit_weight * p_all_torque_limit,
            "Task Penalty / Torque"          : self.cfg.p_all_torque_weight * p_all_torque,
            "Task Penalty / Joint_Vel"       : self.cfg.p_joint_velocity_weight * p_joint_velocity,
            "Task Penalty / Joint_Acc"       : self.cfg.p_joint_accel_weight * p_joint_accel,
            "Task Penalty / Action_Rate"     : self.cfg.p_action_rate_weight * p_action_rate,
        }

        self.previous_actions = self.actions.clone()

        return total_reward
    
    def _get_dones(self):
        self._compute_intermediate_values()

        projected_gravity_x = self.gravity_vector[:, 0]
        projected_gravity_y = self.gravity_vector[:, 1]
        died_fall   = (self.base_height <= self.cfg.height_reset_condition).squeeze(-1)
        died_fall_2 = torch.logical_or(torch.abs(projected_gravity_x) >= self.cfg.termination_gravity,
                                       torch.abs(projected_gravity_y) >= self.cfg.termination_gravity)
        
        terminated = died_fall | died_fall_2
        truncated = self.episode_length_buf >= (self.cfg.max_episode_length - 1)

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):

        # NOTE: Initializations (joint states, material properties, etc..) are implemented by domain randomizer
        # Adjustment the Domain Randomization Parameters
        # self.set_curriculum()
        super()._reset_idx(env_ids)
        # Reset previous action observation
        self.hist_count[env_ids] = 0
        self.previous_actions[env_ids] = torch.zeros_like(self.actions[env_ids], device=self.device)
        self.joint_vel_hist[env_ids] = torch.zeros_like(self.joint_vel_hist[env_ids], device=self.device)
        
        # Reset controller
        self.leg_controller.reset(env_ids)
        self.wheel_controller.reset(env_ids)

        # Reset commands
        self.commands.reset(env_ids)
        
        # ERFI
        if self.cfg.erfi_enabled:
            rao_reset_ids = env_ids[self.rao_env_mask[env_ids]]
            if len(rao_reset_ids) > 0:
                self.rao_torque_offset[rao_reset_ids] = sample_rao_torque(
                    rao_reset_ids, self.cfg.num_total_joints,
                    self.cfg.rao_torque_limit, self.device
                )
            
        # Update planning state
        self._compute_intermediate_values(env_ids)

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        i = env_ids if env_ids is not None else self._robot._ALL_INDICES
        # Robot data
        self.base_pos_w[i] = self._robot.data.root_pos_w[i]
        self.base_rot_w[i] = self._robot.data.root_quat_w[i] # (w, x, y, z)
        self.base_lin_vel[i] = self._robot.data.root_lin_vel_b[i]
        self.base_ang_vel[i] = self._robot.data.root_ang_vel_b[i]
        self.gravity_vector[i] = self._robot.data.projected_gravity_b[i]                  
        self.joint_pos[i] = self._robot.data.joint_pos[i]
        self.joint_vel[i] = self._robot.data.joint_vel[i]
        # Privileged data
        self.base_height[i] = self._robot.data.root_pos_w[i, 2].unsqueeze(-1)
        material_property = self._robot.root_physx_view.get_material_properties().to(self.device)[i] # device is "cpu" not "cuda" 
        self.friction_coefficient[i] = torch.stack([material_property[:, -2, 0], material_property[:, -1, 1]], dim=-1) # Left, Right wheel
        # Information related to Commands Tracking
        self.command_inputs_b[i] = self.commands.command_b[i]
        self.command_inputs_w[i] = self.commands.command_w[i]
        # Action regularization
        self.out_of_limits_joint[i]  = -(self.joint_pos[i] - self._robot.data.soft_joint_pos_limits[i, :, 0]).clip(max=0.0) + \
                                        (self.joint_pos[i] - self._robot.data.soft_joint_pos_limits[i, :, 1]).clip(min=0.0)
        self.out_of_limits_torque[i] = (torch.abs(self._robot.data.applied_torque[i]) - self.torque_limits[i] * self.cfg.soft_torque_limit).clip(min=0.0)
        self.applied_torque[i]       = self._robot.data.applied_torque[i]
        self.joint_deviation[i]      = self.joint_pos[i] - self._robot.data.default_joint_pos[i]
        self.joint_acc[i] = self._robot.data.joint_acc[i]
        # History information (noisy)
        if env_ids is not None:
            self.joint_vel_hist[i, (self.hist_count[i] % self.cfg.vel_hist_length), :] = self._robot.data.joint_vel[i] # Reset env -> default value
        else:
            # NOTE: we assume the history values are assigned last in observation vector
            self.joint_vel_hist[i, (self.hist_count[i] % self.cfg.vel_hist_length), :] = self.obs_buf[i, -self.joint_vel_hist.shape[-1]:].clone() # Noisy signal at previous step
        # Update count
        self.hist_count[i] += 1

    def _update_viz_data(self):
        applied_torque = self._robot.data.applied_torque
        joint_velocity = torch.rad2deg(self._robot.data.joint_vel)
        
        extras = copy.deepcopy(self.extras)
        extras["viz_data"]["left_hip_torque (Nm)"]    = applied_torque[:, 0]
        extras["viz_data"]["right_hip_torque (Nm)"]   = applied_torque[:, 1]
        extras["viz_data"]["left_thigh_torque (Nm)"]  = applied_torque[:, 2]
        extras["viz_data"]["right_thigh_torque (Nm)"] = applied_torque[:, 3]
        extras["viz_data"]["left_knee_torque (Nm)"]   = applied_torque[:, 4]
        extras["viz_data"]["right_knee_torque (Nm)"]  = applied_torque[:, 5]
        extras["viz_data"]["left_wheel_torque (Nm)"]  = applied_torque[:, 6]
        extras["viz_data"]["right_wheel_torque (Nm)"] = applied_torque[:, 7]

        extras["viz_data"]["left_hip_velocity (deg/s)"]    = joint_velocity[:, 0]
        extras["viz_data"]["right_hip_velocity (deg/s)"]   = joint_velocity[:, 1]
        extras["viz_data"]["left_thigh_velocity (deg/s)"]  = joint_velocity[:, 2]
        extras["viz_data"]["right_thigh_velocity (deg/s)"] = joint_velocity[:, 3]
        extras["viz_data"]["left_knee_velocity (deg/s)"]   = joint_velocity[:, 4]
        extras["viz_data"]["right_knee_velocity (deg/s)"]  = joint_velocity[:, 5]
        extras["viz_data"]["left_wheel_velocity (deg/s)"]  = joint_velocity[:, 6]
        extras["viz_data"]["right_wheel_velocity (deg/s)"] = joint_velocity[:, 7]

        extras["viz_data"]["base_linear_velocity (m/s)"] = self.base_lin_vel[:, 0]
        extras["viz_data"]["command_velocity (m/s)"] = self.command_inputs_b[:, 0]
        extras["viz_data"]["command_angular_velocity (deg/s)"] = torch.rad2deg(self.command_inputs_b[:, 2])

        return extras 