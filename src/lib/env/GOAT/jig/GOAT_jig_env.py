from __future__ import annotations

import torch
import copy
import numpy as np

from isaaclab.terrains import TerrainImporter
from isaaclab.markers import VisualizationMarkers 
from isaaclab.sensors import ContactSensor
from isaaclab.assets import RigidObject
from lib.env.GOAT.jig.GOAT_jig_env_cfg import GOATJigEnvCfg, GOATJigPlayEnvCfg
from lib.env.GOAT.base.GOAT_base_env import GOATBaseEnv
from lib.domain_randomizer.randomizer import sample_rao_torque, sample_rfi_torque

class GOATJigEnv(GOATBaseEnv):
    cfg: GOATJigEnvCfg | GOATJigPlayEnvCfg

    def __init__(self, cfg: GOATJigEnvCfg | GOATJigPlayEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Config
        self.cfg = cfg
        self._contact_sensor =  self.scene.sensors["contact_sensor"]
        self.env_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # REFI
        self.erfi_torque = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        if self.cfg.erfi_enabled:
            # RAO offset buffer
            self.rao_torque_offset = torch.zeros(
                self.num_envs, self._robot.num_joints, device=self.device
            )
            # Front 50% = RFI, Next 50% = RAO
            n_rfi = self.num_envs // 2
            self.rfi_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.rfi_env_mask[:n_rfi] = True
            self.rao_env_mask = ~self.rfi_env_mask

        # Robot data
        self.base_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.base_rot_w = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.gravity_vector = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)                  
        self.joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.hist_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Privileged data
        self.base_height = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.friction_coefficient = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        # Command
        self.command_inputs_b   = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.command_inputs_w   = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        # Action regularization
        self.out_of_limits_joint = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.out_of_limits_torque = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.applied_torque = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.joint_deviation = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.joint_acc = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)

        # Index Mapping for external action scaling
        self.cfg.action_scale_factor["joint"][1] = self.joint_ids
        self.cfg.action_scale_factor["wheel"][1] = self.wheel_ids
        self.action_scale_factor = torch.tensor(self.cfg.train_action_scale_factor, device=self.device).repeat((self.num_envs, 1))
        
        # Previous action
        self.previous_actions   = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.previous_joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.joint_pos_history  = torch.zeros((self.num_envs, self.cfg.vel_hist_length, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.action_history     = torch.zeros((self.num_envs, self.cfg.vel_hist_length, self._robot.num_joints), dtype=torch.float32, device=self.device)

        # Plotting boolean
        debug_vis = self.num_envs <= 32
        self.set_debug_vis(debug_vis)
        self.is_plot = (self.num_envs == 1)

        # Default config
        self.default_joint_pos = self._robot.data.default_joint_pos

        # Contact sensor
        self.contact_base_link_id, _ = self.contact_sensors.find_bodies(["base_Link"])
    

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
        # Terrain
        self.terrain = TerrainImporter(self.cfg.terrain)
        self.cfg.dome_light_cfg.spawn.func(self.cfg.dome_light_cfg.prim_path,
                                           self.cfg.dome_light_cfg.spawn)
        # Jig object
        self._jig = RigidObject(self.cfg.jig)
        self.scene.rigid_objects["jig"] = self._jig

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
        self.actions = actions.clone()
        self.processed_actions = actions.clone() * self.action_scale_factor
        
    def _apply_action(self):         
        # Current state
        cmd_joint_pos = self._robot.data.default_joint_pos[:, self.joint_ids] + self.processed_actions[:, self.joint_ids]
        cmd_wheel_vel = self.processed_actions[:, self.wheel_ids]
        
        # Apply command
        self._robot.set_joint_position_target(cmd_joint_pos, joint_ids=self.joint_ids)
        self._robot.set_joint_velocity_target(cmd_wheel_vel, joint_ids=self.wheel_ids)

        # Combine torque perturbation
        if self.cfg.erfi_enabled:
            erfi_perturbation = torch.zeros_like(self.erfi_torque)
            # RFI Env : Random torque purterbation at each step
            erfi_perturbation[self.rfi_env_mask] = sample_rfi_torque(
                self.rfi_env_mask.sum(), self._robot.num_joints,
                self.cfg.rfi_torque_limit, self.device
            )
            # RAO Env : Random constant torque offset
            erfi_perturbation[self.rao_env_mask] = self.rao_torque_offset[self.rao_env_mask]
            self.erfi_torque = erfi_perturbation
            # Load to sim buffer
            self._robot.set_joint_effort_target(self.erfi_torque)

    def _get_observations(self) -> torch.Tensor:
        """
        Get sensor data without curriculum Gaussian noise

        Returns:
            Observation space
        """
        observation = torch.cat((self.base_ang_vel,                                      # [E, 3]
                                 self.base_rot_w,                                        # [E, 4]
                                 self.joint_pos[:, self.joint_ids],                      # [E, 6]
                                 self.joint_vel,                                         # [E, 8]
                                 self.previous_actions,                                  # [E, 8]
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
                                 self.joint_pos[:, self.joint_ids],                      # [E, 6]
                                 self.joint_vel,                                         # [E, 8]
                                 self.previous_actions,                                  # [E, 8]
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
        
        # Joint tracking Reward
        joint_deviation   = torch.sum(torch.abs(self.joint_deviation[:, self.joint_ids]), dim=1) # wheel is not included
        r_joint_deviation = torch.exp(-joint_deviation / 0.5**2)

        # Regularization Penalty
        p_ang_vel           = -torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) # Rolling & Pitching 
        p_joint_limit       = -torch.sum(self.out_of_limits_joint[:, self.joint_ids], dim=1) # wheel is not included
        p_all_torque_limit  = -torch.sum(self.out_of_limits_torque, dim=1)
        p_all_torque        = -torch.sum(torch.square(self.applied_torque), dim=1)
        p_joint_velocity    = -torch.sum(torch.square(self.joint_vel[:, self.joint_ids]), dim=1) # wheel is not included
        p_wheel_velocity    = -torch.sum(torch.square(self.joint_vel[:, self.wheel_ids]), dim=1) 
        p_joint_accel       = -torch.sum(torch.square(self.joint_acc), dim=1) # NOTE: wheel is included
        p_action_rate       = -torch.sum(torch.square((self.actions - self.previous_actions)), dim=1)
        p_terminated        = -self.reset_terminated.float()

        # Total Reward Summation
        total_reward = (
            self.cfg.r_upright_weight * r_upright                           +
            self.cfg.r_joint_deviation_weight * r_joint_deviation           +
            self.cfg.p_ang_vel_weight * p_ang_vel                           +
            self.cfg.p_joint_limit_weight * p_joint_limit                   +
            self.cfg.p_all_torque_limit_weight * p_all_torque_limit         +
            self.cfg.p_all_torque_weight * p_all_torque                     +
            self.cfg.p_joint_velocity_weight * p_joint_velocity             +
            self.cfg.p_wheel_velocity_weight * p_wheel_velocity             +
            self.cfg.p_joint_accel_weight * p_joint_accel                   +
            self.cfg.p_action_rate_weight * p_action_rate                   +
            self.cfg.p_terminated_weight * p_terminated
        )

        self.extras["reward"] = {
            # ==========================================
            # Task Reward (+)
            # ==========================================
            "Task Reward / Upright"             : self.cfg.r_upright_weight * r_upright,
            "Task Reward / Joint_Deviation"     : self.cfg.r_joint_deviation_weight * r_joint_deviation,
            # ==========================================
            # Task Penalty (-)
            # ==========================================
            "Task Penalty / Ang_Vel"         : self.cfg.p_ang_vel_weight * p_ang_vel,
            "Task Penalty / Joint_Limit"     : self.cfg.p_joint_limit_weight * p_joint_limit,
            "Task Penalty / Torque_Limit"    : self.cfg.p_all_torque_limit_weight * p_all_torque_limit,
            "Task Penalty / Torque"          : self.cfg.p_all_torque_weight * p_all_torque,
            "Task Penalty / Joint_Vel"       : self.cfg.p_joint_velocity_weight * p_joint_velocity,
            "Task Penalty / Wheel_Vel"       : self.cfg.p_wheel_velocity_weight * p_wheel_velocity,
            "Task Penalty / Joint_Acc"       : self.cfg.p_joint_accel_weight * p_joint_accel,
            "Task Penalty / Action_Rate"     : self.cfg.p_action_rate_weight * p_action_rate,
        }

        self.previous_actions = self.actions.clone()

        return total_reward
    
    def _get_dones(self):
        self._compute_intermediate_values()

        critical_contact_forces = self.contact_sensors.data.net_forces_w[:, self.contact_base_link_id]
        illegal_contact = torch.sum(torch.norm(critical_contact_forces, dim=-1), dim=-1) > 1.0

        base_fall = (self.base_height <= self.cfg.height_reset_condition).squeeze(-1)
        
        terminated = base_fall
        truncated = self.episode_length_buf >= (self.cfg.max_episode_length - 1)

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        # Reset previous action observation
        self.hist_count[env_ids] = 0
        self.previous_actions[env_ids] = torch.zeros_like(self.actions[env_ids], device=self.device)
        self.action_history[env_ids] = torch.zeros_like(self.action_history[env_ids], device=self.device)
        self.joint_pos_history[env_ids] = torch.zeros_like(self.joint_pos_history[env_ids], device=self.device)
        
        # ERFI
        if self.cfg.erfi_enabled:
            rao_reset_ids = env_ids[self.rao_env_mask[env_ids]]
            if len(rao_reset_ids) > 0:
                self.rao_torque_offset[rao_reset_ids] = sample_rao_torque(
                    rao_reset_ids, self._robot.num_joints,
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
        # Action regularization
        self.out_of_limits_joint[i]  = -(self.joint_pos[i] - self._robot.data.soft_joint_pos_limits[i, :, 0]).clip(max=0.0) + \
                                        (self.joint_pos[i] - self._robot.data.soft_joint_pos_limits[i, :, 1]).clip(min=0.0)
        self.out_of_limits_torque[i] = (torch.abs(self._robot.data.applied_torque[i]) - self.torque_limits[i] * self.cfg.soft_torque_limit).clip(min=0.0)
        self.applied_torque[i]       = self._robot.data.applied_torque[i]
        self.joint_deviation[i]      = self.joint_pos[i] - self._robot.data.default_joint_pos[i]
        self.joint_acc[i] = self._robot.data.joint_acc[i]
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

        return extras 