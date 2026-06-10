from __future__ import annotations

import torch
import copy

from isaaclab.terrains import TerrainImporter
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_apply
from lib.env.GOAT.track.GOAT_track_fixed_env_cfg import GOATTrackFixedEnvCfg, GOATTrackFixedPlayEnvCfg
from lib.env.GOAT.base.GOAT_base_env import GOATBaseEnv
from lib.env.GOAT.track.mdp.commander import UniformJointPositionCommand

class GOATTrackFixedEnv(GOATBaseEnv):
    cfg: GOATTrackFixedEnvCfg | GOATTrackFixedPlayEnvCfg

    def __init__(self, cfg: GOATTrackFixedEnvCfg | GOATTrackFixedPlayEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Config
        self.cfg = cfg
        self.env_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # Commands for target joint position reference (left sampled, right mirrored)
        self.commands = UniformJointPositionCommand(self.cfg.commands, self._robot, self.device)
        self.command_target = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)

        # Target wheel-center for visualization (left/right)
        self.target_wheel_pos_w = torch.zeros((self.num_envs, 2, 3), dtype=torch.float32, device=self.device)

        # Robot data
        self.base_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.base_rot_w = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.gravity_vector = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)                  
        self.joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)

        # Privileged data
        self.base_height = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)

        # Action regularization
        self.out_of_limits_velocity = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.out_of_limits_joint = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.out_of_limits_torque = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.applied_torque = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.joint_tracking_error = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.joint_acc = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        
        # Previous action
        self.previous_actions   = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # Plotting boolean
        debug_vis = self.num_envs <= 32
        self.set_debug_vis(debug_vis)
        self.is_plot = (self.num_envs == 1)

        # Contact sensor
        self.contact_base_link_id, _ = self.contact_sensors.find_bodies(["base_Link"])
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "root_visualizer"):
                self.root_visualizer = VisualizationMarkers(self.cfg.root_visualizer_cfg)
                self.root_visualizer.set_visibility(True)
            if not hasattr(self, "target_wheel_visualizer"):
                self.target_wheel_visualizer = VisualizationMarkers(self.cfg.target_wheel_visualizer_cfg)
                self.target_wheel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "root_visualizer"):
                self.root_visualizer.set_visibility(False)
            if hasattr(self, "target_wheel_visualizer"):
                self.target_wheel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self._robot.is_initialized:
            return

        root_pos = self.base_pos_w
        root_rot = self.base_rot_w
        self.root_visualizer.visualize(root_pos, root_rot)

        # Target wheel centers: FK of the sampled joint command (base frame) -> world frame
        p_left_b = self.commands.wheel_pos_target[:, 0].reshape(-1, 3)  # (E, 3)
        p_right_b = self.commands.wheel_pos_target[:, 1].reshape(-1, 3)  # (E, 3)
        p_left_w = root_pos + quat_apply(root_rot, p_left_b)
        p_right_w = root_pos + quat_apply(root_rot, p_right_b)
        self.target_wheel_pos_w[:, 0] = p_left_w
        self.target_wheel_pos_w[:, 1] = p_right_w
        # stack left/right into a single (2E, 3) marker translation buffer
        translations = self.target_wheel_pos_w.reshape(-1, 3)
        self.target_wheel_visualizer.visualize(translations=translations)
    
    def _setup_scene(self):
        super()._setup_scene()
        # Terrain
        self.terrain = TerrainImporter(self.cfg.terrain)
        self.cfg.dome_light_cfg.spawn.func(self.cfg.dome_light_cfg.prim_path,
                                           self.cfg.dome_light_cfg.spawn)
        # Commands cfg
        self.cfg.commands.num_envs = self.scene.num_envs
        self.cfg.commands.step_dt = self.step_dt
        # Collision filtering
        global_prim_paths = []
        if hasattr(self.cfg, "terrain") and hasattr(self.cfg.terrain, "prim_path"):
            global_prim_paths.append(self.cfg.terrain.prim_path)
        self.scene.filter_collisions(global_prim_paths=global_prim_paths)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.processed_actions = actions.clone()
        self.processed_actions[:, self.joint_ids] *= self.cfg.action_scale_factor["joint"][0]

    def _apply_action(self):    
        # Current state
        cmd_joint_pos = self._robot.data.default_joint_pos[:, self.joint_ids] + self.processed_actions[:, self.joint_ids]
        # Apply command
        self._robot.set_joint_position_target(cmd_joint_pos, joint_ids=self.joint_ids)
        # self._robot.write_joint_state_to_sim(position=self.command_target[:, self.joint_ids], velocity=torch.zeros_like(self.command_target[:, self.joint_ids]), joint_ids=self.joint_ids)

    def _get_observations(self) -> torch.Tensor:
        """
        Get sensor data without curriculum Gaussian noise

        Returns:
            Observation space
        """
        observation = torch.cat((
                                 self.joint_pos[:, self.joint_ids],                      # [E, 6]
                                 self.joint_vel[:, self.joint_ids],                      # [E, 6]
                                 self.previous_actions,                                  # [E, 6]
                                 self.command_target[:, self.joint_ids],                 # [E, 6]
                                ), dim=1)

        return observation
    
    def _get_rewards(self) -> torch.Tensor:
        # Command Tracking Reward (toward sampled target joint position)
        joint_tracking_error = torch.sum(torch.abs(self.joint_tracking_error[:, self.joint_ids]), dim=1) # wheel is not included
        r_joint_tracking = torch.exp(-joint_tracking_error / 0.7**2)

        # Regularization Penalty
        p_joint_limit       = -torch.sum(self.out_of_limits_joint[:, self.joint_ids], dim=1) # wheel is not included
        p_all_torque_limit  = -torch.sum(self.out_of_limits_torque[:, self.joint_ids], dim=1) # wheel is not included
        p_velocity_limit    = -torch.sum(self.out_of_limits_velocity[:, self.joint_ids], dim=1) # wheel is not included
        p_all_torque        = -torch.sum(torch.square(self.applied_torque[:, self.joint_ids]), dim=1) # wheel is not included
        p_joint_velocity    = -torch.sum(torch.square(self.joint_vel[:, self.joint_ids]), dim=1) # wheel is not included
        p_joint_accel       = -torch.sum(torch.square(self.joint_acc[:, self.joint_ids]), dim=1) # wheel is not included
        p_action_rate       = -torch.sum(torch.square((self.actions - self.previous_actions)), dim=1)

        # Total Reward Summation
        total_reward = (
            self.cfg.r_joint_tracking_weight * r_joint_tracking             +
            self.cfg.p_joint_limit_weight * p_joint_limit                   +
            self.cfg.p_all_torque_limit_weight * p_all_torque_limit         +
            self.cfg.p_joint_vel_limit_weight * p_velocity_limit            +
            self.cfg.p_all_torque_weight * p_all_torque                     +
            self.cfg.p_joint_velocity_weight * p_joint_velocity             +
            self.cfg.p_joint_accel_weight * p_joint_accel                   +
            self.cfg.p_action_rate_weight * p_action_rate                   
        )

        self.extras["reward"] = {
            # ==========================================
            # Task Reward (+)
            # ==========================================
            "Task Reward / Joint_Tracking" : self.cfg.r_joint_tracking_weight * r_joint_tracking,
            # ==========================================
            # Task Penalty (-)
            # ==========================================
            "Task Penalty / Joint_Limit"     : self.cfg.p_joint_limit_weight * p_joint_limit,
            "Task Penalty / Torque_Limit"    : self.cfg.p_all_torque_limit_weight * p_all_torque_limit,
            "Task Penalty / Velocity Limit"  : self.cfg.p_joint_vel_limit_weight * p_velocity_limit,
            "Task Penalty / Torque"          : self.cfg.p_all_torque_weight * p_all_torque,
            "Task Penalty / Joint_Vel"       : self.cfg.p_joint_velocity_weight * p_joint_velocity,
            "Task Penalty / Joint_Acc"       : self.cfg.p_joint_accel_weight * p_joint_accel,
            "Task Penalty / Action_Rate"     : self.cfg.p_action_rate_weight * p_action_rate,
        }

        self.previous_actions = self.actions.clone()

        return total_reward
    
    def _get_dones(self):
        self._compute_intermediate_values()

        critical_contact_forces = self.contact_sensors.data.net_forces_w[:, self.contact_base_link_id]
        
        terminated = torch.sum(torch.norm(critical_contact_forces, dim=-1), dim=-1) > 1.0
        truncated = self.episode_length_buf >= (self.cfg.max_episode_length - 1)

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        # Reset previous action observation
        self.previous_actions[env_ids] = torch.zeros_like(self.actions[env_ids], device=self.device)

        # Reset commands (resample target joint position reference)
        self.commands.reset(env_ids)

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
        # Action regularization
        self.out_of_limits_velocity[i] = (torch.abs(self.joint_vel[i]) - self.cfg.joint_vel_limit).clip(min=0.0)
        self.out_of_limits_joint[i]  = -(self.joint_pos[i] - self._robot.data.soft_joint_pos_limits[i, :, 0]).clip(max=0.0) + \
                                        (self.joint_pos[i] - self._robot.data.soft_joint_pos_limits[i, :, 1]).clip(min=0.0)
        self.out_of_limits_torque[i] = (torch.abs(self._robot.data.applied_torque[i]) - self.torque_limits[i] * self.cfg.soft_torque_limit).clip(min=0.0)
        self.applied_torque[i]       = self._robot.data.applied_torque[i]
        # Command tracking reference (target joint position)
        self.command_target[i]       = self.commands.target[i]
        self.joint_tracking_error[i] = self.joint_pos[i] - self.command_target[i]
        self.joint_acc[i] = self._robot.data.joint_acc[i]

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

        extras["viz_data"]["left_hip_velocity (deg/s)"]    = joint_velocity[:, 0]
        extras["viz_data"]["right_hip_velocity (deg/s)"]   = joint_velocity[:, 1]
        extras["viz_data"]["left_thigh_velocity (deg/s)"]  = joint_velocity[:, 2]
        extras["viz_data"]["right_thigh_velocity (deg/s)"] = joint_velocity[:, 3]
        extras["viz_data"]["left_knee_velocity (deg/s)"]   = joint_velocity[:, 4]
        extras["viz_data"]["right_knee_velocity (deg/s)"]  = joint_velocity[:, 5]

        return extras 