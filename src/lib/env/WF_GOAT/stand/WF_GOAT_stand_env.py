from __future__ import annotations

import torch
import copy

from isaaclab.utils.math import quat_apply, quat_from_euler_xyz
from isaaclab.terrains import TerrainImporter
from isaaclab.markers import VisualizationMarkers 
from isaaclab.assets import RigidObject
from lib.env.WF_GOAT.stand.WF_GOAT_stand_env_cfg import WFGOATStandEnvCfg, WFGOATStandPlayEnvCfg
from lib.env.WF_GOAT.base.WF_GOAT_base_env import WFGOATBaseEnv
from lib.env.WF_GOAT.track.mdp.commander import UniformVelocityHeightCommand

class WFGOATStandEnv(WFGOATBaseEnv):
    cfg: WFGOATStandEnvCfg | WFGOATStandPlayEnvCfg

    def __init__(self, cfg: WFGOATStandEnvCfg | WFGOATStandPlayEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Config
        self.cfg = cfg
        self._contact_sensor =  self.scene.sensors["contact_sensor"]
        self.env_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

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
        self.friction_coefficient = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        # Action regularization
        self.joint_deviation_lr = torch.zeros((self.num_envs, int(self._robot.num_joints / 2) - 1), dtype=torch.float32, device=self.device) # Exclude wheel
        self.out_of_limits_velocity = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.out_of_limits_joint = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.out_of_limits_torque = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.applied_torque = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.joint_deviation = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)
        self.joint_acc = torch.zeros((self.num_envs, self._robot.num_joints), dtype=torch.float32, device=self.device)

        # Index Mapping for external action scaling
        self.cfg.action_scale_factor["joint"][1] = self.joint_ids
        self.cfg.action_scale_factor["wheel"][1] = self.wheel_ids
        
        # Previous action
        self.previous_actions   = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # Geometry vector
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # Default config
        self.default_joint_pos = self._robot.data.default_joint_pos

        # Joint ids
        self.left_joint_ids, _ = self._robot.find_joints(".*_L_.*")
        self.right_joint_ids, _ = self._robot.find_joints(".*_R_.*")

        # Contact sensor
        self.contact_base_link_id, _ = self.contact_sensors.find_bodies(["^(?!wheel_).*$"]) # exclude wheel
        self.link_id, _ = self._robot.find_bodies(["^(?!wheel_).*$"])
        self.illegal_force = torch.zeros((self.num_envs, len(self.contact_base_link_id)), dtype=torch.float32, device=self.device)

        # Commands for reference generator
        self.commands = UniformVelocityHeightCommand(self.cfg.commands, self._robot, self.device)
        self.command_inputs_b   = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device) # [vx, vy, vz, h]

        # Jig release
        self.stand_up_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.jig_release = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_release = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Plotting boolean
        debug_vis = self.num_envs <= 32
        self.set_debug_vis(debug_vis)

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
        x_offset_w = quat_apply(self._robot.data.root_quat_w, 0.3 * self.forward_vec)
        goal_pos_w = self._robot.data.root_pos_w.clone() + x_offset_w
        goal_pos_w[:, 2] = self.commands.height_command.squeeze(-1)
        curr_pos_w = self._robot.data.root_pos_w.clone() + x_offset_w
        # Arrow: resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self.commands._resolve_xy_velocity_to_arrow(scale=self.goal_vel_visualizer.cfg.markers["arrow"].scale,
                                                                                              xy_velocity=self.commands.command_b[:, :2])
        vel_arrow_scale, vel_arrow_quat = self.commands._resolve_xy_velocity_to_arrow(scale=self.current_vel_visualizer.cfg.markers["arrow"].scale,
                                                                                      xy_velocity=self._robot.data.root_lin_vel_b[:, :2])

        if hasattr(self, "goal_vel_visualizer"):
            self.goal_vel_visualizer.visualize(goal_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        if hasattr(self, "current_vel_visualizer"):
            self.current_vel_visualizer.visualize(curr_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _setup_scene(self):
        super()._setup_scene()
        # Terrain
        self.terrain = TerrainImporter(self.cfg.terrain)
        self.cfg.dome_light_cfg.spawn.func(self.cfg.dome_light_cfg.prim_path,
                                           self.cfg.dome_light_cfg.spawn)
        # Jig object
        self._jig = RigidObject(self.cfg.jig)
        self.scene.rigid_objects["jig"] = self._jig
        # Commands cfg
        self.cfg.commands.num_envs = self.scene.num_envs
        self.cfg.commands.step_dt = self.step_dt
        # Collision filtering
        global_prim_paths = []
        if hasattr(self.cfg, "terrain") and hasattr(self.cfg.terrain, "prim_path"):
            global_prim_paths.append(self.cfg.terrain.prim_path)
        self.scene.filter_collisions(global_prim_paths=global_prim_paths)

    def _get_observations(self) -> torch.Tensor:
        """
        Get sensor data without curriculum Gaussian noise

        Returns:
            Observation space
        """
        observation = torch.cat((self.base_ang_vel,                                                              # [E, 3]
                                 self.gravity_vector,                                                            # [E, 3]
                                 self.command_inputs_b[:, 3:],                                                   # [E, 1]
                                 self.joint_pos[:, self.joint_ids] - self.default_joint_pos[:, self.joint_ids],  # [E, 4]
                                 self.joint_vel,                                                                 # [E, 6]
                                 self.previous_actions,                                                          # [E, 6]
                                ), dim=1) 

        return observation
    
    def _get_states(self) -> torch.Tensor:
        """"
        Get State space using previleged information

        Returns
            State space
        """
        observation = torch.cat((self.base_ang_vel,                                                               # [E, 3]
                                 self.gravity_vector,                                                             # [E, 3]
                                 self.command_inputs_b[:, 3:],                                                    # [E, 1]
                                 self.joint_pos[:, self.joint_ids] - self.default_joint_pos[:, self.joint_ids],   # [E, 4]
                                 self.joint_vel,                                                                  # [E, 6]
                                 self.previous_actions,                                                           # [E, 6]
                                 ), dim=1)                             
        
        privileged_info = torch.cat((self.base_lin_vel,                                      # [E, 3]
                                     self.base_height,                                       # [E, 1]
                                     self.friction_coefficient), dim=1)                      # [E, 2]
        
        state = torch.cat([observation, privileged_info], dim=-1)

        return state
    
    def _get_rewards(self) -> torch.Tensor:
        # Orientation Reward (Projected Gravity Alignment)
        upright_error = torch.sum(torch.square(self.gravity_vector[:, :2]), dim=1)
        r_upright = torch.exp(-upright_error / 0.05)

        # Height tracking Reward
        height_error = torch.reshape(torch.abs(self.base_height - self.command_inputs_b[:, 3:]), (-1,))
        r_height = torch.exp(-height_error / 0.05)

        # Regularization Penalty
        p_lin_vel            = -torch.norm(self.base_lin_vel[:, :3], dim=-1)
        p_ang_vel            = -torch.norm(self.base_ang_vel[:, :3], dim=-1)        
        p_illegal_contact    = -torch.sum(self.illegal_force, dim=1)
        
        p_velocity_limit     = -torch.sum(self.out_of_limits_velocity[:, self.joint_ids], dim=1)     # wheel is not included
        p_all_torque_limit   = -torch.sum(self.out_of_limits_torque, dim=1)
        p_all_torque         = -torch.sum(torch.square(self.applied_torque), dim=1)
        p_joint_velocity     = -torch.sum(torch.square(self.joint_vel[:, self.joint_ids]), dim=1)    # wheel is not included
        p_joint_accel        = -torch.sum(torch.square(self.joint_acc), dim=1)                       # [NOTE] wheel is included
        p_joint_deviation_lr = -torch.sum(torch.abs(self.joint_deviation_lr), dim=-1)
        p_action_rate        = -torch.sum(torch.abs((self.actions - self.previous_actions)), dim=1)
        p_terminated         = -self.reset_terminated.float()

        # Total Reward Summation
        total_reward = (
            self.cfg.r_upright_weight * r_upright                           +
            self.cfg.r_height_weight * r_height                             +
            self.cfg.p_ang_vel_weight * p_ang_vel                           +
            self.cfg.p_lin_vel_weight * p_lin_vel                           +
            self.cfg.p_illegal_contact_weight * p_illegal_contact           + 
            self.cfg.p_all_torque_limit_weight * p_all_torque_limit         +
            self.cfg.p_all_torque_weight * p_all_torque                     +
            self.cfg.p_joint_vel_limit_weight * p_velocity_limit            +
            self.cfg.p_joint_velocity_weight * p_joint_velocity             +
            self.cfg.p_joint_accel_weight * p_joint_accel                   +
            self.cfg.p_joint_deviation_lr_weight * p_joint_deviation_lr     +
            self.cfg.p_action_rate_weight * p_action_rate                   +
            self.cfg.p_terminated_weight * p_terminated
        )

        self.extras["reward"] = {
            # ==========================================
            # Task Reward (+)
            # ==========================================
            "Task Reward / Upright"             : r_upright,
            "Task Reward / Height"              : r_height,
            # ==========================================
            # Task Penalty (-)
            # ==========================================
            "Task Penalty / Lin_Vel"            : p_lin_vel,
            "Task Penalty / Ang_Vel"            : p_ang_vel,
            "Task Penalty / Contact"            : p_illegal_contact,
            "Task Penalty / Torque_Limit"       : p_all_torque_limit,
            "Task Penalty / Torque"             : p_all_torque,
            "Task Penalty / Vel_Limit"          : p_velocity_limit, 
            "Task Penalty / Joint_Vel"          : p_joint_velocity,
            "Task Penalty / Joint_Acc"          : p_joint_accel,
            "Task Penalty / Joint_Deviation_LR" : p_joint_deviation_lr,
            "Task Penalty / Action_Rate"        : p_action_rate,
        }

        self.previous_actions = self.actions.clone()

        return total_reward
    
    def _get_dones(self):
        self._compute_intermediate_values()

        base_fall = (self.base_height <= self.cfg.height_reset_condition).squeeze(-1)

        projected_gravity_x = self.gravity_vector[:, 0]
        projected_gravity_y = self.gravity_vector[:, 1]
        tilt_fall = torch.logical_or(torch.abs(projected_gravity_x) >= self.cfg.terminated_tilt,
                                     torch.abs(projected_gravity_y) >= self.cfg.terminated_tilt)
        exceed_joint_vel = torch.any(torch.abs(self.joint_vel[:, self.joint_ids]) > self.cfg.terminated_joint_vel_limit, dim=-1)
        exceed_base_vel_z = torch.abs(self.base_lin_vel[:, 2]) > self.cfg.terminated_lin_vel_limit_z
        
        terminated = base_fall | tilt_fall | exceed_joint_vel | exceed_base_vel_z
        truncated = self.episode_length_buf >= (self.cfg.max_episode_length - 1)

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        # Reset previous action observation
        self.previous_actions[env_ids] = torch.zeros_like(self.actions[env_ids], device=self.device)
        # Reset command
        self.commands.reset(env_ids)

        self.jig_release[env_ids] = False
        self.is_release[env_ids] = False
        self.stand_up_counter[env_ids] = 0
            
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
        # Action regularization
        self.joint_deviation_lr[i] = (self.joint_pos[i][:, self.left_joint_ids[:2]]) + (self.joint_pos[i][:, self.right_joint_ids[:2]])
        self.illegal_force[i] = torch.norm(self.contact_sensors.data.net_forces_w[i][:, self.contact_base_link_id], dim=-1)
        self.out_of_limits_joint[i]  = -(self.joint_pos[i] - self._robot.data.soft_joint_pos_limits[i, :, 0]).clip(max=0.0) + \
                                        (self.joint_pos[i] - self._robot.data.soft_joint_pos_limits[i, :, 1]).clip(min=0.0)
        self.out_of_limits_torque[i] = (torch.abs(self._robot.data.applied_torque[i]) - self.torque_limits[i] * self.cfg.soft_torque_limit).clip(min=0.0)
        self.out_of_limits_velocity[i] = (torch.abs(self.joint_vel[i]) - self.cfg.joint_vel_limit).clip(min=0.0)
        self.applied_torque[i]       = self._robot.data.applied_torque[i]
        self.joint_deviation[i]      = self.joint_pos[i] - self._robot.data.default_joint_pos[i]
        self.joint_acc[i] = self._robot.data.joint_acc[i]

        # The jig is only handled on the full-env path
        if env_ids is None:
            # Latch the release once the robot has held itself above the support height
            self.stand_up_counter = torch.where(self.base_height.squeeze(-1) > self.cfg.jig_release_height,
                                                self.stand_up_counter + 1,
                                                torch.zeros_like(self.stand_up_counter))
            self.jig_release |= self.stand_up_counter >= self.cfg.jig_release_hold_step
            # Drop the jig out of the world exactly once per episode
            release_ids = (self.jig_release & ~self.is_release).nonzero(as_tuple=False).flatten()
            if release_ids.numel() > 0:
                jig_pose = self._jig.data.root_state_w[release_ids, :7].clone()
                jig_pose[:, 2] = self.cfg.jig_release_depth
                self._jig.write_root_pose_to_sim(jig_pose, env_ids=release_ids)
                self.is_release[release_ids] = True

    def _update_viz_data(self):
        applied_torque = self._robot.data.applied_torque
        joint_velocity = torch.rad2deg(self._robot.data.joint_vel)
        
        extras = copy.deepcopy(self.extras)
        # extras["viz_data"]["left_hip_torque (Nm)"]    = applied_torque[:, 0]
        # extras["viz_data"]["right_hip_torque (Nm)"]   = applied_torque[:, 1]
        extras["viz_data"]["left_thigh_torque (Nm)"]  = applied_torque[:, 0]
        extras["viz_data"]["right_thigh_torque (Nm)"] = applied_torque[:, 1]
        extras["viz_data"]["left_knee_torque (Nm)"]   = applied_torque[:, 2]
        extras["viz_data"]["right_knee_torque (Nm)"]  = applied_torque[:, 3]
        extras["viz_data"]["left_wheel_torque (Nm)"]  = applied_torque[:, 4]
        extras["viz_data"]["right_wheel_torque (Nm)"] = applied_torque[:, 5]

        # extras["viz_data"]["left_hip_velocity (deg/s)"]    = joint_velocity[:, 0]
        # extras["viz_data"]["right_hip_velocity (deg/s)"]   = joint_velocity[:, 1]
        extras["viz_data"]["left_thigh_velocity (deg/s)"]  = joint_velocity[:, 0]
        extras["viz_data"]["right_thigh_velocity (deg/s)"] = joint_velocity[:, 1]
        extras["viz_data"]["left_knee_velocity (deg/s)"]   = joint_velocity[:, 2]
        extras["viz_data"]["right_knee_velocity (deg/s)"]  = joint_velocity[:, 3]
        extras["viz_data"]["left_wheel_velocity (deg/s)"]  = joint_velocity[:, 4]
        extras["viz_data"]["right_wheel_velocity (deg/s)"] = joint_velocity[:, 5]

        # extras["viz_data"]["left_hip_action"]      = self.actions[:, 0]
        # extras["viz_data"]["right_hip_action"]     = self.actions[:, 1]
        extras["viz_data"]["left_thigh_action"]    = self.actions[:, 0]
        extras["viz_data"]["right_thigh_action"]   = self.actions[:, 1]
        extras["viz_data"]["left_knee_action"]     = self.actions[:, 2]
        extras["viz_data"]["right_knee_action"]    = self.actions[:, 3]
        extras["viz_data"]["left_wheel_action"]    = self.actions[:, 4]
        extras["viz_data"]["right_wheel_action"]   = self.actions[:, 5]

        extras["viz_data"]["effective_contact_force (N)"] = torch.sum((self.illegal_force), dim=1)
        extras["viz_data"]["base_lin_x_velocity (m/s)"] = torch.abs(self.base_lin_vel[:, 0])
        extras["viz_data"]["base_lin_y_velocity (m/s)"] = torch.abs(self.base_lin_vel[:, 1])
        extras["viz_data"]["base_lin_z_velocity (m/s)"] = torch.abs(self.base_lin_vel[:, 2])
        extras["viz_data"]["base_ang_velocity (deg/s)"] = torch.rad2deg(torch.norm(self.base_ang_vel, dim=1))

        return extras 