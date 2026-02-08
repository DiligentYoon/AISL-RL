from __future__ import annotations

import torch

import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import subtract_frame_transforms, quat_apply, quat_apply_inverse

from lib.env.env import Env
from lib.env.humanoid.gnn.humanoid_gnn_env_cfg import HumanoidGNNEnvCfg

from lib.utils.graph_utils import build_node_info

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class HumanoidGNNEnv(Env):
    cfg: HumanoidGNNEnvCfg

    def __init__(self, cfg: HumanoidGNNEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)
        self._joint_dof_idx, _ = self.robot.find_joints(".*")
        self.num_joints = self.robot.num_joints
        self.num_body_links = self.robot.num_bodies - 2
        self.body_joint_mapping = self.create_dof_to_body_mapping()

        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.targets += self.scene.env_origins
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        debug_vis = self.num_envs <= 32
        self.set_debug_vis(debug_vis)

    def _set_debug_vis_impl(self, debug_vis: bool):

        if debug_vis:
            if not hasattr(self, "node_visualizer"):
                self.node_visualizer = VisualizationMarkers(self.cfg.node_marker_cfg)
            self.node_visualizer.set_visibility(True)

        else:
            if hasattr(self, "node_visualizer"):
                self.node_visualizer.set_visibility(False)
    
    def _debug_vis_callback(self, event):
        # update the visualization info
        root_pos = self.torso_position.unsqueeze(1)
        root_rot = self.torso_rotation.unsqueeze(1)
        link_pos_b = torch.cat(subtract_frame_transforms(
            root_pos.expand(-1, self.num_body_links, -1), root_rot.expand(-1, self.num_body_links, -1),
            self.robot.data.body_link_pos_w[:, 2:], self.robot.data.body_link_quat_w[:, 2:]), dim=-1)
        
        link_pos = root_pos.expand(-1, self.num_body_links, -1) + quat_apply(root_rot.expand(-1, self.num_body_links, -1), link_pos_b[:, :, :3])

        self.node_visualizer.visualize(translations=link_pos[:, self.body_joint_mapping, :].view(-1, 3))

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # Node Info for GNN
        self.cfg.node_info, self.cfg.num_nodes = build_node_info(robot_name="Humanoid", device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        forces = self.action_scale * self.joint_gears * self.actions
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        self.torso_position, self.torso_rotation = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        self.velocity, self.ang_velocity = self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel

        root_pos     = self.torso_position.unsqueeze(1).expand(-1, self.num_body_links, -1)
        root_rot     = self.torso_rotation.unsqueeze(1).expand(-1, self.num_body_links, -1)
        root_lin_vel = self.velocity.unsqueeze(1).expand(-1, self.num_body_links, -1)
        root_ang_vel = self.ang_velocity.unsqueeze(1).expand(-1, self.num_body_links, -1)

        link_pos = self.robot.data.body_link_pos_w[:, 2:]
        link_rot = self.robot.data.body_link_quat_w[:, 2:]
        link_lin_vel_w = self.robot.data.body_link_lin_vel_w[:, 2:]
        link_ang_vel_w = self.robot.data.body_link_ang_vel_w[:, 2:]

        link_pos_b = torch.cat(subtract_frame_transforms(root_pos, root_rot, 
                                                         link_pos, link_rot), dim=-1)

        rel_lin_vel_w = link_lin_vel_w - root_lin_vel
        rel_lin_vel_b = quat_apply_inverse(root_rot, rel_lin_vel_w)

        rel_ang_vel_w = link_ang_vel_w - root_ang_vel
        rel_ang_vel_b = quat_apply_inverse(root_rot, rel_ang_vel_w)
        
        self.link_pos_b = link_pos_b[:, self.body_joint_mapping, :]
        self.link_lin_vel_b = rel_lin_vel_b[:, self.body_joint_mapping, :]
        self.link_ang_vel_b = rel_ang_vel_b[:, self.body_joint_mapping, :]

        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.vel_loc,
            self.angvel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.dof_pos_scaled,
            self.prev_potentials,
            self.potentials,
        ) = compute_intermediate_values(
            self.targets,
            self.torso_position,
            self.torso_rotation,
            self.velocity,
            self.ang_velocity,
            self.dof_pos,
            self.robot.data.soft_joint_pos_limits[0, :, 0],
            self.robot.data.soft_joint_pos_limits[0, :, 1],
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.cfg.sim.dt,
        )

    def _get_observations(self) -> torch.Tensor:
        obs_body = torch.cat(
            (
                self.torso_position[:, 2].view(-1, 1),                # (N, 1)
                self.vel_loc,                                         # (N, 3)
                self.angvel_loc * self.cfg.angular_velocity_scale,    # (N, 3)
                normalize_angle(self.yaw).unsqueeze(-1),
                normalize_angle(self.roll).unsqueeze(-1),
                normalize_angle(self.angle_to_target).unsqueeze(-1),  # (N, 3)
                self.up_proj.unsqueeze(-1),                           # (N, 1)
                self.heading_proj.unsqueeze(-1)                       # (N, 1)
            ),
            dim=-1
        ).view(self.num_envs, 1, -1) # (N, 1, 12)

        obs_joint = torch.cat(
            (
                self.dof_pos_scaled.unsqueeze(-1),                    # (N, J, 1)
                self.dof_vel.unsqueeze(-1) * self.cfg.dof_vel_scale,  # (N, J, 1)
                self.actions.unsqueeze(-1),                           # (N, J, 1)
                self.link_pos_b[:, :, :3],                            # (N, J, 3)
                self.link_lin_vel_b,                                  # (N, J, 3)
                self.link_ang_vel_b,                                  # (N, J, 3)
            ),
            dim=-1
        ).view(self.num_envs, self.num_joints, -1) # (N, J, 12)

        obs = {
            "body": obs_body,
            "joint": obs_joint,
        }

        return obs
    
    # def _get_states(self) -> torch.Tensor:
    #     state = torch.cat(
    #         (
    #             self.torso_position[:, 2].view(-1, 1),
    #             self.vel_loc,
    #             self.angvel_loc * self.cfg.angular_velocity_scale,
    #             normalize_angle(self.yaw).unsqueeze(-1),
    #             normalize_angle(self.roll).unsqueeze(-1),
    #             normalize_angle(self.angle_to_target).unsqueeze(-1),
    #             self.up_proj.unsqueeze(-1),
    #             self.heading_proj.unsqueeze(-1),
    #             self.dof_pos_scaled,
    #             self.dof_vel * self.cfg.dof_vel_scale,
    #             self.actions,
    #         ),
    #         dim=-1,
    #     )

    #     return state

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.actions,
            self.reset_terminated,
            self.cfg.up_weight,
            self.cfg.heading_weight,
            self.heading_proj,
            self.up_proj,
            self.dof_vel,
            self.dof_pos_scaled,
            self.potentials,
            self.prev_potentials,
            self.cfg.actions_cost_scale,
            self.cfg.energy_cost_scale,
            self.cfg.dof_vel_scale,
            self.cfg.death_cost,
            self.cfg.alive_reward_scale,
            self.motor_effort_ratio,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self.torso_position[:, 2] < self.cfg.termination_height
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

        self._compute_intermediate_values()

    def create_dof_to_body_mapping(self):
        """
        """
        # Body_names: ['torso', 'head', 'lower_waist', ...]
        target_bodies = self.robot.body_names[2:]
        target_joints = self.robot.joint_names

        mapping_indices = []
        
        for dof in target_joints:
            best_match_idx = -1
            max_overlap = 0
            
            for i, body in enumerate(target_bodies):
                if body in dof:
                    if len(body) > max_overlap:
                        max_overlap = len(body)
                        best_match_idx = i
        
            if best_match_idx == -1:
                print(f"[Warning] Cannot find body match for DoF: {dof}. Using logic based on specific rules.")
                pass

            mapping_indices.append(best_match_idx)

        return mapping_indices

@torch.jit.script
def compute_rewards(
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    up_weight: float,
    heading_weight: float,
    heading_proj: torch.Tensor,
    up_proj: torch.Tensor,
    dof_vel: torch.Tensor,
    dof_pos_scaled: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    actions_cost_scale: float,
    energy_cost_scale: float,
    dof_vel_scale: float,
    death_cost: float,
    alive_reward_scale: float,
    motor_effort_ratio: torch.Tensor,
):
    heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
    heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, heading_weight * heading_proj / 0.8)

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(up_proj > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions**2, dim=-1)
    electricity_cost = torch.sum(
        torch.abs(actions * dof_vel * dof_vel_scale) * motor_effort_ratio.unsqueeze(0),
        dim=-1,
    )

    # dof at limit cost
    dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials

    total_reward = (
        progress_reward
        + alive_reward
        + up_reward
        + heading_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost
    )
    # adjust reward for fallen agents
    total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)
    return total_reward


@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )