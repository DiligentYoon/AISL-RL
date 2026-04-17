import torch
import math
import numpy as np

class PD_Controller():
    """
    Low level model-free error feedback PD torque controller

    τ = Kp*e + Kd*ė
    """
    
    def __init__(self, 
                 kp, kd, alpha: float, pos_margin_factor: float, 
                 num_envs: int, num_dof: int, num_leg: int,
                 min_delay: int, max_delay: int,
                 device: str, dt: float, 
                 pos_limits: float, torque_limits: float, default_joint_pos: float):
        """
        Controller initialization for position control

        Args:
            kp (float or torch.Tensor): Proportional gain. float value or (1, num_dof) size tensor.
            kd (float or torch.Tensor): Derivative gain. float value or (1, num_dof) size tensor.
            alpha (float): Low Pass Filter(LPF) gain.  α * new + (1 - α) * old
            pos_margin_factor: relaxation of joint pos limitation for gravity compensation (> 1).
            num_envs (int): Number of pararell training environments.
            num_dof (int): controllable dof per leg.
            num_leg (int): Number of legs
            min_delay (int): Minimum amount for action delay
            max_delay (int): Maximum amount for action delay
            device (str): "cuda:0" or "cpu".
            dt (float): Simulation time-step.
            pos_limits (float): joint pos limit.
            torque_limits (float): joint torque limit.
            default_joint_pos (float): default joint configuration for residual control.
        """

        self.device = device
        self.num_envs = num_envs
        self.num_dof = num_dof
        self.num_leg = num_leg
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.dt = dt
        self.alpha = alpha
        self.pos_margin_factor = pos_margin_factor
        self.joint_pos_limits = pos_limits
        self.joint_torque_limits = torque_limits
        self.default_joint_pos = default_joint_pos
        self.old_torque = torch.zeros(self.num_envs, num_dof * num_leg, device=self.device)

        # Robot dof
        leg_dof = self.num_dof                          # hip, thigh, knee joints
        self.num_joint = leg_dof * self.num_leg

        # Action buffer
        self.action_buffer = torch.zeros(
            (self.num_envs, self.max_delay + 1, self.num_joint), 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Delay level of each environment
        self.env_delays = torch.zeros(
            self.num_envs, 
            dtype=torch.long, 
            device=self.device
        )
        
        # Batch index
        self.batch_indices = torch.arange(self.num_envs, device=self.device)

        # Isaac sim's Joint order: ['hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint', 'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint']
        self.left_leg_indices = torch.tensor([0, 2, 4], device=self.device, dtype=torch.long)
        self.right_leg_indices = torch.tensor([1, 3, 5], device=self.device, dtype=torch.long)

        # kp gain
        if isinstance(kp, float):
            self.kp = torch.full((num_envs, num_dof), kp, device=device)
        elif isinstance(kp, torch.Tensor):
            if kp.shape != (1, num_dof):
                raise ValueError(f"kp tensor must have shape (1, {num_dof}), but got {kp.shape}")
            self.kp = kp.to(device).expand(num_envs, -1) # Expand as num_envs
        else:
            raise TypeError("kp must be a float or a torch.Tensor of shape (1, num_dof)")

        # kd gain
        if isinstance(kd, float):
            self.kd = torch.full((num_envs, num_dof), kd, device=device)
        elif isinstance(kd, torch.Tensor):
            if kd.shape != (1, num_dof):
                raise ValueError(f"kd tensor must have shape (1, {num_dof}), but got {kd.shape}")
            self.kd = kd.to(device).expand(num_envs, -1) # Expand as num_envs
        else:
            raise TypeError("kd must be a float or a torch.Tensor of shape (1, num_dof)")

    def compute_torque(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        joint_pos_cmd: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute joint input torque using PD controller.

        Args:
            joint_pos (torch.Tensor): Current joint position [rad].
            joint_vel (torch.Tensor): Current joint velocity [rad/s].
            joint_pos_cmd (torch.Tensor): Reference joint pos(angle) [num_env, num_leg, num_joint].
            joint_pos_limits (torch.Tensor): Joint position limits [num_env, num_joint, (low, high)].
            torque_limits (torch.Tensor): Joint torque limits [num_env, num_joint].

        Returns:
            torch.Tensor: Joint torque.
        """
        # Pop queue
        self.action_buffer = torch.roll(self.action_buffer, shifts=1, dims=1)
        
        # Push queue
        self.action_buffer[:, 0, :] = joint_pos_cmd

        # Delayed command
        delayed_actions = self.action_buffer[self.batch_indices, self.env_delays, :]
        
        # Joint pos limit with relaxation
        joint_pos_limits = self.pos_margin_factor * self.joint_pos_limits

        # --- Left Leg slicing ---
        joint_pos_left = torch.index_select(joint_pos, 1, self.left_leg_indices)
        joint_vel_left = torch.index_select(joint_vel, 1, self.left_leg_indices)
        joint_pos_cmd_left = torch.index_select(delayed_actions, 1, self.left_leg_indices)
        joint_default_pos_left = torch.index_select(self.default_joint_pos, 1, self.left_leg_indices)
        joint_limits_left = torch.index_select(joint_pos_limits, 1, self.left_leg_indices)

        # --- Right Leg slicing ---
        joint_pos_right = torch.index_select(joint_pos, 1, self.right_leg_indices)
        joint_vel_right = torch.index_select(joint_vel, 1, self.right_leg_indices)
        joint_pos_cmd_right = torch.index_select(delayed_actions, 1, self.right_leg_indices)
        joint_default_pos_right = torch.index_select(self.default_joint_pos, 1, self.right_leg_indices)
        joint_limits_right = torch.index_select(joint_pos_limits, 1, self.right_leg_indices)      

        # Left foot PD control
        joint_pos_cmd_left = torch.clamp(joint_default_pos_left + joint_pos_cmd_left, joint_limits_left[:, :, 0], joint_limits_left[:, :, 1])
        joint_pos_left_error = joint_pos_cmd_left - joint_pos_left
        joint_vel_left_error = -joint_vel_left                                            # reference joint velocity = 0
        torque_left = self.kp * joint_pos_left_error + self.kd * joint_vel_left_error

        # Right foot PD control
        joint_pos_cmd_right = torch.clamp(joint_default_pos_right + joint_pos_cmd_right, joint_limits_right[:, :, 0], joint_limits_right[:, :, 1])
        joint_pos_right_error = joint_pos_cmd_right - joint_pos_right
        joint_vel_right_error = -joint_vel_right                                           # reference joint velocity = 0
        torque_right = self.kp * joint_pos_right_error + self.kd * joint_vel_right_error
        
        # Combine torque inputs
        torque = torch.zeros(self.num_envs, self.num_joint, device=self.device)
        torque[:, self.left_leg_indices] = torque_left
        torque[:, self.right_leg_indices] = torque_right
        
        # LPF for torque
        torque = self.alpha * torque + (1 - self.alpha)* self.old_torque                                                        # 0.049 default
        self.old_torque = torque.clone()

        # Clip torque based on torque_limits
        torque = torch.clamp(torque, -self.joint_torque_limits[:, :6], self.joint_torque_limits[:, :6])
        
        return torque
    
    def reset(self, env_ids: torch.tensor):
        # New delay level
        sampled_delays = torch.randint(
            self.min_delay, 
            self.max_delay + 1, 
            (len(env_ids),), 
            device=self.device
        )
        self.env_delays[env_ids] = sampled_delays
        
        # Reset buffer
        self.action_buffer[env_ids] = 0.0