import torch

class PI_Controller():
    """
    Low level model-free error feedback PI torque controller

    τ = Kp*e + Ki*∫e
    """

    def __init__(self, 
                 kp, ki, alpha, 
                 num_envs: int, num_dof: int, num_leg: int, 
                 device: str, dt: float,
                 joint_vel_limits: torch.Tensor, torque_limits: torch.Tensor):
        """
        Controller initialization  for velocity control

        Args:
            kp (float or torch.Tensor): Proportional gain. float value or (1, num_dof) size tensor.
            ki (float or torch.Tensor): Intergral gain. float value or (1, num_dof) size tensor.
            alpha (float): Low Pass Filter(LPF) gain.  α * new + (1 - α) * old
            num_envs (int): Number of pararell training environments.
            num_dof (int): controllable dof per leg.
            num_leg (int): Number of legs
            device (str): "cuda:0" or "cpu".
            dt (float): Simulation time-step.
        """

        self.device = device
        self.num_envs = num_envs
        self.num_dof = num_dof
        self.num_leg = num_leg
        self.dt = dt
        self.alpha = alpha
        self.joint_vel_limits = joint_vel_limits
        self.torque_limits = torque_limits
        self.old_torque = torch.zeros(self.num_envs, num_dof * num_leg, device=self.device)

        # Isaac sim's Joint order: ['hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint', 'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint']
        self.left_leg_indices = torch.tensor([6], device=self.device, dtype=torch.long)                                      # Hard coded
        self.right_leg_indices = torch.tensor([7], device=self.device, dtype=torch.long)                                     # Hard coded

        # kp gain
        if isinstance(kp, float):
            self.kp = torch.full((num_envs, num_dof), kp, device=device)
        elif isinstance(kp, torch.Tensor):
            if kp.shape != (1, num_dof):
                raise ValueError(f"kp tensor must have shape (1, {num_dof}), but got {kp.shape}")
            self.kp = kp.to(device).expand(num_envs, -1)        # Expand as num_envs
        else:
            raise TypeError("kp must be a float or a torch.Tensor of shape (1, num_dof)")

        # ki gain
        if isinstance(ki, float):
            self.ki = torch.full((num_envs, num_dof), ki, device=device)
        elif isinstance(ki, torch.Tensor):
            if ki.shape != (1, num_dof):
                raise ValueError(f"ki tensor must have shape (1, {num_dof}), but got {ki.shape}")
            self.ki = ki.to(device).expand(num_envs, -1)        # Expand as num_envs
        else:
            raise TypeError("ki must be a float or a torch.Tensor of shape (1, num_dof)")
    
    def compute_torque(
        self,
        joint_vel: torch.Tensor,
        joint_vel_cmd: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute joint input torque.

        Args:
            joint_vel (torch.Tensor): Current joint velocity [rad/s].
            joint_vel_cmd (torch.Tensor): Reference joint velocity(rad/s) [num_env, num_leg, num_joint].
            joint_limits (torch.Tensor): Joint position limits [num_env, num_joint, 2].
            torque_limits (torch.Tensor): Joint torque limits [num_env, num_joint].

        Returns:
            torch.Tensor: Joint torque.
        """
        # Robot dof
        leg_dof = self.num_dof                          # wheel joint
        num_joint = leg_dof * self.num_leg

        torque_limits = self.torque_limits[:, 6:]            # Extract wheel torque limits

        # --- Left Leg slicing ---
        joint_vel_left = torch.index_select(joint_vel, 1, self.left_leg_indices)
        joint_vel_cmd_left = joint_vel_cmd[:, 0].unsqueeze(-1)
        joint_vel_limits_left = torch.index_select(self.joint_vel_limits, 1, self.left_leg_indices)

        # --- Right Leg slicing ---
        joint_vel_right = torch.index_select(joint_vel, 1, self.right_leg_indices)
        joint_vel_cmd_right = joint_vel_cmd[:, 1].unsqueeze(-1)
        joint_vel_limits_right = torch.index_select(self.joint_vel_limits, 1, self.right_leg_indices)

        # Left foot PI control
        joint_vel_cmd_left = torch.clamp(joint_vel_cmd_left, - joint_vel_limits_left, joint_vel_limits_left)            # Clipping joint position command
        joint_vel_left_error = joint_vel_cmd_left - joint_vel_left
        torque_left = self.kp * joint_vel_left_error + self.ki * joint_vel_left_error * self.dt                         # PI feedback

        # Right foot PI control
        joint_vel_cmd_right = torch.clamp(joint_vel_cmd_right, - joint_vel_limits_right, joint_vel_limits_right)        # Clipping joint position command
        joint_vel_right_error = joint_vel_cmd_right - joint_vel_right
        torque_right = self.kp * joint_vel_right_error + self.ki * joint_vel_right_error * self.dt                      # PI feedback
        
        # Combine torque inputs
        torque = torch.zeros(self.num_envs, num_joint, device=self.device)
        torque = torch.cat((torque_left, torque_right), dim=1)
        
        # LPF for torque
        torque = self.alpha * torque + (1 - self.alpha)* self.old_torque                                                # 0.049 default
        self.old_torque = torque.clone()

        # Clip torque based on torque_limits
        torque = torch.clamp(torque, -torque_limits, torque_limits)
        
        return torque