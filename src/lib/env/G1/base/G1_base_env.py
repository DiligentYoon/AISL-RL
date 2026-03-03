import torch

from abc import abstractmethod
from isaaclab.assets import Articulation

from lib.env.env import Env
from lib.env.G1.base.G1_base_env_cfg import G1BaseEnvCfg


class G1BaseEnv(Env):
    # Load config file
    cfg: G1BaseEnvCfg

    def __init__(self, cfg: G1BaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total env ids
        self.total_env_ids = torch.arange(self.num_envs, device=self.device)
        # Joint & Link Limits
        self.joint_pos_limits = self._robot.data.joint_pos_limits
        self.joint_vel_limits = self._robot.data.joint_vel_limits

        # Joint Torque Limits
        self.soft_joint_torque_limits = 0.9 * self._robot.data.joint_effort_limits

        # Joint Ids
        self._joint_dof_ids, _ = self._robot.find_joints(".*")
        
        self.total_leg_joint_ids, _ = self._robot.find_joints([r".*_hip_(pitch|roll|yaw)_joint",
                                                               r".*_knee_joint",
                                                               r".*_ankle_(pitch|roll)_joint"])
        
        self.total_arm_joint_ids, _ = self._robot.find_joints([r"torso_joint", 
                                                               r".*_shoulder_(pitch|roll|yaw)_joint",
                                                               r".*_elbow_(pitch|roll)_joint",
                                                               r".*_(zero|one|two|three|four|five|six)_joint"])
        # Joint Limits
        self.leg_joint_limits = self.joint_pos_limits[:, self.total_leg_joint_ids]
        self.arm_joint_limits = self.joint_pos_limits[:, self.total_arm_joint_ids]

    # Create scene
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

    # Reset Env
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)


    ## =============== RL main abstract methods ================ ##

    @abstractmethod
    def _pre_physics_step(self, actions: torch.Tensor):
        """Pre-process actions before stepping through the physics.

        This function is responsible for pre-processing the actions before stepping through the physics.
        It is called before the physics stepping (which is decimated).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, action_dim).
        """
        raise NotImplementedError(f"Please implement the '_pre_physics_step' method for {self.__class__.__name__}.")

    @abstractmethod
    def _apply_action(self):
        """Apply actions to the simulator.

        This function is responsible for applying the actions to the simulator. It is called at each
        physics time-step.
        """
        raise NotImplementedError(f"Please implement the '_apply_action' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute and return the states for the environment.

        The state-space is used for asymmetric actor-critic architectures. It is configured
        using the :attr:`DirectRLEnvCfg.state_space` parameter.

        Returns:
            The states for the environment. If the environment does not have a state-space, the function
            returns a None.
        """
        raise NotImplementedError(f"Please implement the '_get_observations' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_states(self) -> torch.Tensor | None:
        """Compute and return the states for the environment.

        The state-space is used for asymmetric actor-critic architectures. It is configured
        using the :attr:`DirectRLEnvCfg.state_space` parameter.

        Returns:
            The states for the environment. If the environment does not have a state-space, the function
            returns a None.
        """
        if self.state_space is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}: state_space is set ({self.state_space}), "
                "so '_get_states' must be implemented to return privileged critic states.")
        else:
            return None  # noqa: R501

    @abstractmethod
    def _get_rewards(self) -> torch.Tensor:
        """Compute and return the rewards for the environment.

        Returns:
            The rewards for the environment. Shape is (num_envs,).
        """
        raise NotImplementedError(f"Please implement the '_get_rewards' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_dones(self):
        """Compute and return the done flags for the environment.

        Returns:
            A tuple containing the done flags for termination and time-out.
            Shape of individual tensors is (num_envs,).
        """
        raise NotImplementedError(f"Please implement the '_get_dones' method for {self.__class__.__name__}.")
    
    @abstractmethod
    def _compute_intermediate_values(self):
        """Compute planning states for convenient observation setting and reward calculating."""

        raise NotImplementedError(f"Please implement the '_compute_intermediate_values' method for {self.__class__.__name__}.")
    



        


