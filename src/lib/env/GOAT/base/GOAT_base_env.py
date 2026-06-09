import torch

from abc import abstractmethod
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from lib.env.GOAT.base.GOAT_base_env_cfg import GOATBaseEnvCfg
from lib.env.env import Env 


class GOATBaseEnv(Env):
    # Load config file
    cfg: GOATBaseEnvCfg

    def __init__(self, cfg: GOATBaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total env ids
        self.total_env_ids = torch.arange(self.num_envs, device=self.device)
        # Joint & Link Limits
        self.joint_pos_limits = self._robot.data.joint_pos_limits
        self.joint_vel_limits = self._robot.data.joint_vel_limits
        # Joint Torque Limits
        self.torque_limits = torch.tensor(self.cfg.torque_limits, device=self.device).unsqueeze(0).expand(self.num_envs, -1) # Isaac sim cannot bring torque limits from urdf
        # Joint Ids
        self.joint_ids, _ = self._robot.find_joints(["hip_.*", "thigh_.*", "knee_.*"])
        self.wheel_ids, _ = self._robot.find_joints(["wheel_.*"])
        
    # Create scene
    def _setup_scene(self):
        # robot
        self._robot = Articulation(self.cfg.GOAT_cfg)
        self.scene.articulations["robot"] = self._robot
        # sensor
        self.scene.sensors["contact_sensor"] = ContactSensor(self.cfg.contact_sensors)
        self.contact_sensors = self.scene.sensors["contact_sensor"]
        self.contact_sensors.update_period = self.cfg.sim_dt
        # clone env
        self.scene.clone_environments(copy_from_source=True)         

    # Reset Env
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Pre-process actions before stepping through the physics.

        This function is responsible for pre-processing the actions before stepping through the physics.
        It is called before the physics stepping (which is decimated).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, action_dim).
        """
        self.actions = actions.clone()
        self.processed_actions = actions.clone()
        self.processed_actions[:, self.joint_ids] *= self.cfg.action_scale_factor["joint"][0]
        self.processed_actions[:, self.wheel_ids] *= self.cfg.action_scale_factor["wheel"][0]
        
    def _apply_action(self):   
        """Apply actions to the simulator.

        This function is responsible for applying the actions to the simulator. It is called at each
        physics time-step.
        """      
        # Current state
        cmd_joint_pos = self._robot.data.default_joint_pos[:, self.joint_ids] + self.processed_actions[:, self.joint_ids]
        cmd_wheel_vel = self.processed_actions[:, self.wheel_ids]
        
        # Apply command
        self._robot.set_joint_position_target(cmd_joint_pos, joint_ids=self.joint_ids)
        self._robot.set_joint_velocity_target(cmd_wheel_vel, joint_ids=self.wheel_ids)
        
        # self._robot.write_joint_state_to_sim(self._robot.data.default_joint_pos, self._robot.data.default_joint_vel)

    ## =============== RL main abstract methods ================ ##

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
    



        


