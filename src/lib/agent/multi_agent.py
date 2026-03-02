import os
import torch
import datetime
import gymnasium as gym

from typing import Dict, Mapping, Union, Sequence
from torch.nn import Module

from lib.utils.seed_utils import set_seed

class MultiAgent:
    """
    Base Agent Class
    """
    def __init__(self,
                 possible_agents: Sequence[str],
                 observation_space: gym.Space,
                 state_space: gym.Space,
                 action_space: gym.Space,
                 cfg: Dict,
                 model: Dict[str, Union[Module, Dict[str, Module]]],
                 device: torch.device):
        """
        Initialize the Base Agent Class
        
        Args:
            cfg: Dictionary including the configuration related to RL Agent
            model: Dictionary including the NN Model for Actor and Critic
            device: torch device [cuda, cpu] 
        """
        self.cfg = cfg
        self.model = model
        self.device = device
        self.seed = self.cfg["seed"]
        self.possible_agents = possible_agents
        self.num_agents = len(possible_agents)

        # set seed
        set_seed(self.seed)

        # Spaces
        self.observation_space = observation_space
        self.state_space = state_space
        self.action_space = action_space

        # checkpoint
        self.checkpoint_modules = {}
        self.checkpoint_interval = self.cfg.get("experiment", {}).get("checkpoint_interval", "auto")
        self.checkpoint_best_modules = {"timestep": 0, "reward": -(2**31), "saved": False, "modules": {}}

        # experiment directory
        directory = self.cfg.get("experiment", {}).get("directory", "")
        experiment_name = self.cfg.get("experiment", {}).get("experiment_name", "")
        if not directory:
            directory = os.path.join(os.getcwd(), "runs")
        if not experiment_name:
            experiment_name = "{}_{}".format(
                datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"), self.__class__.__name__
            )
        self.experiment_dir = os.path.join(directory, experiment_name)


    def set_running_mode(self, mode: str, uid: str = None) -> None:
        """
        Setting the Running Mode (train and evaluation)

        Args:
            mode: the mode for agent implementation ['train', 'eval']
        
        Raises:
            ValueError: Not supported running mode. Please choose 'train' or 'eval'.
        """
        if uid == None:
            for uid in self.possible_agents:
                if mode == "train":
                    for model in self.model[uid].values():
                        model.train()
                elif mode == "eval":
                    for model in self.model[uid].values():
                        model.eval()
                else:
                    raise ValueError("Not supported running mode. Please choose 'train' or 'eval'.")
        
        else:
            if mode == "train":
                for model in self.model[uid].values():
                    model.train()
            elif mode == "eval":
                for model in self.model[uid].values():
                    model.eval()
            else:
                raise ValueError("Not supported running mode. Please choose 'train' or 'eval'.")


    def save(self, path: str, path_jit: str | None = None) -> None:
        """
        Save the agent to the specified path

        Args:
            path: Path to save the model to
            path_jit: Path to save jit scriptModule to (TODO: Multi-agent is not supported yet.)
        """
        modules = {}
        for uid in self.possible_agents:
            uid_module = {}
            for name, module in self.checkpoint_modules[uid].items():
                uid_module[name] = module.state_dict()

            modules[uid] = uid_module
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(modules, path)
        
    

    def load(self, path: str) -> None:
        """
        Load the agent from the specified path
        
        Args:
            path: Path to load the model from
        """
        modules = torch.load(path, map_location=self.device)
        if type(modules) is dict:
            for uid in self.possible_agents:
                for name, data in modules[uid].items():
                    module = self.checkpoint_modules[uid].get(name, None)
                    if module is not None:
                        module.load_state_dict(data)
                    else:
                        print(f"Cannot load the {name} module. The agent doesn't have such an instance")
        else:
            raise RuntimeError(f"Cannot load the module at {path}. It is not a dictionary")