from abc import abstractmethod
import os
import torch
import datetime

from typing import Dict
from torch.nn import Module

class Agent:
    """
    Base Agent Class
    """
    def __init__(self,
                 cfg: Dict,
                 model: Dict[str, Module],
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


    def set_running_mode(self, mode: str) -> None:
        """
        Setting the Running Mode (train and evaluation)

        Args:
            mode: the mode for agent implementation ['train', 'eval']
        
        Raises:
            ValueError: Not supported running mode. Please choose 'train' or 'eval'.
        """
        if mode == "train":
            for model in self.model.values():
                model.train()
        elif mode == "eval":
            for model in self.model.values():
                model.eval()
        else:
            raise ValueError("Not supported running mode. Please choose 'train' or 'eval'.")


    def save(self, path: str) -> None:
        """
        Save the agent to the specified path

        Args:
            path: Path to save the model to
        """
        modules = {}
        for name, module in self.checkpoint_modules.items():
            modules[name] = module.state_dict()
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
            for name, data in modules.items():
                module = self.checkpoint_modules.get(name, None)
                if module is not None:
                    module.load_state_dict(data)
                else:
                    print(f"Cannot load the {name} module. The agent doesn't have such an instance")
