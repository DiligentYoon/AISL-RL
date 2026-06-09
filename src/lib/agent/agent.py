import os
import copy
import torch
import datetime

from typing import Dict
from torch.nn import Module
from lib.model.MLP import ActorInference

from lib.utils.seed_utils import set_seed

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
        self.seed = self.cfg["seed"]

        # set seed
        set_seed(self.seed)

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
        if self.model.get("actor", None) is not None:
            self.num_action = self.model["actor"].num_actions


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


    def save(self, path: str, path_onnx: str | None = None) -> None:
        """
        Save the agent to the specified path

        Args:
            path: Path to save the full checkpoint (state_dicts) to
            path_onnx: Path to save the deterministic ONNX actor for deployment. Skipped if None.
        """
        modules = {}
        for name, module in self.checkpoint_modules.items():
            modules[name] = module.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(modules, path)

        if path_onnx is not None:
            actor = self.model["actor"]

            # Build a CPU-resident deterministic wrapper for export.
            actor_export = copy.deepcopy(actor).to("cpu").eval()
            actor_onnx = ActorInference(actor_export, squash=actor.squash).eval()

            dummy = torch.zeros(1, actor.num_observations, dtype=torch.float32)
            os.makedirs(os.path.dirname(path_onnx), exist_ok=True)
            torch.onnx.export(
                actor_onnx,
                (dummy,),
                path_onnx,
                input_names=["observations"],
                output_names=["actions"],
                dynamic_axes={
                    "observations": {0: "batch"},
                    "actions": {0: "batch"},
                },
                opset_version=17,
                do_constant_folding=True,
            )
    

    def load(self, path: str) -> None:
        """
        Load the agent from the specified path
        
        Args:
            path: Path to load the model from
        """
        modules = torch.load(path, map_location=self.device, weights_only=False)
        if type(modules) is dict:
            for name, data in modules.items():
                module = self.checkpoint_modules.get(name, None)
                if module is not None:
                    module.load_state_dict(data)
                else:
                    print(f"Cannot load the {name} module. The agent doesn't have such an instance")
