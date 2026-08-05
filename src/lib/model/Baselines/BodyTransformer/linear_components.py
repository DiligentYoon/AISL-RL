import torch
import gymnasium as gym

import torch.nn as nn
import torch.nn.functional as F

from lib.model.model import Model
from lib.utils.graph_utils import Mapping

class ObsTokenizer(Model):
    def __init__(self,
                 mapping: Mapping, 
                 output_dim=64, 
                 device='cuda'):
        super().__init__()

        self.mapping = mapping
        self.map = self.mapping.map
        self.output_dim = output_dim
        self.device = device

        self.tokenizers = torch.nn.ModuleDict()
        for k, v in self.map.items():
            # input dims = [Batch, N_j, input_dim_per_joint] or [Batch, N_b, input_dim_per_body]
            input_dims, _ = v 
            input_dim = input_dims[1]
            self.tokenizers[k] = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: dict[str, torch.Tensor]):
        """
        :param x: dict of tensors with keys as body parts related to DOF Names (e.g., 'lower_waist', 'upper_arm', etc.)
        """
        outputs = []
        for key in x.keys():
            inputs = x[key]
            outputs.append(self.tokenizers[key](inputs))
        return torch.cat(outputs, dim=1)  # [batch_size, nbodies, embedding_dim]


class ActionDetokenizer(Model):
    def __init__(self, 
                 mapping: Mapping, 
                 action_dim: int, 
                 embedding_dim=64, 
                 use_mlp=False, 
                 device='cuda'):
        super().__init__()

        self.mapping = mapping
        self.map = self.mapping.map
        self.nbodies = len(self.map.keys()) # Body + Joints
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim # Total Action dim (Actuated Parts)
        self.device = device

        self.detokenizers = torch.nn.ModuleDict()
        for k, v in self.map.items():
            if k == 'body':
                continue  # No action for body
            _, output_indices = v
            output_dim = len(output_indices)
            if use_mlp:
                self.detokenizers[k] = nn.Sequential(
                    torch.nn.Linear(embedding_dim, 256),
                    nn.ReLU(),
                    torch.nn.Linear(256, 256),
                    nn.ReLU(),
                    torch.nn.Linear(256, output_dim),
                    torch.nn.Tanh(),
                )
            else:
                self.detokenizers[k] = nn.Sequential(
                    torch.nn.Linear(embedding_dim, output_dim),
                    torch.nn.Tanh(),
                )

    def forward(self, x):
        """
        :param x: [batch_size, nbodies, embedding_dim]
        """
        action = torch.zeros(x.shape[0], self.action_dim).to(self.device)
        for i, k in enumerate(self.map.keys()):
            if k == 'body':
                continue  # No action for body
            curr_action = self.detokenizers[k](x[:, i, :])
            action[:, self.map[k][1]] = curr_action.float()
        return action


class ValueDetokenizer(Model):
    def __init__(self, 
                 mapping: Mapping, 
                 embedding_dim=64, 
                 use_mlp=False, 
                 device='cuda'):
        super().__init__()

        self.mapping = mapping
        self.map = self.mapping.map
        self.nbodies = len(self.map.keys())
        self.embedding_dim = embedding_dim
        self.device = device

        self.detokenizers = torch.nn.ModuleDict()
        for k in self.map.keys():
            if use_mlp:
                self.detokenizers[k] = nn.Sequential(
                    torch.nn.Linear(embedding_dim, 256),
                    nn.ReLU(),
                    torch.nn.Linear(256, 256),
                    nn.ReLU(),
                    torch.nn.Linear(256, 1)
                )
            else:
                self.detokenizers[k] = torch.nn.Linear(embedding_dim, 1)

    def forward(self, x):
        values = torch.zeros(x.shape[0], x.shape[1]).to(self.device)
        for i, k in enumerate(self.map.keys()):
            values[:, i] = self.detokenizers[k](x[:, i, :]).squeeze(-1)
        
        # All body parts' values are averaged to get final value
        return torch.mean(values, dim=1, keepdim=True)