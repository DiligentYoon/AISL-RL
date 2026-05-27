import torch
import torch.nn as nn
import gymnasium as gym

from typing import Union
from torch.distributions import Normal
from lib.utils.Running_mean_std import RunningMeanStd
from lib.model.model import Model

from lib.model.Baselines.BodyTransformer.linear_components import ObsTokenizer, ActionDetokenizer, ValueDetokenizer
from lib.model.Baselines.BodyTransformer.transformer_components import BodyTransformer
from lib.utils.graph_utils import Mapping

from lib.utils.wrapper_utils import unflatten_tensorized_space


class BodyLevelActor(Model):
    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 mapping: Mapping,
                 tokenizer: ObsTokenizer,
                 trunk: BodyTransformer,
                 detokenizer: ActionDetokenizer,
                 device: torch.device,
                 min_log_std: float = -20,
                 max_log_std: float = 2):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.mapping = mapping

        self.tokenizer = tokenizer
        self.trunk = trunk
        self.detokenizer = detokenizer
                                             
        self.model = nn.Sequential(tokenizer, trunk, detokenizer)

        self.num_actions = action_space.shape[0]
        self.log_std = nn.Parameter(torch.zeros(self.num_actions, device=device))
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        # Running mean, standard deviation standardizer
        self.body_standardizer = RunningMeanStd(shape=observation_space['body'].shape, device=device)
        self.joint_standardizer = RunningMeanStd(shape=observation_space['joint'].shape, device=device)

        self.init_weights()
        self.init_biases(val=0)

        # === Parameters Summary ===
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[Actor Network] Total Parameters: {total_params}")

    def forward(self, 
                observations: Union[dict[str, torch.Tensor], torch.Tensor],
                taken_actions: Union[torch.Tensor, None],
                deterministic: bool = False,
                update_rms: bool = False):
        """
        Action Processing by GNN Policy

        :param observations: observations from buffer and environment
        :type observations: dict[str, torch.Tensor] | torch.Tensor
        :param taken_actions: actions from buffer for log_prob calculation
        """
        # From Tensor to Dict switching {'body': [B, N_b, m], 'joint': [Batch, N_j, n]}
        if isinstance(observations, torch.Tensor):
            observations = unflatten_tensorized_space(self.observation_space, observations)
        # Standardization
        observations['body'] = self.body_standardizer.standardize(observations['body'], update=update_rms)
        observations['joint'] = self.joint_standardizer.standardize(observations['joint'], update=update_rms)
        # Env <-> Mapping Dictionary Conversion
        observations = self.mapping.create_observation(observations)

        # Forward Propagation (Tokenization -> Positional Embedding -> Attention-Based Message Passing -> Detokenization)
        actions_mean = self.model(observations) # [Batch, Nbodies, Action_dim]

        # Our detokenizer assumes actions dim is 1 per body part
        if len(actions_mean.shape) > 2:
            actions_mean = actions_mean.squeeze(-1) # [Batch, Nbodies]

        # Log std parameter
        log_std = self.log_std
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)

        # Action distribution
        dist = Normal(actions_mean, log_std.exp())

        # Action sampling
        if deterministic:
            actions = actions_mean
        else:
            actions = dist.rsample()

        # Log prob calculation
        if taken_actions is not None:
            log_prob = dist.log_prob(taken_actions)
        else:
            log_prob = dist.log_prob(actions)
        log_prob = log_prob.sum(dim=-1)

        # Entropy
        entropy = dist.entropy().mean(dim=-1)

        return actions, log_prob, entropy


class BodyLevelCritic(Model):
    def __init__(self,
                 state_space: gym.Space,
                 mapping: Mapping,
                 tokenizer: ObsTokenizer,
                 trunk: BodyTransformer,
                 detokenizer: ValueDetokenizer,
                 device: torch.device):
        super().__init__()

        self.state_space = state_space
        self.device = device
        self.mapping = mapping

        self.tokenizer = tokenizer
        self.trunk = trunk
        self.detokenizer = detokenizer

        self.model = nn.Sequential(tokenizer, trunk, detokenizer)
        
        # Running mean, standard deviation standardizer
        self.body_standardizer = RunningMeanStd(shape=state_space['body'].shape, device=device)
        self.joint_standardizer = RunningMeanStd(shape=state_space['joint'].shape, device=device)

        # Parameters Initialization
        self.init_weights()
        self.init_biases(val=0)

        # === Parameters Summary ===
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[Critic Network] Total Parameters: {total_params}")

    def forward(self, 
                states: Union[dict[str, torch.Tensor], torch.Tensor],
                update_rms: bool = False):
        """
        Value Processing by GNN Critic

        :param states: states from buffer and environment
        :type states: dict[str, torch.Tensor] | torch.Tensor
        :param taken_actions: actions from buffer for log_prob calculation
        """
        # From Tensor to Dict switching {'body': [B, N_b, m], 'joint': [Batch, N_j, n]}
        if isinstance(states, torch.Tensor):
            states = unflatten_tensorized_space(self.state_space, states)
        # Standardization
        states['body'] = self.body_standardizer.standardize(states['body'], update=update_rms)
        states['joint'] = self.joint_standardizer.standardize(states['joint'], update=update_rms)
        # Env <-> Mapping Dictionary Conversion
        states = self.mapping.create_observation(states)

        # Forward Propagation (Tokenization -> Positional Embedding -> Attention-Based Message Passing -> Detokenization)
        values = self.model(states) # [Batch, 1]

        return values, None, None