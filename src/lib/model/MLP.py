import torch
import torch.nn as nn

from typing import Optional
from torch.distributions import Normal
from lib.utils.Running_mean_std import RunningMeanStd
from lib.model.model import Model


class SharedBackbone(Model):
    def __init__(self, in_dim, d_arm=64, d_leg=64):
        super().__init__()

        self.d_arm = d_arm
        self.d_leg = d_leg

        self.shared = nn.Sequential(nn.Linear(in_dim, 256), 
                                    nn.ELU(),
                                    nn.Linear(256, 256), 
                                    nn.ELU())
        
        self.head_arm = nn.Sequential(nn.Linear(256, 128), 
                                      nn.ELU(), 
                                      nn.Linear(128, d_arm))
        
        self.head_leg = nn.Sequential(nn.Linear(256, 128), 
                                      nn.ELU(), 
                                      nn.Linear(128, d_leg))

    def forward(self, x, role):
        g = self.shared(x)
        if role == "arm":
            h_arm = self.head_arm(g)
            return h_arm
        else:
            h_leg = self.head_leg(g)
            return h_leg


class SharedActor(Model):
    def __init__(self,
                 possible_agents: list[str],
                 num_observations: dict[str, int], 
                 num_actions: dict[str, int],
                 encoder_hidden_dim: int,
                 RMA_hidden_dim: int,
                 min_log_std: float, 
                 max_log_std: float,
                 squash: bool, 
                 device: torch.device):
        super().__init__()

        # Possible Agents
        self.possible_agents = possible_agents

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        
        # Define instances 
        self.device = device
        self.num_observations = num_observations
        self.num_actions = num_actions

        # Running mean, standard deviation standardizer
        self.actor_standardizer = nn.ModuleDict()
        self.actor_standardizer["arm"] = RunningMeanStd(shape=self.num_observations["arm"], device=device)
        self.actor_standardizer["leg"] = RunningMeanStd(shape=self.num_observations["leg"], device=device)
        
        # Action Squashing
        self.squash = squash

        # RMA
        self.is_rma = RMA_hidden_dim > 0

        # Encoder
        self.encoder = nn.ModuleDict()
        self.encoder["arm"] = nn.Sequential(nn.Linear(self.num_observations["arm"], 256),
                                            nn.ELU(),
                                            nn.Linear(256, encoder_hidden_dim),
                                            nn.ELU())
        self.encoder["leg"] = nn.Sequential(nn.Linear(self.num_observations["leg"], 256),
                                            nn.ELU(),
                                            nn.Linear(256, encoder_hidden_dim),
                                            nn.ELU())
        
        # Shared Backbone
        self.shared_backbone = SharedBackbone(in_dim=encoder_hidden_dim+encoder_hidden_dim+RMA_hidden_dim)
        self.shared_output_dim = {
            "arm": self.shared_backbone.d_arm,
            "leg": self.shared_backbone.d_leg}

        # Output Head
        self.head = nn.ModuleDict()
        self.head["arm"] = nn.Sequential(nn.Linear(encoder_hidden_dim + self.shared_output_dim["arm"], 128),
                                         nn.ELU(),
                                         nn.Linear(128, self.num_actions["arm"]))
        
        self.head["leg"] = nn.Sequential(nn.Linear(encoder_hidden_dim + self.shared_output_dim["leg"], 128),
                                         nn.ELU(),
                                         nn.Linear(128, self.num_actions["leg"]))

        
        self.log_std_parameter = nn.ParameterDict()
        self.log_std_parameter["arm"] = nn.Parameter(torch.zeros(self.num_actions["arm"], device=device), requires_grad=True) # State independent log std
        self.log_std_parameter["leg"] = nn.Parameter(torch.zeros(self.num_actions["leg"], device=device), requires_grad=True) # State independent log std

        self.init_weights()
        self.init_biases(val=0)

    def forward(self, 
                observations: torch.Tensor | dict[str, torch.Tensor],
                shared_infos: torch.Tensor | None,
                taken_actions: torch.Tensor | dict[str, torch.Tensor] | None, 
                deterministic: bool = False, 
                update_rms: bool = False):
        
        # eps
        eps = 1e-6
        # Input standardization
        standardized_input_arm = self.actor_standardizer["arm"].standardize(observations["arm"], update=update_rms)
        standardized_input_leg = self.actor_standardizer["leg"].standardize(observations["leg"], update=update_rms)
        # forward propagation

        # 1. Encoding
        z_arm = self.encoder["arm"](standardized_input_arm) 
        z_leg = self.encoder["leg"](standardized_input_leg)

        # 2. Shared info concat
        if self.is_rma:
            x_arm = torch.cat([z_arm, z_leg.detach(), shared_infos], dim=-1)
            x_leg = torch.cat([z_leg, z_arm.detach(), shared_infos], dim=-1)
        else:
            x_arm = torch.cat([z_arm, z_leg.detach()], dim=-1)
            x_leg = torch.cat([z_leg, z_arm.detach()], dim=-1)

        # 3. Shared info encoding
        h_arm = self.shared_backbone(x_arm, role="arm")
        h_leg = self.shared_backbone(x_leg, role="leg")

        # 4. Final input
        x_arm = torch.cat([z_arm, h_arm], dim=-1)
        x_leg = torch.cat([z_leg, h_leg], dim=-1)

        # 5. Action
        mean_action_arm = self.head["arm"](x_arm)
        mean_action_leg = self.head["leg"](x_leg)
        mean_action = {
            "arm": mean_action_arm,
            "leg": mean_action_leg}

        # log std
        log_std_arm = torch.clamp(self.log_std_parameter["arm"], self.min_log_std, self.max_log_std)
        log_std_leg = torch.clamp(self.log_std_parameter["leg"], self.min_log_std, self.max_log_std)
        log_std = {
            "arm": log_std_arm,
            "leg": log_std_leg}

        # Action Processing
        actions = {}
        log_probs = {}
        entropies = {}
        for uid in self.possible_agents:
            action_distribution = Normal(mean_action[uid], log_std[uid].exp())

            if deterministic:
                raw_actions = mean_action[uid]
            else:
                # Sample using the reparameterization trick
                raw_actions = action_distribution.rsample()

            # Log of the probability density function
            if self.squash:
                # tanh squasing with log probability dorrection
                action = torch.tanh(raw_actions)
                if taken_actions is not None:
                    taken_actions[uid] = torch.clip(taken_actions[uid], -1.0 + eps, 1.0 - eps)
                    raw_taken_actions = torch.atanh(taken_actions[uid])
                    log_prob = action_distribution.log_prob(raw_taken_actions) - torch.log(1 - taken_actions[uid].pow(2) + eps)
                else:
                    log_prob = action_distribution.log_prob(raw_actions) - torch.log(1 - action.pow(2) + eps)

            else:
                # no squasing without correction
                action = raw_actions
                if taken_actions is not None:
                    log_prob = action_distribution.log_prob(taken_actions[uid])
                else:
                    log_prob = action_distribution.log_prob(action)

            log_prob = log_prob.sum(dim=-1)

            # Entropy : mean of (Batch, Action) dimension
            entropy = action_distribution.entropy().mean()

            actions[uid] = action
            log_probs[uid] = log_prob
            entropies[uid] = entropy

        return actions, log_probs, entropies



class Actor(Model):
    def __init__(self, num_observations, num_actions, min_log_std, max_log_std, squash, device):
        super().__init__()

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        
        # Define instances 
        self.device = device
        self.num_observations = num_observations
        self.num_actions = num_actions
        
        # Action Squashing
        self.squash = squash

        # Running mean, standard deviation standardizer
        self.actor_standardizer = RunningMeanStd(shape=self.num_observations, device=device)

        # Backbone
        # self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
        #                          nn.ELU(),
        #                          nn.Linear(256, 128),
        #                          nn.ELU(),
        #                          nn.Linear(128, 64),
        #                          nn.ELU(),
        #                          nn.Linear(64, self.num_actions),
        #                          nn.Tanh())
        
        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 nn.Linear(64, self.num_actions))

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device), requires_grad=True) # State independent log std

        # Initialize parameters
        self.init_weights()
        self.init_biases(val=0)

    def forward(self, observations: torch.Tensor, taken_actions: torch.Tensor | None, deterministic: bool = False, update_rms: bool = False):
        """
        Forward propagation of actor NN
        
        :param observations: Observation vector
        :type observations: torch.Tensor
        :param taken_actions: Action vector
        :type taken_actions: torch.Tensor
        :param deterministic: Is actor evaluation mode
        :type deterministic: bool
        :param update_rms: Update a Runningmeanstd distrubution
        :type update_rms: bool 
        """
        # eps
        eps = 1e-6
        # Input standardization
        standardized_input = self.actor_standardizer.standardize(observations, update=update_rms)

        mean_actions = self.net(standardized_input)
        log_std = self.log_std_parameter
        
        # Clamp log standard deviations
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)

        # Action distribution
        action_distribution = Normal(mean_actions, log_std.exp())

        if deterministic:
            raw_actions = mean_actions
        else:
            # Sample using the reparameterization trick
            raw_actions = action_distribution.rsample()

        # Log of the probability density function
        if self.squash:
            # tanh squasing with log probability dorrection
            actions = torch.tanh(raw_actions)
            if taken_actions is not None:
                taken_actions = torch.clip(taken_actions, -1.0 + eps, 1.0 - eps)
                raw_taken_actions = torch.atanh(taken_actions)
                log_prob = action_distribution.log_prob(raw_taken_actions) - torch.log(1 - taken_actions.pow(2) + eps)
            else:
                log_prob = action_distribution.log_prob(raw_actions) - torch.log(1 - actions.pow(2) + eps)

        else:
            # no squasing without correction
            actions = raw_actions
            if taken_actions is not None:
                log_prob = action_distribution.log_prob(taken_actions)
            else:
                log_prob = action_distribution.log_prob(actions)

        log_prob = log_prob.sum(dim=-1)

        # Entropy : mean of (Batch, Action) dimension
        entropy = action_distribution.entropy().mean()

        return actions, log_prob, entropy
    
    def random_act(self, observations: torch.Tensor):
        """
        Random act for RL's beginning
        """
        eps = 1e-6
        batch_size = observations.shape[0]

        # Create a standard normal distribution for random actions (mean=0, std=1)
        mean = torch.zeros((batch_size, self.num_actions), device=self.device)
        std = torch.ones((batch_size, self.num_actions), device=self.device)
        action_distribution = Normal(mean, std)

        # Sample raw actions and squash them to [-1, 1] using tanh
        raw_actions = action_distribution.rsample()
        if self.squash:
            # tanh squasing with log probability dorrection
            actions = torch.tanh(raw_actions)
            log_prob = action_distribution.log_prob(raw_actions) - torch.log(1 - actions.pow(2) + eps)

        else:
            # no squasing without correction
            actions = raw_actions
            log_prob = action_distribution.log_prob(actions)

        log_prob = log_prob.sum(dim=-1)

        # Entropy of the base (Normal) distribution
        entropy = action_distribution.entropy().mean()

        return actions, log_prob, entropy

class Critic(Model):
    def __init__(self, num_states, device):
        super().__init__()

        # Define instances 
        self.device = device
        self.num_states = num_states

        # Running mean, standard deviation standardizer
        self.critic_standardizer = RunningMeanStd(shape=self.num_states, device=device)

        # Backbone
        self.net = nn.Sequential(nn.Linear(self.num_states, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 nn.Linear(64, 1))

        # Initialize parameters
        self.init_weights()
        self.init_biases(val=0)

    def forward(self, inputs: torch.Tensor, deterministic: bool = False, update_rms: bool = False):
        """
        Forward propagation of critic NN
        
        :param inputs: State vector
        :type inputs: torch.Tensor
        :param deterministic: Is critic evaluation mode
        :type deterministic: bool 
        """

        standardized_input = self.critic_standardizer.standardize(inputs, update=update_rms)
        expected_return = self.net(standardized_input)

        return expected_return, None, None


class ActorInference(nn.Module):
    def __init__(self, actor: nn.Module, squash: bool = True, action_scale_factor: Optional[torch.Tensor] = None):
        super().__init__()
        self.net = actor.net
        self.log_std_parameter = actor.log_std_parameter
        self.min_log_std = float(actor.min_log_std)
        self.max_log_std = float(actor.max_log_std)
        self.squash = bool(squash)
        self.actor_standardizer = actor.actor_standardizer

        # action_scale_factor: shape (num_actions,)
        if action_scale_factor is None:
            action_scale_factor = torch.ones_like(self.log_std_parameter)  # (num_actions,)
        self.register_buffer("action_scale_factor", action_scale_factor)

    def forward(self, observations: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        x = self.actor_standardizer.standardize(observations, update=False)
        mean = self.net(x)

        if deterministic:
            a = mean
        else:
            log_std = torch.clamp(self.log_std_parameter, self.min_log_std, self.max_log_std)
            std = torch.exp(log_std)
            a = mean + std * torch.randn_like(mean)

        if self.squash:
            a = torch.tanh(a)

        a = a * 0.1 * self.action_scale_factor
        return a