import torch
import torch.nn as nn

from torch.distributions import Normal
from lib.utils.Running_mean_std import RunningMeanStd
from lib.model.model import Model

class Actor(Model):
    def __init__(self, num_observations, num_actions, min_log_std, max_log_std, device):
        super().__init__()

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        
        # Define instances 
        self.device = device
        self.num_observations = num_observations
        self.num_actions = num_actions

        # Running mean, standard deviation standardizer
        self.actor_standardizer = RunningMeanStd(shape=self.num_observations, device=device)

        # Backbone
        self.net = nn.Sequential(nn.Linear(self.num_observations, 400),
                                 nn.ReLU(),
                                 nn.Linear(400, 200),
                                 nn.ReLU(),
                                 nn.Linear(200, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, self.num_actions),
                                 nn.Tanh())

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device))            # State independent log std

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
        # Input standardization
        standardized_input = self.actor_standardizer.standardize(observations, update=update_rms)

        mean_actions = self.net(standardized_input)
        log_std = self.log_std_parameter
        
        # Clamp log standard deviations
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)

        # Action distribution
        self.action_distribution = Normal(mean_actions, log_std.exp())

        if deterministic:
            actions = mean_actions
        else:
            # Sample using the reparameterization trick
            actions = self.action_distribution.rsample()

        # Log of the probability density function
        if taken_actions is not None:
            log_prob = self.action_distribution.log_prob(taken_actions)
        else:
            log_prob = self.action_distribution.log_prob(actions)

        log_prob = log_prob.sum(dim=-1)

        # Entropy : mean of (Batch, Action) dimension
        entropy = self.action_distribution.entropy().mean()

        return actions, log_prob, entropy
    
    def random_act(self, observations: torch.Tensor):
        """
        Random act for RL's beginning
        """
        
        batch_size = observations.shape[0]

        # Create a standard normal distribution for random actions (mean=0, std=1)
        mean = torch.zeros((batch_size, self.num_actions), device=self.device)
        std = torch.ones((batch_size, self.num_actions), device=self.device)
        action_distribution = Normal(mean, std)

        # Sample raw actions and squash them to [-1, 1] using tanh
        raw_actions = action_distribution.rsample()
        actions = torch.tanh(raw_actions)

        # Calculate log probability, correcting for the tanh transformation
        log_prob = action_distribution.log_prob(raw_actions)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6)
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
        self.net = nn.Sequential(nn.Linear(self.num_states, 400),
                                 nn.ReLU(),
                                 nn.Linear(400, 200),
                                 nn.ReLU(),
                                 nn.Linear(200, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 1))

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