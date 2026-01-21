import torch
import torch.nn as nn

from torch.distributions import Normal
from lib.utils.Running_mean_std import RunningMeanStd

class Actor(nn.Module):
    def __init__(self, observation_space_size, action_space_size, device):
        super().__init__()

        self.log_std_min = -20.0
        self.log_std_max = 2.0
        # Define instances 
        self.device = device
        self.num_observations = observation_space_size
        self.num_actions = action_space_size

        # Running mean, standard deviation standardizer
        self.actor_standardizer = RunningMeanStd(shape=self.num_observations, device=device)

        # Backbone
        self.net = nn.Sequential(nn.Linear(self.num_observations, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, self.num_actions),
                                 nn.Tanh())

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))            # State independent log std

    def forward(self, inputs):
        '''
        Forward propagation of actor NN
        
        :param inputs: Observation vector
        :type inputs: torch.Tensor
        '''
        standardized_input = self.actor_standardizer.standardization(inputs)

        mean_actions = self.net(standardized_input)
        log_std = self.log_std_parameter
        
        # Clamp log standard deviations
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # Action distribution
        self.action_distribution = Normal(mean_actions, log_std.exp())

        # Sample using the reparameterization trick
        actions = self.action_distribution.rsample()

        # Log of the probability density function
        log_prob = self.action_distribution.log_prob(actions)
        log_prob = log_prob.sum(dim=-1)

        return actions, log_prob
    
    ## =============== Auxilary functions =============== ##

    def init_weights(self) -> None:
        """
        Orthogonal initialize the model weights

        The following layers will be initialized:
        - torch.nn.Linear
        """

        def _update_weights(module):                                            # Recursive method
            for layer in module:
                if isinstance(layer, torch.nn.Sequential):
                    _update_weights(layer)
                elif isinstance(layer, torch.nn.Linear):
                    torch.nn.init.orthogonal_(layer.weight)                     # Initialize weight

        _update_weights(self.children())

    def init_biases(self, val: int = 0) -> None:
        """
        Constant initialize the model biases

        The following layers will be initialized:
        - torch.nn.Linear

        :param val: constant value to initialize biases
        :type val: int
        """

        def _update_biases(module):                                             # Recursive method
            for layer in module:
                if isinstance(layer, torch.nn.Sequential):
                    _update_biases(layer)
                elif isinstance(layer, torch.nn.Linear):
                    torch.nn.init.constant_(layer.bias, val=val)                # Initialize bias

        _update_biases(self.children())
    
    def save(self, path: str) -> None:
        """
        Save the model to the specific path

        :param path: Path to save the model to
        :type path: str
        """

        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load the model from the specific path

        The final storage device is determined by the constructor of the model

        :param path: Path to load the model from
        :type path: str
        """
        
        state_dict = torch.load(path, map_location=self.device, weights_only=False)  # prevent torch:FutureWarning
        self.load_state_dict(state_dict)

class Critic(nn.Module):
    def __init__(self, state_space_size, device):
        super().__init__()

        # Define instances 
        self.device = device
        self.num_states = state_space_size

        # Running mean, standard deviation standardizer
        self.critic_standardizer = RunningMeanStd(shape=self.num_states, device=device)

        # Backbone
        self.net = nn.Sequential(nn.Linear(self.num_states, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 1)
                                 )

    def forward(self, inputs: torch.Tensor):
        '''
        Forward propagation of critic NN
        
        :param inputs: State vector
        :type inputs: torch.Tensor
        '''
        standardized_input = self.critic_standardizer.standardization(inputs)
        expected_return = self.net(standardized_input)

        return expected_return, None
    
    ## =============== Auxilary functions =============== ##

    def init_weights(self) -> None:
        """
        Orthogonal initialize the model weights

        The following layers will be initialized:
        - torch.nn.Linear
        """

        def _update_weights(module):                                            # Recursive method
            for layer in module:
                if isinstance(layer, torch.nn.Sequential):
                    _update_weights(layer)
                elif isinstance(layer, torch.nn.Linear):
                    torch.nn.init.orthogonal_(layer.weight)                     # Initialize weight

        _update_weights(self.children())

    def init_biases(self, val: int = 0) -> None:
        """
        Constant initialize the model biases

        The following layers will be initialized:
        - torch.nn.Linear

        :param val: constant value to initialize biases
        :type val: int
        """

        def _update_biases(module):                                             # Recursive method
            for layer in module:
                if isinstance(layer, torch.nn.Sequential):
                    _update_biases(layer)
                elif isinstance(layer, torch.nn.Linear):
                    torch.nn.init.constant_(layer.bias, val=val)                # Initialize bias

        _update_biases(self.children())
    
    def save(self, path: str) -> None:
        """
        Save the model to the specific path

        :param path: Path to save the model to
        :type path: str
        """

        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load the model from the specific path

        The final storage device is determined by the constructor of the model

        :param path: Path to load the model from
        :type path: str
        """
        
        state_dict = torch.load(path, map_location=self.device, weights_only=False)  # prevent torch:FutureWarning
        self.load_state_dict(state_dict)