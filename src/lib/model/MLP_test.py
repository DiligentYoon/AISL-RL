import torch
import torch.nn as nn

from typing import Optional
from torch.distributions import Normal
from lib.utils.Running_mean_std import RunningMeanStd
from lib.model.model import Model


class CommunetActor(Model):
    def __init__(self,
                 possible_agents: list[str],
                 num_observations: dict[str, int], 
                 num_actions: dict[str, int],
                 hidden_dim: int,
                 communet_depth: int,
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
        self.arm_module_list = nn.ModuleList()
        self.leg_module_list = nn.ModuleList()

        # Running mean, standard deviation standardizer
        self.actor_standardizer = nn.ModuleDict()
        self.actor_standardizer["arm"] = RunningMeanStd(shape=self.num_observations["arm"], device=device)
        self.actor_standardizer["leg"] = RunningMeanStd(shape=self.num_observations["leg"], device=device)
        
        # Action Squashing
        self.squash = squash

        # Define module list
        arm_head = nn.Sequential(
            nn.Linear(self.num_observations["arm"], 256),
            nn.ELU(),
            nn.Linear(256, hidden_dim),
            nn.ELU()
        )

        self.arm_module_list.append(arm_head)

        leg_head = nn.Sequential(
            nn.Linear(self.num_observations["leg"], 256),
            nn.ELU(),
            nn.Linear(256, hidden_dim),
            nn.ELU()
        )

        self.leg_module_list.append(leg_head)

        for _ in range(communet_depth - 2):
            arm_module = nn.Sequential(
                nn.Linear(hidden_dim * 2, 256),
                nn.ELU(),
                nn.Linear(256, hidden_dim),
                nn.ELU()
            )
            self.arm_module_list.append(arm_module)

            leg_module = nn.Sequential(
                nn.Linear(hidden_dim * 2, 256),
                nn.ELU(),
                nn.Linear(256, hidden_dim),
                nn.ELU()
            )
            self.leg_module_list.append(leg_module)

        arm_tail = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ELU(),
            nn.Linear(256, self.num_actions["arm"]),
            nn.ELU()
        )

        self.arm_module_list.append(arm_tail)

        leg_tail = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ELU(),
            nn.Linear(256, self.num_actions["leg"]),
            nn.ELU()
        )

        self.leg_module_list.append(leg_tail)

        # Log std parameter initialization
        self.log_std_parameter = nn.ParameterDict()
        self.log_std_parameter["arm"] = nn.Parameter(torch.zeros(self.num_actions["arm"], device=device), requires_grad=True) # State independent log std
        self.log_std_parameter["leg"] = nn.Parameter(torch.zeros(self.num_actions["leg"], device=device), requires_grad=True) # State independent log std

        self.init_weights()
        self.init_biases(val=0)

    def forward(self, 
                observations: torch.Tensor | dict[str, torch.Tensor],
                taken_actions: torch.Tensor | dict[str, torch.Tensor] | None, 
                deterministic: bool = False, 
                update_rms: bool = False):
        # eps
        eps = 1e-6
        
        # Input standardization
        arm_input = self.actor_standardizer["arm"].standardize(observations["arm"], update=update_rms)
        leg_input = self.actor_standardizer["leg"].standardize(observations["leg"], update=update_rms)

        for i in range(len(self.arm_module_list)):
            arm_output = self.arm_module_list[i](arm_input)
            leg_output = self.leg_module_list[i](leg_input)
            arm_input = torch.cat([arm_output, leg_output], dim=-1)
            leg_input = torch.cat([leg_output, arm_output], dim=-1)
    
        # 5. Action
        mean_action_arm = arm_output
        mean_action_leg = leg_output
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
    
class SharedBackbone(Model):
    def __init__(self, in_dim, num_arm_action, num_leg_action):
        super().__init__()

        self.num_arm_action = num_arm_action
        self.num_leg_action = num_leg_action

        self.shared = nn.Sequential(nn.Linear(in_dim, 256), 
                                    nn.ELU(),
                                    nn.Linear(256, 256), 
                                    nn.ELU(),
                                    nn.Linear(256, 256), 
                                    nn.ELU())
        
        self.head_arm = nn.Sequential(nn.Linear(256, 128), 
                                      nn.ELU(), 
                                      nn.Linear(128, self.num_arm_action))
        
        self.head_leg = nn.Sequential(nn.Linear(256, 256), 
                                      nn.ELU(), 
                                      nn.Linear(256, self.num_leg_action))

    def forward(self, x, role):
        g = self.shared(x)
        if role == "arm":
            arm_action = self.head_arm(g)
            return arm_action
        else:
            leg_action = self.head_leg(g)
            return leg_action

class SuperConnectedActor(Model):
    def __init__(self, 
                 possible_agents: list[str],
                 num_observations: dict[str, int], 
                 num_actions: dict[str, int],
                 min_log_std: float, 
                 max_log_std: float,
                 squash: bool, 
                 device: torch.device):
        super().__init__()

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        
        # Define instances 
        self.device = device
        self.num_observations = num_observations["arm"]
        self.num_actions = num_actions
        self.possible_agents = possible_agents

        # Action Squashing
        self.squash = squash

        # Running mean, standard deviation standardizer
        self.actor_standardizer = RunningMeanStd(shape=self.num_observations, device=device)

        # Superconnected backbone
        self.sharedbackbone = SharedBackbone(num_observations, num_actions["arm"], num_actions["leg"])

        # Log std parameter initialization
        self.log_std_parameter = nn.ParameterDict()
        self.log_std_parameter["arm"] = nn.Parameter(torch.zeros(self.num_actions["arm"], device=device), requires_grad=True) # State independent log std
        self.log_std_parameter["leg"] = nn.Parameter(torch.zeros(self.num_actions["leg"], device=device), requires_grad=True) # State independent log std

        self.init_weights()
        self.init_biases(val=0)

    def forward(self, 
                observations: torch.Tensor, 
                taken_actions: torch.Tensor | None, 
                deterministic: bool = False, 
                update_rms: bool = False):
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

        # 5. Action
        mean_action_arm = self.sharedbackbone(standardized_input, role="arm")
        mean_action_leg = self.sharedbackbone(standardized_input, role="leg")
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