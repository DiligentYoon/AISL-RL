from typing import Any, Union, Dict, Sequence, Mapping

import copy
import itertools
import gymnasium as gym
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.agent.mappo import MAPPO
from lib.buffer.rolloutbuffer import RolloutBuffer

from lib.utils.Learning_rate_scheduler import KLAdaptiveLR
from lib.utils.wrapper_utils import unflatten_tensorized_space

class CooperativeMAPPO(MAPPO):
    def __init__(self,
                 observation_space: gym.Space,
                 state_space: gym.Space,
                 action_space: gym.Space,
                 possible_agents: Sequence[str],
                 model: Dict[str, Union[nn.Module, Dict[str, nn.Module]]],
                 buffer: Dict[str, RolloutBuffer],
                 device: Union[str, torch.device],
                 cfg: Dict) -> None:
        """Cooperative Multi Agent Proximal Policy Optimization with parameter-shared backbone

        This architecture is expanded version of MAPPO for cooperative behaviors.

        Args:
            observation_space: observation space for each policy network
            state_space: state space for each value network
            action_space: action space of each policy network
            possible_agents: Name of all possible agents the environment could generate
            model: Models used by the agent
            buffer: Memory to storage the transitions.
            device: Device on which a tensor/array is or will be allocated (cuda, cpu).
            cfg: Configuration dictionary
        """
        super().__init__(observation_space, state_space, action_space, possible_agents, model, buffer, device, cfg)
        # Shared
        self.shared_actor = self.actors["arm"]

        # Checkpoint Module
        self.checkpoint_modules = {}

        # Optimizers
        self.optimizers = {}
        self.optimizers["actor"] = {}
        self.optimizers["critic"] = {}
        
        # Learning Rate Scheduler
        self.optimizer_schedulers = {}
        self.optimizer_schedulers["actor"] = {}
        self.optimizer_schedulers["critic"] = {}
        
        arm_params = []
        leg_params = []

        if hasattr(self.shared_actor, "encoder"):
            arm_params += list(self.shared_actor.encoder["arm"].parameters())
            leg_params += list(self.shared_actor.encoder["leg"].parameters())

        if hasattr(self.shared_actor, "head"):
            arm_params += list(self.shared_actor.head["arm"].parameters())
            leg_params += list(self.shared_actor.head["leg"].parameters())

        arm_params += [self.shared_actor.log_std_parameter["arm"]]
        leg_params += [self.shared_actor.log_std_parameter["leg"]]

        shared_params = list(self.shared_actor.shared_backbone.parameters())

        self.optimizers["actor"]["arm"] = torch.optim.Adam(arm_params, lr=self.learning_rate)
        self.optimizers["actor"]["leg"] = torch.optim.Adam(leg_params, lr=self.learning_rate)
        self.optimizers["actor"]["shared"] = torch.optim.Adam(shared_params, lr=self.learning_rate)

        self.optimizers["critic"]["arm"] = torch.optim.Adam(self.critics["arm"].parameters(), lr=self.learning_rate)
        self.optimizers["critic"]["leg"] = torch.optim.Adam(self.critics["leg"].parameters(), lr=self.learning_rate)

        # self.optimizer_schedulers["actor"]["arm"] = KLAdaptiveLR(self.optimizers["actor"]["arm"])
        # self.optimizer_schedulers["actor"]["leg"] = KLAdaptiveLR(self.optimizers["actor"]["leg"])
        # self.optimizer_schedulers["actor"]["shared"] = KLAdaptiveLR(self.optimizers["actor"]["shared"])
        # self.optimizer_schedulers["critic"]["arm"] = KLAdaptiveLR(self.optimizers["critic"]["arm"])
        # self.optimizer_schedulers["critic"]["leg"] = KLAdaptiveLR(self.optimizers["critic"]["leg"])


        # Checkpoint Modules
        self.checkpoint_modules["shared"] = {
            "actor": self.shared_actor,
            "actor_optimizer": self.optimizers["actor"]["shared"]
        }

        for uid in self.possible_agents:
            self.checkpoint_modules[uid] = {
                "critic": self.critics[uid],
                "value_standardizer": self.value_standardizers[uid],
                "actor_optimizer": self.optimizers["actor"][uid],
                "critic_optimizer": self.optimizers["critic"][uid]
            }

        # State Space for previled learning & Asyncronous Actor Critic
        if self.is_async_actor_critic:
            self.tensors_names = ["observations", "next_observations",
                                  "states", "next_states",
                                  "actions", "action_log_probs", 
                                  "value_preds", "rewards", 
                                  "truncated", "terminated",
                                  "returns", "advantages"]
            
            self.tensors_name_for_update = ["observations", "states",
                                            "actions", "action_log_probs",
                                            "value_preds", "returns", "advantages"]
        else:
            self.tensors_names = ["observations", "next_observations",
                                  "actions", "action_log_probs", 
                                  "value_preds", "rewards", 
                                  "truncated", "terminated",
                                  "returns", "advantages"]

            self.tensors_name_for_update = ["observations", 
                                            "actions", "action_log_probs",
                                            "value_preds", "returns", "advantages"]
        
        # Shared Info 
        self.shared_infos = False

        if self.shared_infos:
            self.tensors_names.append("shared_infos")
            self.tensors_name_for_update.append("shared_infos")
            for uid in self.possible_agents:
                self.buffer[uid].create_tensor("shared_infos", self.shared_actor.num_shared)


    def save(self, path: str, path_jit: str | None = None):
        modules = {}
        # Shared module
        shared_module = {}
        for name, module in self.checkpoint_modules["shared"].items():
            shared_module[name] = module.state_dict()
        modules["shared"] = shared_module

        # Independent modules
        for uid in self.possible_agents:
            uid_module = {}
            for name, module in self.checkpoint_modules[uid].items():
                uid_module[name] = module.state_dict()

            modules[uid] = uid_module
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(modules, path)


    def load(self, path):
        modules = torch.load(path, map_location=self.device)
        # Shared module
        for name, data in modules["shared"].items():
            module = self.checkpoint_modules["shared"].get(name, None)
            if module is not None:
                module.load_state_dict(data)
            else:
                print(f"Cannot load the {name} module. The agent doesn't have such an instance")

        # Independent modules
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

    
    def act(self, observations, infos, timestep, deterministic = False, update_rms = False):
        """
        Process the environment's observations to make a decision (actions) using the main policy

        Args:
            observations(torch.Tensor): Environment's observations
            infos(dict[str, torch.Tensor]): Environment's additional information for RMA Functionality
            timestep(int): Current timestep
            deterministic(bool): Deterministic action (No Gaussian)
            update_rms(bool): Update a Runningmeanstd distrubution

        Returns:
            actions : RL actions
            log_prob : Log probability of RL actions
            entropy : Entropy of RL actions
        """
        # From Tensor to Dict
        observations = unflatten_tensorized_space(self.observation_space, observations)

        # Shared Action Processing
        non_scaled_actions, log_probs, entropies = self.shared_actor(observations=observations,
                                                                     shared_infos=infos,
                                                                     taken_actions=None,
                                                                     deterministic=deterministic,
                                                                     update_rms=update_rms)
        # From Dict to Tensor
        data = []
        for uid in self.possible_agents:
            actions = non_scaled_actions[uid].clone() * self.action_scale_factor[uid][0]
            data.append((actions, non_scaled_actions[uid], log_probs[uid], entropies[uid]))

        
        actions  = torch.cat([d[0] for d in data], dim=-1) # [B, A]
        nonscaled_actions  = torch.cat([d[1] for d in data], dim=-1) # [B, A]
        log_probs = torch.stack([d[2] for d in data], dim=-1) # [B, A]
        entropy  = torch.stack([d[3] for d in data], dim=-1) # [A]

        return actions, nonscaled_actions, log_probs, entropy
    

    def insert_data(self,
                    observations: torch.Tensor,
                    states: Union[torch.Tensor | None],
                    actions: torch.Tensor,
                    action_log_probs: torch.Tensor,
                    rewards: torch.Tensor,
                    next_observations: torch.Tensor,
                    next_states: Union[torch.Tensor | None],
                    truncated: torch.Tensor,
                    terminated: torch.Tensor,
                    infos: Any) -> None:
        
        """
        Record an environment transition in buffer

        Args:
            observations: observations
            states: states of the environment used to make the decision
            actions: Actions taken by the agent
            rewards: Instant rewards achieved by the current actions
            next_observations: Next observations of the environment
            next_states: Next states of the environment
            done: Signals to indicate that episodes have done
            infos: Additional information about the environment
        """
        # Unflatten from Tensor to Dict
        observations = unflatten_tensorized_space(self.observation_space, observations)
        next_observations = unflatten_tensorized_space(self.observation_space, next_observations)
        actions = unflatten_tensorized_space(self.action_space, actions)
        if states is not None:
            states = unflatten_tensorized_space(self.state_space, states)
            next_states = unflatten_tensorized_space(self.state_space, next_states)
        # Reshape for multi agent scale [B * N, 1] -> [B, N]
        buffer_action_log_probs = action_log_probs.view(-1, self.num_agents).clone()
        buffer_rewards = rewards.view(-1, self.num_agents).clone()
        
        critic_inputs = states if states is not None else observations


        for i, uid in enumerate(self.possible_agents):
            with torch.no_grad():
                value_preds, _, _ = self.critics[uid](critic_inputs[uid]) # [E, 1]
                value_preds = self.value_standardizers[uid].destandardize(value_preds)
                
            # time-limit (truncation) bootstrapping
            if self.time_limit_bootstrap:
                buffer_rewards[:, i:i+1] += self.discount_factor * value_preds * truncated # [E, 1]

            if self.shared_infos:
                # Additional Info Extraction
                shared_infos = infos.get("shared_infos")

            if self.is_async_actor_critic:
                if self.shared_infos:
                    self.buffer[uid].add_samples(observations=observations[uid],
                                                states=states[uid],
                                                actions=actions[uid],
                                                rewards=buffer_rewards[:, i].unsqueeze(-1),
                                                next_observations=next_observations[uid],
                                                next_states=next_states[uid],
                                                truncated=truncated,
                                                terminated=terminated,
                                                action_log_probs=buffer_action_log_probs[:, i].unsqueeze(-1),
                                                value_preds=value_preds,
                                                shared_infos=shared_infos)
                else:
                    self.buffer[uid].add_samples(observations=observations[uid],
                                                states=states[uid],
                                                actions=actions[uid],
                                                rewards=buffer_rewards[:, i].unsqueeze(-1),
                                                next_observations=next_observations[uid],
                                                next_states=next_states[uid],
                                                truncated=truncated,
                                                terminated=terminated,
                                                action_log_probs=buffer_action_log_probs[:, i].unsqueeze(-1),
                                                value_preds=value_preds)
            else:
                if self.shared_infos:
                     self.buffer[uid].add_samples(observations=observations[uid],
                                                  actions=actions[uid],
                                                  rewards=buffer_rewards[:, i].unsqueeze(-1),
                                                  next_observations=next_observations[uid],
                                                  truncated=truncated,
                                                  terminated=terminated,
                                                  action_log_probs=buffer_action_log_probs[:, i].unsqueeze(-1),
                                                  value_preds=value_preds,
                                                  shared_infos=shared_infos)
                else:
                    self.buffer[uid].add_samples(observations=observations[uid],
                                                actions=actions[uid],
                                                rewards=buffer_rewards[:, i].unsqueeze(-1),
                                                next_observations=next_observations[uid],
                                                truncated=truncated,
                                                terminated=terminated,
                                                action_log_probs=buffer_action_log_probs[:, i].unsqueeze(-1),
                                                value_preds=value_preds)


    def update(self) -> float:
        """
        Algorithm's main update step
        """
        for uid in self.possible_agents:
            with torch.no_grad():
                critic_input = self.buffer[uid].get_tensor_by_name("next_states")[-1] if self.is_async_actor_critic else self.buffer[uid].get_tensor_by_name("next_observations")[-1]
                last_values, _, _ = self.critics[uid](critic_input)
                last_values = self.value_standardizers[uid].destandardize(last_values)
                    
        
            # GAE Calculation
            self.buffer[uid].compute_gae(last_values, self.discount_factor, self.gae_lambda)

            # Value Standardization
            value_preds = self.value_standardizers[uid].standardize(self.buffer[uid].get_tensor_by_name("value_preds").reshape(-1, 1), update=True)
            returns = self.value_standardizers[uid].standardize(self.buffer[uid].get_tensor_by_name("returns").reshape(-1, 1), update=True)
            self.buffer[uid].set_tensor_by_name("value_preds", value_preds.reshape(self.buffer[uid].buffer_size, -1, 1))
            self.buffer[uid].set_tensor_by_name("returns", returns.reshape(self.buffer[uid].buffer_size, -1, 1))

            self.set_running_mode(mode="train", uid=uid)
        
        # ==== Cooperative Training Pipeline ====

        # Loss initialization
        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # Parameter Update
        kl_divergences = {
            "arm": [],
            "leg": []
        }
        
        # Sample mini batch (Indexing Sharing)
        indexes = torch.randperm(len(self.buffer["arm"]), dtype=torch.long)

        # Sampling Proactive-Reactive Mini Batch
        mini_batches_a = self.buffer["arm"].sample_by_index(
            names=self.tensors_name_for_update,
            indexes=indexes,
            mini_batches=self.mini_batches)
        
        mini_batches_l = self.buffer["leg"].sample_by_index(
            names=self.tensors_name_for_update,
            indexes=indexes,
            mini_batches=self.mini_batches
        )

    
        for epoch in range(self.learning_epochs):
            for mb_a, mb_l in zip(mini_batches_a, mini_batches_l):
                # (mini batch size, Data-specific)
                if self.is_async_actor_critic:
                    if self.shared_infos:
                        (sampled_observations_a,
                        sampled_states_a,
                        sampled_actions_a,
                        sampled_action_log_probs_a,
                        sampled_value_preds_a,
                        sampled_returns_a,
                        sampled_advantages_a,
                        sampled_infos_a) = mb_a
                        
                        (sampled_observations_l,
                        sampled_states_l,
                        sampled_actions_l,
                        sampled_action_log_probs_l,
                        sampled_value_preds_l,
                        sampled_returns_l,
                        sampled_advantages_l,
                        sampled_infos_l) = mb_l

                        shared_info = sampled_infos_a 

                    else:
                        (sampled_observations_a,
                        sampled_states_a,
                        sampled_actions_a,
                        sampled_action_log_probs_a,
                        sampled_value_preds_a,
                        sampled_returns_a,
                        sampled_advantages_a) = mb_a
                        
                        (sampled_observations_l,
                        sampled_states_l,
                        sampled_actions_l,
                        sampled_action_log_probs_l,
                        sampled_value_preds_l,
                        sampled_returns_l,
                        sampled_advantages_l) = mb_l

                        shared_info = {}

                    actor_input = {
                        "arm": sampled_observations_a,
                        "leg": sampled_observations_l
                    }
                    
                    critic_input = {
                        "arm": sampled_states_a,
                        "leg": sampled_states_l
                    }  
                
                else:
                    if self.shared_infos:
                        (sampled_observations_a,
                        sampled_actions_a,
                        sampled_action_log_probs_a,
                        sampled_value_preds_a,
                        sampled_returns_a,
                        sampled_advantages_a,
                        sampled_infos_a) = mb_a

                        (sampled_observations_l,
                        sampled_actions_l,
                        sampled_action_log_probs_l,
                        sampled_value_preds_l,
                        sampled_returns_l,
                        sampled_advantages_l,
                        sampled_infos_l) = mb_l

                        shared_info = sampled_infos_a

                    else:
                        (sampled_observations_a,
                        sampled_actions_a,
                        sampled_action_log_probs_a,
                        sampled_value_preds_a,
                        sampled_returns_a,
                        sampled_advantages_a) = mb_a

                        (sampled_observations_l,
                        sampled_actions_l,
                        sampled_action_log_probs_l,
                        sampled_value_preds_l,
                        sampled_returns_l,
                        sampled_advantages_l) = mb_l

                        shared_info = {}
            
                    actor_input = {
                        "arm": sampled_observations_a,
                        "leg": sampled_observations_l
                    }
                    
                    critic_input = {
                        "arm": sampled_observations_a,
                        "leg": sampled_observations_l
                    }

                sampled_actions = {
                    "arm": sampled_actions_a,
                    "leg": sampled_actions_l
                }
            
                sampled_action_log_probs = {
                    "arm": sampled_action_log_probs_a,
                    "leg": sampled_action_log_probs_l
                }
            
                sampled_value_preds = {
                    "arm": sampled_value_preds_a,
                    "leg": sampled_value_preds_l,
                }

                sampled_returns = {
                    "arm": sampled_returns_a,
                    "leg": sampled_returns_l
                }

                sampled_advantages = {
                    "arm": sampled_advantages_a,
                    "leg": sampled_advantages_l
                }


                _, new_log_probs, new_entropy = self.shared_actor(observations=actor_input,
                                                                  shared_infos=shared_info,
                                                                  taken_actions=sampled_actions,
                                                                  deterministic=False,
                                                                  update_rms = not epoch)
                # Loss calculation
                policy_losses = {}
                value_losses = {}
                entropy_losses = {}
                for uid in self.possible_agents:
                    # Shape syncronization
                    if len(new_log_probs[uid].shape) == 1:
                        new_log_probs[uid] = new_log_probs[uid].unsqueeze(-1)

                    # Compute approximate KL divergence
                    with torch.no_grad():
                        ratio = new_log_probs[uid] - sampled_action_log_probs[uid]
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences[uid].append(kl_divergence)
                    
                    # Compute entropy loss
                    if self.entropy_loss_scale:
                        entropy_losses[uid] = -self.entropy_loss_scale * new_entropy[uid]
                    else:
                        entropy_losses[uid] = 0

                    # Compute Policy loss
                    ratio = torch.exp(new_log_probs[uid] - sampled_action_log_probs[uid])
                    surrogate = sampled_advantages[uid] * ratio
                    surrogate_clipped = sampled_advantages[uid] * torch.clip(
                        ratio, 1.0 - self.ratio_clip, 1.0 + self.ratio_clip)
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # Compute value loss
                    predicted_values, _, _ = self.critics[uid](critic_input[uid], update_rms=not epoch)
                    if self.clip_predicted_values:
                        predicted_values = sampled_value_preds[uid] + torch.clip(
                            predicted_values - sampled_value_preds[uid], min=-self.value_clip, max=self.value_clip
                        )
                    value_loss = self.value_loss_scale * F.mse_loss(sampled_returns[uid], predicted_values)

                    # Store Loss Info
                    policy_losses[uid] = policy_loss
                    value_losses[uid] = value_loss

                # Parameter setting
                arm_params = []
                leg_params = []

                if hasattr(self.shared_actor, "encoder"):
                    arm_params += list(self.shared_actor.encoder["arm"].parameters())
                    leg_params += list(self.shared_actor.encoder["leg"].parameters())

                if hasattr(self.shared_actor, "head"):
                    arm_params += list(self.shared_actor.head["arm"].parameters())
                    leg_params += list(self.shared_actor.head["leg"].parameters())

                arm_params += [self.shared_actor.log_std_parameter["arm"]]
                leg_params += [self.shared_actor.log_std_parameter["leg"]]

                shared_params = list(self.shared_actor.shared_backbone.parameters())

                # Total Loss
                policy_total = (policy_losses["arm"] + entropy_losses["arm"]) + \
                               (policy_losses["leg"] + entropy_losses["leg"])

                value_total  = value_losses["arm"] + value_losses["leg"]

                # Optimizer Step
                self.optimizers["actor"]["arm"].zero_grad(set_to_none=True)
                self.optimizers["actor"]["leg"].zero_grad(set_to_none=True)
                self.optimizers["actor"]["shared"].zero_grad(set_to_none=True)
                self.optimizers["critic"]["arm"].zero_grad(set_to_none=True)
                self.optimizers["critic"]["leg"].zero_grad(set_to_none=True)

                # Back propagation
                policy_total.backward()
                value_total.backward()

                # Grad Clipping
                if self.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(arm_params, self.grad_norm_clip)
                    nn.utils.clip_grad_norm_(leg_params, self.grad_norm_clip)
                    nn.utils.clip_grad_norm_(shared_params, self.grad_norm_clip)
                    nn.utils.clip_grad_norm_(self.critics["arm"].parameters(), self.grad_norm_clip)
                    nn.utils.clip_grad_norm_(self.critics["leg"].parameters(), self.grad_norm_clip)

                # Step
                self.optimizers["actor"]["arm"].step()
                self.optimizers["actor"]["leg"].step()
                self.optimizers["actor"]["shared"].step()
                self.optimizers["critic"]["arm"].step()
                self.optimizers["critic"]["leg"].step()

                # Update cumulative losses
                cumulative_policy_loss += policy_total.item()
                cumulative_value_loss += value_total.item()
                if self.entropy_loss_scale:
                    cumulative_entropy_loss += (entropy_losses["arm"] + entropy_losses["leg"]).item()

        self.set_running_mode("eval")
        mean_policy_loss = cumulative_policy_loss / (self.learning_epochs * self.mini_batches * self.num_agents)
        mean_value_loss = cumulative_value_loss / (self.learning_epochs * self.mini_batches * self.num_agents)
        mean_entropy_loss = cumulative_entropy_loss / (self.learning_epochs * self.mini_batches * self.num_agents)
        mean_kl_divergence = (sum(kl_divergences["arm"]) + sum(kl_divergences["leg"])).item() / (self.learning_epochs * self.mini_batches * self.num_agents)
                
        return mean_policy_loss, mean_value_loss, mean_entropy_loss, mean_kl_divergence