from typing import Any, Mapping, Optional, Tuple, Union, Dict

import copy
import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.agent.agent import Agent
from lib.buffer.rolloutbuffer import RolloutBuffer
from lib.utils.Running_mean_std import RunningMeanStd

class PPO(Agent):
    def __init__(self,
                 model: Dict[str, nn.Module],
                 buffer: RolloutBuffer,
                 device: Union[str, torch.device],
                 cfg: Dict,
                 shared: bool = False) -> None:
        """Proximal Policy Optimization (PPO)

        https://arxiv.org/abs/1707.06347

        Args:
            model: Models used by the agent
            buffer: Memory to storage the transitions.
            device: Device on which a tensor/array is or will be allocated (cuda, cpu).
            cfg: Configuration dictionary
            shared: Whether the actor and critic share the specific model components.
        """
        super().__init__(cfg, model, device)

        # Models
        self.shared = shared
        self.actor = self.model.get("actor", None).to(self.device)
        self.critic = self.model.get("critic", None).to(self.device)
        self.value_standardizer = RunningMeanStd(shape=1, device=device)
        
        # Buffer
        self.buffer = buffer

        # Checkpoint models
        self.checkpoint_modules["actor"] = self.actor
        self.checkpoint_modules["critic"] = self.critic
        self.checkpoint_modules["value_standardizer"] = self.value_standardizer

        # Load parameters form cfg
        self.rollouts = self.cfg["rollouts"]
        self.learning_epochs = self.cfg["learning_epochs"]
        self.mini_batches = self.cfg["mini_batches"]

        self.learning_rate = self.cfg["learning_rate"]
        self.discount_factor = self.cfg["discount_factor"]
        self.gae_lambda = self.cfg["lambda"]

        self.random_timesteps = self.cfg["random_timesteps"]
        self.learning_starts = self.cfg["learning_starts"]

        self.grad_norm_clip = self.cfg["grad_norm_clip"]
        self.ratio_clip = self.cfg["ratio_clip"]
        self.value_clip = self.cfg["value_clip"]

        self.entropy_loss_scale = self.cfg["entropy_loss_scale"]
        self.value_loss_scale = self.cfg["value_loss_scale"]

        self.time_limit_bootstrap = self.cfg["time_limit_bootstrap"]
        self.clip_predicted_values = self.cfg["clip_predicted_values"]

        self.is_async_actor_critic = self.cfg.get("async_actor_critic", False)

        self.joint_action_weight = self.cfg["joint_action_weight"]
        self.wheel_action_weight = self.cfg["wheel_action_weight"]

        # Set up Adam optimizer
        if self.actor is not None and self.critic is not None:
            if self.shared:
                self.optimizer = torch.optim.Adam(
                    list(set(self.actor.parameters()).union(set(self.critic.parameters()))), lr=self.learning_rate)
                
            else:
                self.optimizer = torch.optim.Adam(
                        itertools.chain(self.actor.parameters(), self.critic.parameters()), lr=self.learning_rate)
                
            self.checkpoint_modules["optimizer"] = self.optimizer

        self.tensors_names = ["observations", "next_observations", "actions", "action_log_probs", 
                              "value_preds", "rewards", "truncated", "terminated",
                              "returns", "advantages"]
        
        self.tensors_name_for_update = ["observations", "actions", "action_log_probs",
                                        "value_preds", "returns", "advantages"]

        # Default Mode : Evaluation for disconnecting gradient flow
        self.set_running_mode("eval")

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
        
    
    def act(self, observations: torch.Tensor, timestep: int, deterministic: bool = False, update_rms: bool = False) -> torch.Tensor:
        """
        Process the environment's observations to make a decision (actions) using the main policy

        Args:
            observations(torch.Tensor): Environment's observations
            timestep(int): Current timestep
            deterministic(bool): Deterministic action (No Gaussian)
            update_rms(bool): Update a Runningmeanstd distrubution

        Returns:
            actions : RL actions
            log_prob : Log probability of RL actions
            values : Value(Return) preidctions
        """
        # # Standardization               TODO: 이거 필요없을거 같음 지우기 ㄱㄱㄱㄱ
        # standardized_observations = self.actor_standardizer.standardize(observations, update=update_rms)

        if timestep < self.random_timesteps:
            # Random act
            nonscaled_actions, log_prob, entropy = self.actor.random_act(observations)
        else:
            # Normal act
            nonscaled_actions, log_prob, entropy = self.actor(observations=observations,
                                                              taken_actions=None,
                                                              deterministic=deterministic,
                                                              update_rms=update_rms)
        
        # Action scaling
        actions = nonscaled_actions.clone()
        actions[:, :-2] *= self.joint_action_weight
        actions[:, -2:] *= self.wheel_action_weight

        return actions, log_prob, entropy, nonscaled_actions                        # Add raw action which is inserted into buffer

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
        critic_inputs = states if states is not None else observations

        with torch.no_grad():
            value_preds, _, _ = self.critic(critic_inputs)
            value_preds = self.value_standardizer.destandardize(value_preds)
            
        # time-limit (truncation) bootstrapping
        if self.time_limit_bootstrap:
            rewards += self.discount_factor * value_preds * truncated

        if self.is_async_actor_critic:
            self.buffer.add_samples(observations=observations,
                                    states=states,
                                    actions=actions,
                                    rewards=rewards,
                                    next_observations=next_observations,
                                    next_states=next_states,
                                    truncated=truncated,
                                    terminated=terminated,
                                    action_log_probs=action_log_probs,
                                    value_preds = value_preds)
        else:
            self.buffer.add_samples(observations=observations,
                                    actions=actions,
                                    rewards=rewards,
                                    next_observations=next_observations,
                                    truncated=truncated,
                                    terminated=terminated,
                                    action_log_probs=action_log_probs,
                                    value_preds = value_preds)

    
    def update(self) -> float:
        """
        Algorithm's main update step
        """
        with torch.no_grad():
            critic_input = self.buffer.get_tensor_by_name("next_states")[-1] if self.is_async_actor_critic else self.buffer.get_tensor_by_name("next_observations")[-1]
            last_values, _, _ = self.critic(critic_input)
            last_values = self.value_standardizer.destandardize(last_values)
        
        # GAE Calculation
        self.buffer.compute_gae(last_values, self.discount_factor, self.gae_lambda)

        # Value Standardization
        value_preds = self.value_standardizer.standardize(self.buffer.get_tensor_by_name("value_preds").reshape(-1, 1), update=True)
        returns = self.value_standardizer.standardize(self.buffer.get_tensor_by_name("returns").reshape(-1, 1), update=True)
        self.buffer.set_tensor_by_name("value_preds", value_preds.reshape(self.buffer.buffer_size, -1, 1))
        self.buffer.set_tensor_by_name("returns", returns.reshape(self.buffer.buffer_size, -1, 1))
        
        # Loss initialization
        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # Parameter Update
        kl_divergences = []
        self.set_running_mode("train")

        # Sample mini batch
        mini_batches = self.buffer.sample(
            names=self.tensors_name_for_update,
            mini_batch=self.mini_batches)
        
        for epoch in range(self.learning_epochs):
            for mb in mini_batches:
                # (mini batch size, Data-specific)
                if self.is_async_actor_critic:
                    (sampled_observations,
                     sampled_states,
                     sampled_actions,
                     sampled_action_log_probs,
                     sampled_value_preds,
                     sampled_returns,
                     sampled_advantages) = mb

                    actor_input = sampled_observations
                    critic_input = sampled_states                
                
                else:
                    (sampled_observations,
                     sampled_actions,
                     sampled_action_log_probs,
                     sampled_value_preds,
                     sampled_returns,
                     sampled_advantages) = mb
                    
                    actor_input = sampled_observations
                    critic_input = sampled_observations
                
                # State, observation standardization

                _, new_log_probs, dist_entropy = self.actor(observations=actor_input, 
                                                            taken_actions=sampled_actions,
                                                            deterministic=False,
                                                            update_rms=not epoch)

                # Shape syncronization
                if len(new_log_probs.shape) != len(sampled_action_log_probs.shape):
                    new_log_probs = new_log_probs.reshape(sampled_action_log_probs.shape)

                # Compute approximate KL divergence
                with torch.no_grad():
                    ratio = new_log_probs - sampled_action_log_probs
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)
                
                # Compute entropy loss
                if self.entropy_loss_scale:
                    entropy_loss = -self.entropy_loss_scale * dist_entropy
                else:
                    entropy_loss = 0

                # Compute policy loss
                ratio = torch.exp(new_log_probs - sampled_action_log_probs)
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clip(
                    ratio, 1.0 - self.ratio_clip, 1.0 + self.ratio_clip)
                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                # Compute value loss
                predicted_values, _, _ = self.critic(critic_input, update_rms=not epoch)
                # predicted_values = self.value_standardizer.standardize(predicted_values)
                if self.clip_predicted_values:
                    predicted_values = sampled_value_preds + torch.clip(
                        predicted_values - sampled_value_preds, min=-self.value_clip, max=self.value_clip
                    )
                value_loss = self.value_loss_scale * F.mse_loss(sampled_returns, predicted_values)
                
                # Optimization step
                self.optimizer.zero_grad()
                (policy_loss + entropy_loss + value_loss).backward()

                if self.grad_norm_clip > 0:
                    if self.shared:
                        nn.utils.clip_grad_norm_(list(set(self.actor.parameters()).union(set(self.critic.parameters()))), self.grad_norm_clip)

                    else:
                        nn.utils.clip_grad_norm_(itertools.chain(self.actor.parameters(), self.critic.parameters()), self.grad_norm_clip)
                
                self.optimizer.step()

                # Update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                
        self.set_running_mode("eval")

        mean_policy_loss = cumulative_policy_loss / (self.learning_epochs * self.mini_batches)
        mean_value_loss = cumulative_value_loss / (self.learning_epochs * self.mini_batches)
        mean_entropy_loss = cumulative_entropy_loss / (self.learning_epochs * self.mini_batches)
        mean_kl_divergence = sum(kl_divergences) / (self.learning_epochs * self.mini_batches)


        return mean_policy_loss, mean_value_loss, mean_entropy_loss, mean_kl_divergence
            