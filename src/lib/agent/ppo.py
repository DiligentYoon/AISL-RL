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


class PPO(Agent):
    def __init__(self,
                 model: Dict[str, nn.Module],
                 buffer: RolloutBuffer,
                 device: Union[str, torch.device],
                 cfg: Dict) -> None:
        """Proximal Policy Optimization (PPO)

        https://arxiv.org/abs/1707.06347

        Args:
            model: Models used by the agent
            buffer: Memory to storage the transitions.
            device: Device on which a tensor/array is or will be allocated (cuda, cpu).
            cfg: Configuration dictionary

        Raises:
            KeyError: If the models dictionary is missing a required key
        """
        super().__init__(cfg, model, device)

        # models
        self.actor = self.model.get("actor", None)
        self.critic = self.model.get("critic", None)
        
        # buffer
        self.buffer = buffer

        # checkpoint models
        self.checkpoint_modules["actor"] = self.actor
        self.checkpoint_modules["critic"] = self.critic

        # configuration
        self.learning_epochs = self.cfg["learning_epochs"]
        self.mini_batches = self.cfg["mini_batches"]
        self.rollouts = self.cfg["rollouts"]

        self.grad_norm_clip = self.cfg["grad_norm_clip"]
        self.ratio_clip = self.cfg["ratio_clip"]
        self.value_clip = self.cfg["value_clip"]

        self.value_loss_scale = self.cfg["value_loss_scale"]
        self.entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self.learning_rate = self.cfg["learning_rate"]

        self.discount_factor = self.cfg["discount_factor"]
        self.gae_lambda = self.cfg["lambda"]
        self.time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        self.random_timesteps = self.cfg["random_timesteps"]
        self.learning_starts = self.cfg["learning_starts"]

        # set up optimizer and learning rate scheduler
        if self.actor is not None and self.critic is not None:
            self.optimizer = torch.optim.Adam(
                    itertools.chain(self.actor.parameters(), self.critic.parameters()), lr=self.learning_rate)
            self.checkpoint_modules["optimizer"] = self.optimizer


        self.tensors_names = ["states", "next_states", "actions", "action_log_probs", 
                              "value_preds", "rewards", "truncated", "terminated"
                              "returns", "advantages"]
        
        self.tensors_name_for_update = ["states", "actions", "action_log_probs",
                                        "value_preds", "returns", "advantages"]

        # Default Mode : Evaluation for disconecting gradient flow
        self.set_running_mode("eval")
        
    
    def act(self, states: torch.Tensor, timestep: int, deterministic: bool = False) -> torch.Tensor:
        """
        Process the environment's states to make a decision (actions) using the main policy

        Args:
            states: Environment's states
            timestep: Current timestep

        Returns:
            actions : RL actions
            log_prob : Log probability of RL actions
            values : Value preidctions
        """
        if timestep < self.random_timesteps:
            actions, log_prob = self.actor.random_act(states)
        else:
            actions, log_prob = self.actor.act(states, deterministic)
        
        return actions, log_prob
    

    def insert_data(self,
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    action_log_probs: torch.Tensor,
                    rewards: torch.Tensor,
                    next_states: torch.Tensor,
                    truncated: torch.Tensor,
                    terminated: torch.Tensor,
                    infos: Any) -> None:
        
        """
        Record an environment transition in buffer
        
        TODO: Buffer revision after PR

        Args:
            states: Observations/states of the environment used to make the decision
            actions: Actions taken by the agent
            rewards: Instant rewards achieved by the current actions
            next_states: Next observations/states of the environment
            done: Signals to indicate that episodes have done
            infos: Additional information about the environment
        """
        with torch.no_grad():
            value_preds = self.critic.act(states)
        
        # time-limit (truncation) bootstrapping
        if self.time_limit_bootstrap:
            rewards += self.discount_factor * value_preds * truncated

        self.buffer.add_sampels(states=states,
                                actions=actions,
                                rewards=rewards,
                                next_states=next_states,
                                truncated=truncated,
                                terminated=terminated,
                                action_log_probs=action_log_probs,
                                value_preds = value_preds)
    
    
    def update(self) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        with torch.no_grad():
            last_values = self.critic.act(self.buffer.get_tensor_by_name("next_states")[-1])
        
        self.buffer.compute_gae(last_values, self.discount_factor, self.gae_lambda)
        
        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        self.set_running_mode("train")
        for epoch in range(self.learning_epochs):
            kl_divergences = []
            # sample mini batch for SGD at each epoch
            mini_batches = self.buffer.sample(
                names=self.tensors_name_for_update,
                batch_size=self.rollouts,
                mini_batch=self.mini_batches)
            
            for mb in mini_batches:
                (sampled_states, 
                 sampled_actions,
                 sampled_action_log_probs,
                 sampled_value_preds,
                 sampled_returns,
                 sampled_advantages) = mb
                
                _, next_log_probs = self.actor.act(sampled_states)

                # compute approximate KL divergence
                with torch.no_grad():
                    ratio = next_log_probs - sampled_action_log_probs
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)
                
                # compute entropy loss
                if self._entropy_loss_scale:
                    entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                else:
                    entropy_loss = 0

                # compute policy loss
                ratio = torch.exp(next_log_probs - sampled_action_log_probs)
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clip(
                    ratio, 1.0 - self.ratio_clip, 1.0 + self.ratio_clip)
                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                # compute value loss
                predicted_values = self.critic.act(sampled_states)
                if self._clip_predicted_values:
                    predicted_values = sampled_value_preds + torch.clip(
                        predicted_values - sampled_value_preds, min=-self.value_clip, max=self.value_clip
                    )
                value_loss = self.value_loss_scale * F.mse_loss(sampled_returns, predicted_values)
                

                # optimization step
                self.optimizer.zero_grad()
                (policy_loss + self.entropy_loss_scale * entropy_loss + self.value_loss_scale * value_loss).backward()

                if self.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.critic.parameters(), self.grad_norm_clip)
                
                self.optimizer.step()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                
        
        mean_policy_loss = cumulative_policy_loss / (self.learning_epochs * self.mini_batches)
        mean_value_loss = cumulative_value_loss / (self.learning_epochs * self.mini_batches)
        mean_entropy_loss = cumulative_entropy_loss / (self.learning_epochs * self.mini_batches)
        mean_kl_divergence = sum(kl_divergences) / (self.learning_epochs * self.mini_batches)

        return mean_policy_loss, mean_value_loss, mean_entropy_loss, mean_kl_divergence
            