from typing import Any, Union, Dict, Sequence, Mapping

import copy
import itertools
import gymnasium as gym
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.agent.mappo import MAPPO
from lib.buffer.rolloutbuffer import RolloutBuffer

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
        """Cooperative Multi Agent Proximal Policy Optimization (C-MAPPO)

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

        self.proactive_head_id = cfg.get('proactive', None)
        self.reactive_head_id = cfg.get('reactive', None)

        if self.proactive_head_id is None or self.reactive_head_id is None:
            raise RuntimeError("Proactive and Reactive relationship should be assigned for this network.")

    
    def act(self, observations, timestep, deterministic = False, update_rms = False):
        """
        Process the environment's observations to make a decision (actions) using the main policy

        >>> (Each Encoder) ---> (Proactive Head) ---> (Reactive Head)
        >>> e.g, Leg & Arm Encoder ---> Leg Head ---> Arm Head

        Args:
            observations(torch.Tensor): Environment's observations
            timestep(int): Current timestep
            deterministic(bool): Deterministic action (No Gaussian)
            update_rms(bool): Update a Runningmeanstd distrubution

        Returns:
            actions : RL actions
            log_prob : Log probability of RL actions
            entropy : Entropy of RL actions
        """
        p_id = self.proactive_head_id
        r_id = self.reactive_head_id
        # From Tensor to Dict
        observations = unflatten_tensorized_space(self.observation_space, observations)

        # Proactive action Processing
        p_actions, p_log_probs, p_entropy = self.actors[p_id](observations=observations[p_id],
                                                              taken_actions=None,
                                                              deterministic=deterministic, 
                                                              update_rms=update_rms)
        
        # Reactive action Processing
        r_actions, r_log_probs, r_entropy = self.actors[r_id](observations=torch.cat([observations[r_id], p_actions], dim=-1),
                                                              taken_actions=None,
                                                              deterministic=deterministic,
                                                              update_rms=update_rms)
        
        actions = torch.cat([p_actions, r_actions], dim=-1)
        log_probs = torch.stack([p_log_probs, r_log_probs], dim=-1)
        entropy = torch.stack([p_entropy, r_entropy], dim=-1)

        return actions, log_probs, entropy
    

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
        action_log_probs = action_log_probs.view(-1, self.num_agents)
        rewards = rewards.view(-1, self.num_agents)
        

        critic_inputs = states if states is not None else observations

        for i, uid in enumerate(self.possible_agents):
            with torch.no_grad():
                if uid == self.reactive_head_id:
                    value_preds, _, _ = self.critics[uid](torch.cat([critic_inputs[uid], actions[self.proactive_head_id]], dim=-1)) # [E, 1]
                else:
                    value_preds, _, _ = self.critics[uid](critic_inputs[uid]) # [E, 1]
                
            # time-limit (truncation) bootstrapping
            if self.time_limit_bootstrap:
                rewards[:, i:i+1] += self.discount_factor * value_preds * truncated # [E, 1]

            if self.is_async_actor_critic:
                self.buffer[uid].add_samples(observations=observations[uid],
                                             states=states[uid],
                                             actions=actions[uid],
                                             rewards=rewards[:, i].unsqueeze(-1),
                                             next_observations=next_observations[uid],
                                             next_states=next_states[uid],
                                             truncated=truncated,
                                             terminated=terminated,
                                             action_log_probs=action_log_probs[:, i].unsqueeze(-1),
                                             value_preds = value_preds)
            else:
                self.buffer[uid].add_samples(observations=observations[uid],
                                             actions=actions[uid],
                                             rewards=rewards[:, i].unsqueeze(-1),
                                             next_observations=next_observations[uid],
                                             truncated=truncated,
                                             terminated=terminated,
                                             action_log_probs=action_log_probs[:, i].unsqueeze(-1),
                                             value_preds = value_preds)


    def update(self) -> float:
        """
        Algorithm's main update step
        """
        for uid in self.possible_agents:
            with torch.no_grad():
                critic_input = self.buffer[uid].get_tensor_by_name("next_states")[-1] if self.is_async_actor_critic else self.buffer[uid].get_tensor_by_name("next_observations")[-1]
                if uid == self.reactive_head_id:
                    last_proactive_action = self.buffer[self.proactive_head_id].get_tensor_by_name("actions")[-1]
                    last_values, _, _ = self.critics[uid](torch.cat([critic_input, last_proactive_action], dim=-1))
                else:
                    last_values, _, _ = self.critics[uid](critic_input)
                    
        
            # GAE Calculation
            self.buffer[uid].compute_gae(last_values, self.discount_factor, self.gae_lambda)

            # Value Standardization
            returns = self.value_standardizers[uid].standardize(self.buffer[uid].get_tensor_by_name("returns").reshape(-1, 1), update=True)
            value_preds = self.value_standardizers[uid].standardize(self.buffer[uid].get_tensor_by_name("value_preds").reshape(-1, 1))
            self.buffer[uid].set_tensor_by_name("returns", returns.reshape(self.buffer[uid].buffer_size, -1, 1))
            self.buffer[uid].set_tensor_by_name("value_preds", value_preds.reshape(self.buffer[uid].buffer_size, -1, 1))

            self.set_running_mode(mode="train", uid=uid)
        
        # ==== Cooperative Training Pipeline ====

        # Id Definition
        p_id = self.proactive_head_id
        r_id = self.reactive_head_id

        # Loss initialization
        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # Parameter Update
        kl_divergences = []

        # Sample mini batch (Indexing Sharing)
        indexes = torch.randperm(len(self.buffer[p_id]), dtype=torch.long)

        # Sampling Proactive-Reactive Mini Batch
        mini_batches_p = self.buffer[p_id].sample_by_index(
            names=self.tensors_name_for_update,
            indexes=indexes,
            mini_batches=self.mini_batches)
        
        mini_batches_r = self.buffer[r_id].sample_by_index(
            names=self.tensors_name_for_update,
            indexes=indexes,
            mini_batches=self.mini_batches
        )
    
        for epoch in range(self.learning_epochs):
            for mb_p, mb_r in zip(mini_batches_p, mini_batches_r):
                # (mini batch size, Data-specific)
                if self.is_async_actor_critic:
                    (sampled_observations_p,
                     sampled_states_p,
                     sampled_actions_p,
                     sampled_action_log_probs_p,
                     sampled_value_preds_p,
                     sampled_returns_p,
                     sampled_advantages_p) = mb_p
                    
                    (sampled_observations_r,
                     sampled_states_r,
                     sampled_actions_r,
                     sampled_action_log_probs_r,
                     sampled_value_preds_r,
                     sampled_returns_r,
                     sampled_advantages_r) = mb_r

                    actor_input_p = sampled_observations_p
                    critic_input_p = sampled_states_p        

                    actor_input_r = sampled_observations_r
                    critic_input_r = sampled_states_r        
                
                else:
                    (sampled_observations_p,
                    sampled_actions_p,
                    sampled_action_log_probs_p,
                    sampled_value_preds_p,
                    sampled_returns_p,
                    sampled_advantages_p) = mb_p


                    (sampled_observations_r,
                    sampled_actions_r,
                    sampled_action_log_probs_r,
                    sampled_value_preds_r,
                    sampled_returns_r,
                    sampled_advantages_r) = mb_r
                    
                    actor_input_p = sampled_observations_p
                    critic_input_p = sampled_observations_p
            
                    actor_input_r = sampled_observations_r
                    critic_input_r = sampled_observations_r

                
                # ==== Proactive Network Update ====

                # Policy
                new_actions_p, new_log_probs_p, new_entropy_p = self.actors[p_id](observations=actor_input_p,
                                                                                  taken_actions=sampled_actions_p,
                                                                                  deterministic=False,
                                                                                  update_rms=not epoch)
                
                _, new_log_probs_r, new_entropy_r = self.actors[r_id](observations=torch.cat([actor_input_r, new_actions_p], dim=-1),
                                                                      taken_actions=sampled_actions_r,
                                                                      deterministic=False,
                                                                      update_rms=False)
                
                # Shape syncronization
                if len(new_log_probs_p.shape) != len(sampled_action_log_probs_p.shape):
                    new_log_probs_p = new_log_probs_p.reshape(sampled_action_log_probs_p.shape)
                    new_log_probs_r = new_log_probs_r.reshape(sampled_action_log_probs_r.shape)

                # Compute approximate KL divergence
                with torch.no_grad():
                    ratio = new_log_probs_p - sampled_action_log_probs_p
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)
                
                # Compute entropy loss
                if self.entropy_loss_scale:
                    entropy_loss_p = -self.entropy_loss_scale * new_entropy_p
                else:
                    entropy_loss_p = 0

                # Compute policy loss (L_{proactive}, L_{reactive})
                ratio_p = torch.exp(new_log_probs_p - sampled_action_log_probs_p)
                surrogate_p = sampled_advantages_p * ratio_p
                surrogate_clipped_p = sampled_advantages_p * torch.clip(
                    ratio_p, 1.0 - self.ratio_clip, 1.0 + self.ratio_clip)
                
                ratio_r = torch.exp(new_log_probs_r - sampled_action_log_probs_r)
                surrogate_r = sampled_advantages_r * ratio_r
                surrogate_clipped_r = sampled_advantages_r * torch.clip(
                    ratio_r, 1.0 - self.ratio_clip, 1.0 + self.ratio_clip)
                
                policy_loss_p = -torch.min(surrogate_p, surrogate_clipped_p).mean()
                policy_loss_r = -torch.min(surrogate_r, surrogate_clipped_r).mean()

                total_policy_loss_p = policy_loss_p + policy_loss_r

                # Value
                predicted_values_p, _, _ = self.critics[p_id](critic_input_p, update_rms=not epoch)
                predicted_values_p = self.value_standardizers[p_id].standardize(predicted_values_p)
                if self.clip_predicted_values:
                    predicted_values_p = sampled_value_preds_p + torch.clip(
                        predicted_values_p - sampled_value_preds_p, min=-self.value_clip, max=self.value_clip
                    )
                value_loss_p = self.value_loss_scale * F.mse_loss(sampled_returns_p, predicted_values_p)

                # Optimizer Step
                self.optimizers[p_id].zero_grad()
                (total_policy_loss_p + value_loss_p + entropy_loss_p).backward()

                if self.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(itertools.chain(self.actors[p_id].parameters(), self.critics[p_id].parameters()), self.grad_norm_clip)
                
                self.optimizers[p_id].step()


                # ==== Reactive Network Update ====

                # Policy
                self.optimizers[r_id].zero_grad()

                _, new_log_probs_r, new_entropy_r = self.actors[r_id](observations=torch.cat([actor_input_r, sampled_actions_p], dim=-1),
                                                                      taken_actions=sampled_actions_r,
                                                                      deterministic=False,
                                                                      update_rms=not epoch)
                
                # Shape syncronization
                if len(new_log_probs_r.shape) != len(sampled_action_log_probs_r.shape):
                    new_log_probs_r = new_log_probs_r.reshape(sampled_action_log_probs_r.shape)

                # Compute approximate KL divergence
                with torch.no_grad():
                    ratio = new_log_probs_r - sampled_action_log_probs_r
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)

                # Compute entropy loss
                if self.entropy_loss_scale:
                    entropy_loss_r = -self.entropy_loss_scale * new_entropy_r
                else:
                    entropy_loss_r = 0
                
                ratio_r = torch.exp(new_log_probs_r - sampled_action_log_probs_r)
                surrogate_r = sampled_advantages_r * ratio_r
                surrogate_clipped_r = sampled_advantages_r * torch.clip(
                    ratio_r, 1.0 - self.ratio_clip, 1.0 + self.ratio_clip)
                
                total_policy_loss_r = -torch.min(surrogate_r, surrogate_clipped_r).mean()

                # Value
                predicted_values_r, _, _ = self.critics[r_id](torch.cat([critic_input_r, sampled_actions_p], dim=-1), update_rms=not epoch)
                predicted_values_r = self.value_standardizers[r_id].standardize(predicted_values_r)
                if self.clip_predicted_values:
                    predicted_values_r = sampled_value_preds_r + torch.clip(
                        predicted_values_r - sampled_value_preds_r, min=-self.value_clip, max=self.value_clip
                    )
                value_loss_r = self.value_loss_scale * F.mse_loss(sampled_returns_r, predicted_values_r)

                # Optimizer Step
                (total_policy_loss_r + value_loss_r + entropy_loss_r).backward()

                if self.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(itertools.chain(self.actors[r_id].parameters(), self.critics[r_id].parameters()), self.grad_norm_clip)

                self.optimizers[r_id].step()


                # Update cumulative losses
                cumulative_policy_loss += (total_policy_loss_p.item() + total_policy_loss_r.item())
                cumulative_value_loss  += (value_loss_p.item() + value_loss_r.item())
                if self.entropy_loss_scale:
                    cumulative_entropy_loss += (entropy_loss_p.item() + entropy_loss_r.item())
        
        self.set_running_mode("eval")
        mean_policy_loss = cumulative_policy_loss / (self.learning_epochs * self.mini_batches * self.num_agents)
        mean_value_loss = cumulative_value_loss / (self.learning_epochs * self.mini_batches * self.num_agents)
        mean_entropy_loss = cumulative_entropy_loss / (self.learning_epochs * self.mini_batches * self.num_agents)
        mean_kl_divergence = sum(kl_divergences) / (self.learning_epochs * self.mini_batches * self.num_agents)
                
        return mean_policy_loss, mean_value_loss, mean_entropy_loss, mean_kl_divergence