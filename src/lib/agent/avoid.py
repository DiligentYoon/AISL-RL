from typing import Any, Mapping, Optional, Tuple, Union, Dict

import copy
import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.agent.agent import Agent
from lib.buffer.avoid.replaybuffer import HindSightReplayBuffer

class Avoid(Agent):
    def __init__(self,
                 model: Dict[str, nn.Module],
                 buffer: Optional[HindSightReplayBuffer],
                 device: Union[str, torch.device],
                 cfg: Dict) -> None:
        """Safety value function (Avoid Problem)

        Args:
            model: Models used by the agent
            buffer: Memory to storage the transitions. Only required for training;
                    inference-only users (evaluation, deployment) may pass None.
            device: Device on which a tensor/array is or will be allocated (cuda, cpu).
            cfg: Configuration dictionary
        """
        super().__init__(cfg, model, device)

        # Models
        self.critic = self.model.get("critic", None).to(self.device)
        
        # Buffer
        self.buffer = buffer

        # Checkpoint models
        self.checkpoint_modules["critic"] = self.critic

        # Load parameters form cfg
        self.learning_epochs = self.cfg["learning_epochs"]
        self.batch_size = self.cfg["batch_size"]

        self.learning_rate = self.cfg["learning_rate"]
        self.discount_factor = self.cfg["discount_factor"]

        self.grad_norm_clip = self.cfg["grad_norm_clip"]
        self.learning_stars = self.cfg["learning_starts"]

        # Set up Adam optimizer
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.checkpoint_modules["optimizer"] = self.optimizer


        # State Space for previled learning & Asyncronous Actor Critic
        self.tensors_names = ["states", "next_states", "g_values", 
                              "truncated", "terminated"]
        
        self.tensors_name_for_update = ["states", "next_states", "g_values", 
                                        "truncated", "terminated"]
            
        # Default Mode : Evaluation for disconnecting gradient flow
        self.set_running_mode("eval")
        

    def insert_data(self,
                    infos: Dict[str, torch.Tensor],
                    next_infos: Dict[str, torch.Tensor],
                    terminated: torch.Tensor,
                    truncated: torch.Tensor) -> None:
        """
        Store one transition for RA value learning.

        Expected keys in infos / next_infos:
            "ra_states" : RA state vector s
            "g_values"  : g(s)
        """
        states = infos["ra_states"]
        next_states = next_infos["ra_states"]
        g_values = infos["g_values"]

        # shape normalize: (N,) -> (N, 1)
        if g_values.ndim == 1:
            g_values = g_values.unsqueeze(-1)

        self.buffer.add_samples(states=states,
                                next_states=next_states,
                                g_values=g_values,
                                truncated=truncated,
                                terminated=terminated)


    def _compute_target(self,
                           next_states: torch.Tensor,
                           g_values: torch.Tensor,
                           truncated: torch.Tensor,
                           terminated: torch.Tensor) -> torch.Tensor:
        """
        Compute discounted Reach-Avoid Bellman target:

            y = gamma * max(g(s), V(next)) + (1 - gamma) * g(s)
        """
        with torch.no_grad():
            next_values, _, _ = self.critic(next_states, update_rms=False)

            done = torch.logical_or(truncated.bool(), terminated.bool())

            # V(next)=+inf for episode end
            inf_value = torch.full_like(next_values, float("inf"))
            next_values = torch.where(done, inf_value, next_values)

            targets = (1.0 - self.discount_factor) * g_values + self.discount_factor * torch.max(g_values, next_values)

        return targets

    
    def update(self) -> float:
        """
        Main update step for RA value approximation.
        """
        self.set_running_mode("train")

        cumulative_value_loss = 0.0
        num_updates = 0

        for step in range(self.learning_epochs):
            batch = self.buffer.sample_batch(
                names=self.tensors_name_for_update,
                batch_size=self.batch_size
            )

            (sampled_states,
            sampled_next_states,
            sampled_g_values,
            sampled_truncated,
            sampled_terminated) = batch

            # Predict V(s)
            predicted_values, _, _ = self.critic(
                sampled_states,
                update_rms=(step == 0)
            )

            # Compute RA target
            target_values = self._compute_target(
                next_states=sampled_next_states,
                g_values=sampled_g_values,
                truncated=sampled_truncated,
                terminated=sampled_terminated,
            )

            value_loss = F.mse_loss(predicted_values, target_values)

            self.optimizer.zero_grad()
            value_loss.backward()

            if self.grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_clip)

            self.optimizer.step()

            cumulative_value_loss += value_loss.item()
            num_updates += 1

        self.set_running_mode("eval")

        mean_value_loss = cumulative_value_loss / max(num_updates, 1)
        return mean_value_loss
            