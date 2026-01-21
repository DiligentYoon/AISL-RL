
import gymnasium
import torch

from typing import Optional, Union, Tuple, List
from lib.buffer.buffer import Buffer


class RolloutBuffer(Buffer):
    def __init__(
        self,
        buffer_size: int = 1,
        num_envs: int = 1,
        device: Optional[Union[str, torch.device]] = None) -> None:
        """Base class representing a Rolloutbuffer for On-Policy Algorithm (PPO)
        
        Args:
            buffer_size: Maximum number of elements in the first dimension of each internal storage
            num_envs: Number of parallel environments (default: ``1``)
            device: Device on which a tensor/array is or will be allocated (default: ``None``).
                    If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        """
        super().__init__(buffer_size, num_envs, device)


    def init_buffer(self, observation_space:gymnasium.Space, action_space:gymnasium.Space):
        """
        Initialization Rollout Buffer for On-Policy Setting including GAE Buffer

        Args:
            observation_space: observation space of agent
            action_space: action space of agent
        """
        super().init_buffer(observation_space, action_space)
        # GAE Buffer
        self.create_tensor("returns", 1, dtype=torch.float32)
        self.create_tensor("advantages", 1, dtype=torch.float32)


    def sample(self, names:Tuple[str], batch_size: int, mini_batch: int) -> List[List[torch.Tensor]]:
        """Random sampling from Stochastic Gradient Descent in On-policy Algorithm
        
        Args:
            names: Tensors names from which to obtain the samples
            batch_size: Number of element to sample
            mini_batches: Number of mini-batches to sample

        Returns:
            Sampled data from tensors sorted according to their position in the list of names.
            The sampled tensors will have the following shape: (batch size, data size)
            rtype: list of torch.Tensor list
        """
        # compute valid memory size
        size = len(self)


        # generate random samples
        indexes = torch.randperm(size, dtype=torch.long)[:batch_size]

        self.sampling_indexes = indexes
        return self.sample_by_index(names=names, indexes=indexes, mini_batches=mini_batch)


    def compute_gae(self, next_value: torch.Tensor, gamma: float, lamb: float):
        """
        Compute Generalized Advantage Expectation (GAE)
        
        Args:
            next_value: Bootstrapping value at final step in rollouts
            gamma: Discount Factor
            lamb : GAE Forgetting Factor
        """
        return_tensor = torch.zeros_like(self.tensors['returns'], dtype=torch.float32, device=self.device)
        advantage_tensor = torch.zeros_like(self.tensors['advantages'], dtype=torch.float32, device=self.device)
        rollout = self.buffer_size

        reward = self.tensors["rewards"] # [T, E, 1]
        value_preds = self.tensors["value_preds"] # [T, E, 1]
        mask = torch.logical_not(self.tensors["done"]).float() # [T, E, 1]

        gae = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        for step in reversed(range(rollout)):
            if step == (rollout-1):
                value_pred = next_value
            else:
                value_pred = value_preds[step + 1]

            # Predicted value : Syncronization
            delta = reward[step] + gamma * value_pred * mask[step] - value_preds[step]
            gae = delta + gamma * lamb * mask[step] * gae

            advantage_tensor[step] = gae
            return_tensor[step] = gae + value_preds[step]
            
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std() + 1e-8)
        
        self.set_tensor_by_name('returns', return_tensor)
        self.set_tensor_by_name('advantages', advantage_tensor)
        



    