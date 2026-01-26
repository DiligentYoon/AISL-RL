from typing import List, Optional, Tuple, Union

import csv
import datetime
import functools
import operator
import os
import gymnasium

import numpy as np
import torch

from lib.utils.space_utils import compute_space_size


class Buffer:
    def __init__(
        self,
        buffer_size: int = 1,
        num_envs: int = 1,
        device: Optional[Union[str, torch.device]] = None) -> None:
        """Base class representing a buffer with circular buffers

        Buffers are torch tensors with shape (memory size, number of environments, data size).
        Circular buffers are implemented with two integers: a memory index and an environment index

        Args:
            buffer_size: Maximum number of elements in the first dimension of each internal storage
            num_envs: Number of parallel environments (default: ``1``)
            device: Device on which a tensor/array is or will be allocated (default: ``None``).
                    If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        """
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = torch.device(device)

        # internal variables
        self.filled = False
        self.env_index = 0
        self.memory_index = 0

        self.tensors = {}
        self.tensors_view = {}
        self.tensors_keep_dimensions = {}

        self.sampling_indexes = None


    def __len__(self) -> int:
        """Compute and return the current (valid) size of the memory

        The valid size is calculated as the ``buffer_size * num_envs`` if the memory is full (filled).
        Otherwise, the ``memory_index * num_envs + env_index`` is returned
        Args:

        Returns:
            Valid size
        """
        return self.buffer_size * self.num_envs if self.filled else self.memory_index * self.num_envs + self.env_index
    

    def set_tensor_by_name(self, name: str, tensor: torch.Tensor) -> None:
        """Set a tensor by its name

        Args:
            name: Name of the tensor to set
            tensor: Tensor to set

        Raises:
            KeyError: The tensor does not exist
        """
        with torch.no_grad():
            self.tensors[name].copy_(tensor)


    def get_tensor_by_name(self, name: str, keepdim: bool = True) -> torch.Tensor:
        """Get a tensor by its name

        Args:
            name: Name of the tensor to retrieve
            keepdim: Keep the tensor's shape (memory size, number of environments, size) (default: ``True``)
                            If False, the returned tensor will have a shape of (memory size * number of environments, size)
        Returns:
            return: Tensor
        """
        return self.tensors[name] if keepdim else self.tensors_view[name]


    def create_tensor(
        self,
        name: str,
        size: Union[int, Tuple[int], gymnasium.Space],
        dtype: Optional[torch.dtype] = None,
        keep_dimensions: bool = False,
    ) -> bool:
        """Create a new internal tensor in memory

        The tensor will have a 3-components shape (memory size, number of environments, size).
        The internal representation will use _tensor_<name> as the name of the class property

        Args:
            name: Tensor name (the name has to follow the python PEP 8 style)
            size: Number of elements in the last dimension (effective data size).
                The product of the elements will be computed for sequences or gymnasium spaces
            dtype: Data type (torch.dtype) (default: ``None``).
                        If None, the global default torch data type will be used
            keep_dimensions: Whether or not to keep the dimensions defined through the size parameter (default: ``False``)

        Raises:
            ValueError: The tensor name exists already but the size or dtype are different

        Returns:
            return: True if the tensor was created, otherwise False
        """
        # compute data size
        if not keep_dimensions:
            size = compute_space_size(size, occupied_size=True)
        # check dtype and size if the tensor exists
        if name in self.tensors:
            tensor = self.tensors[name]
            if tensor.size(-1) != size:
                raise ValueError(f"Size of tensor {name} ({size}) doesn't match the existing one ({tensor.size(-1)})")
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(f"Dtype of tensor {name} ({dtype}) doesn't match the existing one ({tensor.dtype})")
            return False
        # define tensor shape
        tensor_shape = (
            (self.buffer_size, self.num_envs, *size) if keep_dimensions else (self.buffer_size, self.num_envs, size)
        )
        view_shape = (-1, *size) if keep_dimensions else (-1, size)
        # create tensor (_tensor_<name>) and add it to the internal storage
        setattr(self, f"_tensor_{name}", torch.zeros(tensor_shape, device=self.device, dtype=dtype))
        # update internal variables
        self.tensors[name] = getattr(self, f"_tensor_{name}")
        self.tensors_view[name] = self.tensors[name].view(*view_shape)
        self.tensors_keep_dimensions[name] = keep_dimensions
        # fill the tensors (float tensors) with NaN
        for tensor in self.tensors.values():
            if torch.is_floating_point(tensor):
                tensor.fill_(float("nan"))
        return True


    def reset(self) -> None:
        """Reset the memory by cleaning internal indexes and flags

        Old data will be retained until overwritten, but access through the available methods will not be guaranteed

        Default values of the internal indexes and flags

        - filled: False
        - env_index: 0
        - memory_index: 0
        """
        self.filled = False
        self.env_index = 0
        self.memory_index = 0


    def add_samples(self, **tensors: torch.Tensor) -> None:
        """Record samples in memory

        Samples should be a tensor with 2-components shape (number of environments, data size).
        All tensors must be of the same shape

        According to the number of environments, the following classification is made:

        - number of environments equals num_envs:
          Store the samples and increment the memory index (first index) by one

        - one environment:
          Store a single sample (tensors with one dimension) and increment the environment index (second index) by one

        Args:
            tensors: Sampled data as key-value arguments where the keys are the names of the tensors to be modified.
                     Non-existing tensors will be skipped

        Raises:
            ValueError: No tensors were provided or the tensors have incompatible shapes
        """
        if not tensors:
            raise ValueError(
                "No samples to be recorded in memory. Pass samples as key-value arguments (where key is the tensor name)"
            )

        # dimensions and shapes of the tensors (assume all tensors have the dimensions of the first tensor)
        tmp = tensors.get("observations", tensors[next(iter(tensors))]) 
        dim, shape = tmp.ndim, tmp.shape

        # multi environment
        if dim > 1:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][self.memory_index].copy_(tensor)
            # Memory index
            self.memory_index += 1

        # single environment
        elif dim == 1:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][self.memory_index, self.env_index].copy_(tensor)
            self.env_index += 1
        else:
            raise ValueError(f"Expected shape (number of environments = {self.num_envs}, data size), got {shape}")

        # update indexes and flags for circluar memory structure
        if self.env_index >= self.num_envs:
            # single env-specific
            self.env_index = 0
            self.memory_index += 1
        if self.memory_index >= self.buffer_size:
            # multi env-specific
            self.memory_index = 0
            self.filled = True

    def init_buffer(self, observation_space: gymnasium.Space, state_space: Union[gymnasium.Space | None], action_space: gymnasium.Space) -> None:
        """
        Initialization Buffer for default dataset
        
        Args:
            observation_space: observation space of agent
            action_space: action space of agent
        """
        self.create_tensor("observations", observation_space, dtype=torch.float32)
        self.create_tensor("next_observations", observation_space, dtype=torch.float32)
        self.create_tensor("actions", action_space, dtype=torch.float32)
        self.create_tensor("action_log_probs", 1, dtype=torch.float32)
        self.create_tensor("value_preds", 1, dtype=torch.float32)
        self.create_tensor("rewards", 1, dtype=torch.float32)
        self.create_tensor("terminated", 1, dtype=torch.bool)
        self.create_tensor("truncated", 1, dtype=torch.bool)

        if state_space is not None:
            self.create_tensor("states", state_space, dtype=torch.float32)
            self.create_tensor("next_states", state_space, dtype=torch.float32)

    
    def sample(
        self, names: Tuple[str], mini_batches: int = 1
    ) -> List[List[torch.Tensor]]:
        """Data sampling method to be implemented by the inheriting classes
        Args:
            names: Tensors names from which to obtain the samples
            type names: tuple or list of strings
            mini_batches: Number of mini-batches to sample (default: ``1``)
            type mini_batches: int, optional

        Raises:
            NotImplementedError: The method has not been implemented

        Returns:
            Sampled data from tensors sorted according to their position in the list of names.
            The sampled tensors will have the following shape: (batch size, data size)
            rtype: list of torch.Tensor list
        """
        raise NotImplementedError("The sampling method (.sample()) is not implemented")


    def sample_by_index(
        self, names: Tuple[str], indexes: Union[tuple, np.ndarray, torch.Tensor], mini_batches: int = 1
    ) -> List[List[torch.Tensor]]:
        """Sample data from memory according to their indexes

        :names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :indexes: Indexes used for sampling
        :type indexes: tuple or list, numpy.ndarray or torch.Tensor
        :mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional

        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (number of indexes, data size)
        :rtype: list of torch.Tensor list
        """
        batches = np.array_split(indexes, mini_batches)
        return [[self.tensors_view[name][batch] for name in names] for batch in batches]