# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch


from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

import torch

##
# Noise as functions.
##

def constant_noise(data: torch.Tensor, cfg: ConstantNoiseCfg) -> torch.Tensor:
    """Applies a constant noise bias to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for constant noise.

    Returns:
        The data modified by the noise parameters provided.
    """

    # fix tensor device for bias on first call and update config parameters
    if isinstance(cfg.bias, torch.Tensor):
        cfg.bias = cfg.bias.to(device=data.device)

    if cfg.operation == "add":
        return data + cfg.bias
    elif cfg.operation == "scale":
        return data * cfg.bias
    elif cfg.operation == "abs":
        return torch.zeros_like(data) + cfg.bias
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


def uniform_noise(data: torch.Tensor, cfg: UniformNoiseCfg) -> torch.Tensor:
    """Applies a uniform noise to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for uniform noise.

    Returns:
        The data modified by the noise parameters provided.
    """

    # fix tensor device for n_max on first call and update config parameters
    if isinstance(cfg.n_max, torch.Tensor):
        cfg.n_max = cfg.n_max.to(data.device)
    # fix tensor device for n_min on first call and update config parameters
    if isinstance(cfg.n_min, torch.Tensor):
        cfg.n_min = cfg.n_min.to(data.device)

    if cfg.operation == "add":
        return data + torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
    elif cfg.operation == "scale":
        return data * (torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min)
    elif cfg.operation == "abs":
        return torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


def gaussian_noise(data: torch.Tensor, cfg: GaussianNoiseCfg) -> torch.Tensor:
    """Applies a gaussian noise to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for gaussian noise.

    Returns:
        The data modified by the noise parameters provided.
    """

    # fix tensor device for mean on first call and update config parameters
    if isinstance(cfg.mean, torch.Tensor):
        cfg.mean = cfg.mean.to(data.device)
    # fix tensor device for std on first call and update config parameters
    if isinstance(cfg.std, torch.Tensor):
        cfg.std = cfg.std.to(data.device)

    if cfg.operation == "add":
        return data + cfg.mean + cfg.std * torch.randn_like(data)
    elif cfg.operation == "scale":
        return data * (cfg.mean + cfg.std * torch.randn_like(data))
    elif cfg.operation == "abs":
        return cfg.mean + cfg.std * torch.randn_like(data)
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


##
# Noise models as classes
##

class NoiseModel:
    """Base class for noise models."""

    def __init__(self, noise_model_cfg: NoiseModelCfg, num_envs: int, device: str):
        """Initialize the noise model.

        Args:
            noise_model_cfg: The noise configuration to use.
            num_envs: The number of environments.
            device: The device to use for the noise model.
        """
        self._noise_model_cfg = noise_model_cfg
        self._num_envs = num_envs
        self._device = device

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the noise model.

        This method can be implemented by derived classes to reset the noise model.
        This is useful when implementing temporal noise models such as random walk.

        Args:
            env_ids: The environment ids to reset the noise model for. Defaults to None,
                in which case all environments are considered.
        """
        pass

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the noise to the data.

        Args:
            data: The data to apply the noise to. Shape is (num_envs, ...).

        Returns:
            The data with the noise applied. Shape is the same as the input data.
        """
        return self._noise_model_cfg.noise_cfg.func(data, self._noise_model_cfg.noise_cfg)


class NoiseModelWithAdditiveBias(NoiseModel):
    """Noise model with an additive bias.

    The bias term is sampled from a the specified distribution on reset.
    """

    def __init__(self, noise_model_cfg: NoiseModelWithAdditiveBiasCfg, num_envs: int, device: str):
        # initialize parent class
        super().__init__(noise_model_cfg, num_envs, device)
        # store the bias noise configuration
        self._bias_noise_cfg = noise_model_cfg.bias_noise_cfg
        self._bias = torch.zeros((num_envs, 1), device=self._device)
        self._num_components: int | None = None
        self._sample_bias_per_component = noise_model_cfg.sample_bias_per_component

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the noise model.

        This method resets the bias term for the specified environments.

        Args:
            env_ids: The environment ids to reset the noise model for. Defaults to None,
                in which case all environments are considered.
        """
        # resolve the environment ids
        if env_ids is None:
            env_ids = slice(None)
        # reset the bias term
        self._bias[env_ids] = self._bias_noise_cfg.func(self._bias[env_ids], self._bias_noise_cfg)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply bias noise to the data.

        Args:
            data: The data to apply the noise to. Shape is (num_envs, ...).

        Returns:
            The data with the noise applied. Shape is the same as the input data.
        """
        # if sample_bias_per_component, on first apply, expand bias to match last dim of data
        if self._sample_bias_per_component and self._num_components is None:
            *_, self._num_components = data.shape
            # expand bias from (num_envs,1) to (num_envs, num_components)
            self._bias = self._bias.repeat(1, self._num_components)
            # now re-sample that expanded bias in-place
            self.reset()
        return super().__call__(data) + self._bias


from isaaclab.utils import configclass

@configclass
class NoiseCfg:
    """Base configuration for a noise term."""

    func: Callable[[torch.Tensor, NoiseCfg], torch.Tensor] = MISSING
    """The function to be called for applying the noise.

    Note:
        The shape of the input and output tensors must be the same.
    """
    operation: Literal["add", "scale", "abs"] = "add"
    """The operation to apply the noise on the data. Defaults to "add"."""


@configclass
class ConstantNoiseCfg(NoiseCfg):
    """Configuration for an additive constant noise term."""

    func = constant_noise

    bias: torch.Tensor | float = 0.0
    """The bias to add. Defaults to 0.0."""


@configclass
class UniformNoiseCfg(NoiseCfg):
    """Configuration for a additive uniform noise term."""

    func = uniform_noise

    n_min: torch.Tensor | float = -1.0
    """The minimum value of the noise. Defaults to -1.0."""
    n_max: torch.Tensor | float = 1.0
    """The maximum value of the noise. Defaults to 1.0."""


@configclass
class GaussianNoiseCfg(NoiseCfg):
    """Configuration for an additive gaussian noise term."""

    func = gaussian_noise

    mean: torch.Tensor | float = 0.0
    """The mean of the noise. Defaults to 0.0."""
    std: torch.Tensor | float = 1.0
    """The standard deviation of the noise. Defaults to 1.0."""


##
# Noise models
##


@configclass
class NoiseModelCfg:
    """Configuration for a noise model."""

    class_type: type = NoiseModel
    """The class type of the noise model."""

    noise_cfg: NoiseCfg = MISSING
    """The noise configuration to use."""

    func: Callable[[torch.Tensor], torch.Tensor] | None = None
    """Function or callable class used by this noise model.

    The function must take a single `torch.Tensor` (the batch of observations) as input
    and return a `torch.Tensor` of the same shape with noise applied.

    It also supports `callable classes <https://docs.python.org/3/reference/datamodel.html#object.__call__>`_,
    i.e. classes that implement the ``__call__()`` method. In this case, the class should inherit from the
    :class:`NoiseModel` class and implement the required methods.

    This field is used internally by :class:ObservationManager and is not meant to be set directly.
    """


@configclass
class NoiseModelWithAdditiveBiasCfg(NoiseModelCfg):
    """Configuration for an additive gaussian noise with bias model."""

    class_type: type = NoiseModelWithAdditiveBias

    bias_noise_cfg: NoiseCfg = MISSING
    """The noise configuration for the bias.

    Based on this configuration, the bias is sampled at every reset of the noise model.
    """

    sample_bias_per_component: bool = True
    """Whether to sample a separate bias for each data component.

    Defaults to True.
    """
