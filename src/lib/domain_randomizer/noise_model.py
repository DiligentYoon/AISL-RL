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

def constant_noise(data: torch.Tensor, bias: float, operation: str = "add") -> torch.Tensor:
    """Applies a constant noise bias to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for constant noise.

    Returns:
        The data modified by the noise parameters provided.
    """

    # fix tensor device for bias on first call and update config parameters
    if isinstance(bias, torch.Tensor):
        bias = bias.to(device=data.device)

    if operation == "add":
        return data + bias
    elif operation == "scale":
        return data * bias
    elif operation == "abs":
        return torch.zeros_like(data) + bias
    else:
        raise ValueError(f"Unknown operation in noise: {operation}")


def uniform_noise(data: torch.Tensor, n_min: float, n_max: float, operation: str = "add") -> torch.Tensor:
    """Applies a uniform noise to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for uniform noise.

    Returns:
        The data modified by the noise parameters provided.
    """

    # fix tensor device for n_max on first call and update config parameters
    if isinstance(n_max, torch.Tensor):
        n_max = n_max.to(data.device)
    # fix tensor device for n_min on first call and update config parameters
    if isinstance(n_min, torch.Tensor):
        n_min = n_min.to(data.device)

    if operation == "add":
        return data + torch.rand_like(data) * (n_max - n_min) + n_min
    elif operation == "scale":
        return data * (torch.rand_like(data) * (n_max - n_min) + n_min)
    elif operation == "abs":
        return torch.rand_like(data) * (n_max - n_min) + n_min
    else:
        raise ValueError(f"Unknown operation in noise: {operation}")


def gaussian_noise(data: torch.Tensor, mean: float, std: float, operation: str = "add") -> torch.Tensor:
    """Applies a gaussian noise to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for gaussian noise.

    Returns:
        The data modified by the noise parameters provided.
    """

    # fix tensor device for mean on first call and update config parameters
    if isinstance(mean, torch.Tensor):
        mean = mean.to(data.device)
    # fix tensor device for std on first call and update config parameters
    if isinstance(std, torch.Tensor):
        std = std.to(data.device)

    if operation == "add":
        return data + mean + std * torch.randn_like(data)
    elif operation == "scale":
        return data * (mean + std * torch.randn_like(data))
    elif operation == "abs":
        return mean + std * torch.randn_like(data)
    else:
        raise ValueError(f"Unknown operation in noise: {operation}")
