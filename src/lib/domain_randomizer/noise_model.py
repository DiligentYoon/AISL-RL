from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch


from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

import torch

##
# Internal helpers.
##

def _ensure_tensor(value, data: torch.Tensor) -> torch.Tensor:
    """Convert scalar/list/tensor to a tensor matching data's dtype and device."""
    if not isinstance(value, torch.Tensor):
        return torch.as_tensor(value, dtype=data.dtype, device=data.device)
    if value.device != data.device:
        return value.to(device=data.device)
    return value


##
# Noise as functions.
##

def constant_noise(data: torch.Tensor, bias: float, operation: str = "add") -> torch.Tensor:
    """Applies a constant noise bias to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        bias: Constant bias value. Can be a scalar, list, or tensor.
              If a list or tensor of length == data.shape[-1], bias is applied per-dimension.
        operation: One of "add", "scale", or "abs".

    Returns:
        The data modified by the noise parameters provided.
    """
    bias = _ensure_tensor(bias, data)

    if operation == "add":
        return data + bias
    elif operation == "scale":
        return data * bias
    elif operation == "abs":
        return torch.zeros_like(data) + bias
    else:
        raise ValueError(f"Unknown operation in noise: {operation}")


def uniform_noise(data: torch.Tensor, min: float, max: float, operation: str = "add") -> torch.Tensor:
    """Applies a uniform noise to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        min: Lower bound of the uniform distribution. Can be a scalar, list, or tensor.
               If a list or tensor of length == data.shape[-1], applied per-dimension.
        max: Upper bound of the uniform distribution. Can be a scalar, list, or tensor.
               If a list or tensor of length == data.shape[-1], applied per-dimension.
        operation: One of "add", "scale", or "abs".

    Returns:
        The data modified by the noise parameters provided.
    """
    min = _ensure_tensor(min, data)
    max = _ensure_tensor(max, data)

    if operation == "add":
        return data + torch.rand_like(data) * (max - min) + min
    elif operation == "scale":
        return data * (torch.rand_like(data) * (max - min) + min)
    elif operation == "abs":
        return torch.rand_like(data) * (max - min) + min
    else:
        raise ValueError(f"Unknown operation in noise: {operation}")


def gaussian_noise(data: torch.Tensor, mean: float, std: float, operation: str = "add") -> torch.Tensor:
    """Applies a gaussian noise to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        mean: Mean of the gaussian distribution. Can be a scalar, list, or tensor.
              If a list or tensor of length == data.shape[-1], applied per-dimension.
        std: Standard deviation of the gaussian distribution. Can be a scalar, list, or tensor.
             If a list or tensor of length == data.shape[-1], applied per-dimension.
             Set individual elements to 0.0 to suppress noise for specific observation axes.
        operation: One of "add", "scale", or "abs".

    Returns:
        The data modified by the noise parameters provided.
    """
    mean = _ensure_tensor(mean, data)
    std = _ensure_tensor(std, data)

    if operation == "add":
        return data + mean + std * torch.randn_like(data)
    elif operation == "scale":
        return data * (mean + std * torch.randn_like(data))
    elif operation == "abs":
        return mean + std * torch.randn_like(data)
    else:
        raise ValueError(f"Unknown operation in noise: {operation}")


##
# Config-time helpers.
##

def build_noise_std_vector(groups: dict) -> list:
    """Expand per-group noise std into a flat list matching observation dimension.

    Converts a human-readable group definition into a flat per-dimension std list
    that can be passed directly to gaussian_noise (or any noise function) as the
    ``std`` parameter. The order of groups must match the observation tensor's
    concat order in ``_get_observations()``.

    Args:
        groups: dict of {"group_name": {"dim": int, "std": float}, ...}.
                Keys are used only for readability; insertion order is preserved (Python 3.7+).
                Set "std" to 0.0 for axes that should receive no noise (e.g., internal values).

    Returns:
        Flat list of std values with length == sum of all "dim" values.

    Example::

        groups = {
            "base_lin_vel": {"dim": 3, "std": 0.05},
            "command":      {"dim": 3, "std": 0.0},
        }
        build_noise_std_vector(groups)
        # → [0.05, 0.05, 0.05, 0.0, 0.0, 0.0]
    """
    std_vector = []
    for name, spec in groups.items():
        std_vector.extend([spec["std"]] * spec["dim"])
    return std_vector


def build_noise_uniform_vector(groups: dict) -> list:
    n_min = []
    n_max = []

    for group in groups.values():
        dim = group["dim"]
        low = group["min"]
        high = group["max"]

        n_min.extend([low] * dim)
        n_max.extend([high] * dim)

    return n_min, n_max