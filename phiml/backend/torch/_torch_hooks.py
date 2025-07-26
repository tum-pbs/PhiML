# This module is only imported on demand
from typing import Callable

import torch

from .._backend import get_backend

from ... import Tensor, tensor, batch, channel, spatial, math

TORCH = get_backend('torch')


def handle_torch_function(func: Callable, types, args: tuple, kwargs: dict):  # called by Tensor.__torch_function__
    """
    1. Converts PhiML tensors to PyTorch tensors, even if their native is of another backend.
    2. Calls the PyTorch function.
    3. Converts the result back to PhiML Tensor backed by PyTorch.

    Returns:
        NotImplemented if the function is not supported
    """
    if func == torch.nn.functional.interpolate:
        input: Tensor = args[0]
        if input.backend.name != 'torch':
            input = math.convert(input, TORCH)
        if kwargs['mode'] == 'linear' and spatial(input).rank > 1:
            kwargs['mode'] = 'bilinear' if spatial(input).rank == 2 else 'trilinear'
        native_result = func(input.native([batch, channel, *spatial(input)]), **kwargs)
        return tensor(native_result, [batch(input), channel(input), *spatial(input).without_sizes()], convert=False)
    return NotImplemented
