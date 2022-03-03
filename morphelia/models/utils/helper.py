from typing import List

import torch
from torch import nn


def get_activation_fn(activation: str):
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU
    elif activation == "gelu":
        return nn.GELU
    raise NotImplementedError(
        f"Activation can be either 'relu' or 'gelu', instead got {activation}"
    )


def partition(x: torch.Tensor, y: torch.Tensor, n_partitions) -> List[torch.Tensor]:
    partitions = []
    y = y.flatten()

    for i in range(n_partitions):
        mask = y == i
        partitions.append(x[mask, ...])

    return partitions
