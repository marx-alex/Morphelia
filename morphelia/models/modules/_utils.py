import torch
from torch import nn
import torch.nn.functional as F


class ArgSequential(nn.Sequential):
    """
    Pytorch Sequential Module. Optional keyword arguments are passed to every module.
    """

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x, **kwargs):
        for module in self:
            x = module(x, **kwargs)
        return x


def add_condition(x: torch.Tensor, c: torch.Tensor, n_conditions: int) -> torch.Tensor:
    """
    Concatenate the input tensor with hot-encoded conditions.

    Args:
        x: Input tensor.
        c: Tensor with conditions.
        n_conditions: Number of conditions.
    """
    c = F.one_hot(c, num_classes=n_conditions)

    if len(x.shape) == 3:
        seq_len = x.shape[1]
        c = c.reshape(-1, 1, n_conditions)  # [N, 1, n_conditions]
        c = torch.tile(c, (1, seq_len, 1))  # [N, S, n_conditions]

    return torch.cat((x, c), dim=-1)
