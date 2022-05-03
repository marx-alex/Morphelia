import torch
from torch import nn
import torch.nn.functional as F


class MultOutSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x: torch.Tensor, **kwargs) -> tuple:
        outs = []
        for module in self:
            output = module(x, **kwargs)
            if isinstance(output, tuple):
                x, out = output[0], output[1:]
                if len(out) == 1:
                    out = out[0]
                outs.append(out)
            else:
                x = output

        if len(outs) > 0:
            if all(isinstance(item, tuple) for item in outs):
                outs = list(zip(*outs))
            return x, outs
        return x


class PosArgSequential(nn.Sequential):
    """
    Pytorch Sequential Module. Optional keyword arguments are passed to every module
    depending on their position.
    """

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for i, module in enumerate(self):
            sub_kwargs = {k: v[i] for k, v in kwargs.items()}
            x = module(x, **sub_kwargs)
        return x


class ArgSequential(nn.Sequential):
    """
    Pytorch Sequential Module. Optional keyword arguments are passed to every module.
    """

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
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
