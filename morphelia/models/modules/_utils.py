import torch
from torch import nn
import torch.nn.functional as F

from typing import Union


class MultOutSequential(nn.Sequential):
    """Sequential module for multiple outputs.

    Pytorch module to concatenate module sequences, where one or more modules generate
    multiple outputs. In this case, the data is returned together with the outputs.

    Parameters
    ----------
    *args
        `pytorch.nn.Module`
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def forward(self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, tuple]:
        """Pass a tensor through all modules.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor, list
            Tensor and list of all collected outputs
        """
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
    """Sequential module with positional keyword arguments.

    This module allows passing keyword arguments to submodules.
    Keyword arguments are passed to every module
    depending on their position.

    Parameters
    ----------
    *args
        `pytorch.nn.Module`
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Pass a tensor through all modules.

        Parameters
        ----------
        x : torch.Tensor
        **kwargs
            Keyword arguments with list as values. List items are passed to module depending
            on their position in the list on the module position in the sequence

        Returns
        -------
        torch.Tensor
        """
        for i, module in enumerate(self):
            sub_kwargs = {k: v[i] for k, v in kwargs.items()}
            x = module(x, **sub_kwargs)
        return x


class ArgSequential(nn.Sequential):
    """Sequential module with keyword arguments.

    This module allows passing keyword arguments to all submodules.

    Parameters
    ----------
    *args
        `pytorch.nn.Module`
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Pass a tensor through all modules.

        Parameters
        ----------
        x : torch.Tensor
        **kwargs
            Keyword arguments are passed to every module

        Returns
        -------
            torch.Tensor
        """
        for module in self:
            x = module(x, **kwargs)
        return x


def add_condition(x: torch.Tensor, c: torch.Tensor, n_conditions: int) -> torch.Tensor:
    """Concatenate the input tensor with hot-encoded conditions.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    c : torch.Tensor
        Tensor with conditions
    n_conditions : int
        Number of conditions

    Returns
    -------
    torch.Tensor
        Tensor of shape [n, features + number of conditions]
    """
    c = F.one_hot(c, num_classes=n_conditions)

    if len(x.shape) == 3:
        seq_len = x.shape[1]
        c = c.reshape(-1, 1, n_conditions)  # [N, 1, n_conditions]
        c = torch.tile(c, (1, seq_len, 1))  # [N, S, n_conditions]

    return torch.cat((x, c), dim=-1)
