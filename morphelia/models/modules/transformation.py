import torch
from torch import nn

from typing import Optional


class PermuteAxis(nn.Module):
    """Permute tensor axis.

    Parameters
    ----------
    *dims
        Tensor dimensions
    """

    def __init__(self, *dims):
        super(PermuteAxis, self).__init__()
        self.dims = [d for d in dims]

    def forward(self, x: torch.Tensor):
        """Pass tensor through module.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return x.permute(*self.dims)


class AddLayer(nn.Module):
    """Add module output to tensor.

    Parameters
    ----------
    layer : pytorch.nn.Module
        Pytorch compatible module
    """

    def __init__(self, layer: torch.nn.Module):
        super(AddLayer, self).__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor):
        """Pass tensor through module.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return x + self.layer(x)


class MultLayer(nn.Module):
    """Multiply tensor by factor.

    Parameters
    ----------
    factor : float
        Tensor is multiplied with factor
    """

    def __init__(self, factor: float):
        super(MultLayer, self).__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor):
        """Pass tensor through module.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return x * self.factor


class Reshape(nn.Module):
    """Reshape tensor.

    Parameters
    ----------
    *args
        Tensor shape
    fixed : int, optional
        Keep one dimension fixed
    """

    def __init__(self, *args, fixed: Optional[int] = None):
        super(Reshape, self).__init__()
        self.shape = [a for a in args]
        self.fixed = fixed

    def forward(self, x: torch.Tensor):
        """Pass tensor through module.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
            Reshaped tensor
        """
        if self.fixed is None:
            return x.reshape(*self.shape)
        else:
            fixed = x.shape[: self.fixed]
            return x.reshape(*fixed, *self.shape)
