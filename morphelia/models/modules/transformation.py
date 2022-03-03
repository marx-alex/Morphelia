import torch
from torch import nn


class PermuteAxis(nn.Module):
    def __init__(self, *dims):
        super(PermuteAxis, self).__init__()
        self.dims = [d for d in dims]

    def forward(self, x: torch.Tensor):
        return x.permute(*self.dims)


class AddLayer(nn.Module):
    def __init__(self, layer):
        super(AddLayer, self).__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor):
        return x + self.layer(x)


class MultLayer(nn.Module):
    def __init__(self, factor):
        super(MultLayer, self).__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor):
        return x * self.factor


class Reshape(nn.Module):
    def __init__(self, *args, fixed=None):
        super(Reshape, self).__init__()
        self.shape = [a for a in args]
        self.fixed = fixed

    def forward(self, x: torch.Tensor):
        if self.fixed is None:
            return x.reshape(*self.shape)
        else:
            fixed = x.shape[: self.fixed]
            return x.reshape(*fixed, *self.shape)
