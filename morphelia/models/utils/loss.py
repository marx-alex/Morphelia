import torch
from torch import nn

from .helper import partition


class MaskedMSELoss(nn.Module):
    """Masked MSE Loss"""

    def __init__(self, reduction: str = "mean"):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.
        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns:
            if reduction == 'none':
                (num_active,) Loss for each active batch element as a tensor with gradient attached.
            if reduction == 'mean':
                scalar mean loss over batch as a tensor with gradient attached.
        """
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the gaussian kernel between two tensors.
    """
    x_size, y_size = x.shape[0], y.shape[0]
    dim = x.shape[1]

    tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
    tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)

    return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / dim)


def mmd(x, y):
    """Compute the Maximum Mean Discrepancy"""
    x_kernel = gaussian_kernel(x, x)
    y_kernel = gaussian_kernel(y, y)
    xy_kernel = gaussian_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy between every different condition."""

    def __init__(self, n_conditions: int = None):
        super().__init__()

        self.mmd = mmd
        self.n_conditions = n_conditions
        self.partition = partition

    def forward(
        self,
        y: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:

        loss = torch.tensor(0.0, device=y.device)

        partitions = self.partition(y, c, self.n_conditions)
        for i in range(len(partitions)):
            if partitions[i].shape[0] > 0:
                for j in range(i):
                    if partitions[j].shape[0] > 0:
                        loss += self.mmd(partitions[i], partitions[j])

        return loss
