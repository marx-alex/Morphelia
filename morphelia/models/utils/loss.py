import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

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


# https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
def gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Gaussian kernel between two tensors.
    """
    x_n, y_n = x.size(0), y.size(0)
    n_dim = x.size(1)
    x = x.unsqueeze(1)  # [x_n, 1, n_dim]
    y = y.unsqueeze(0)  # [1, y_n, n_dim)
    tiled_x = x.expand(x_n, y_n, n_dim)
    tiled_y = y.expand(x_n, y_n, n_dim)
    return torch.exp(-(tiled_x - tiled_y).pow(2).mean(2) / float(n_dim))


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy Loss."""

    def __init__(self):
        super().__init__()
        self.kernel = gaussian_kernel

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        """
        Args:
            source: Source features in Hilbert space.
            target: Target features in Hilbert space

        Returns:
            Maximum mean discrepancy between kernel embeddings of source and target.
        """
        x_kernel = self.kernel(source, source)
        y_kernel = self.kernel(target, target)
        xy_kernel = self.kernel(source, target)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)


class ConditionalMMDLoss(nn.Module):
    """Maximum Mean Discrepancy between every different condition."""

    def __init__(self, n_conditions: int = None):
        super().__init__()

        self.mmd_loss = MMDLoss()
        self.n_conditions = n_conditions
        self.partition = partition

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch size, features].
            c: Tensor of shape [batch size,]

        Returns:
            Loss
        """

        loss = torch.tensor(0.0, device=x.device)

        partitions = self.partition(x, c, self.n_conditions)
        for i in range(len(partitions)):
            if partitions[i].shape[0] > 0:
                for j in range(i):
                    if partitions[j].shape[0] > 0:
                        loss += self.mmd(partitions[i], partitions[j])

        return loss


class KLDLoss(nn.Module):
    """
    Kullback-Leibler Divergence between input distribution and normal distribution.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, mu: torch.Tensor, log_sigma: torch.Tensor):
        """
        Args:
            mu: Means.
            log_sigma: Logarithmic variance
        """
        var = torch.exp(log_sigma) + self.eps
        kld = (
            kl_divergence(
                Normal(mu, var.sqrt()),
                Normal(torch.zeros_like(mu), torch.ones_like(var)),
            )
            .sum(dim=1)
            .mean()
        )
        return kld
