import torch
from torch import nn


class MeanClassifier(nn.Module):
    """Mean classification module.

    Parameters
    ----------
    input_dim : int
        Input dimension
    n_classes : int
        Number of classes
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
    ) -> None:
        super().__init__()

        self.n_classes = n_classes
        self.feature_norm = nn.LayerNorm(input_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, n_classes)
        )  # [B,F] -> [B, n_classes]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes tensor through module.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `[batch size, sequence length, features]`

        Returns
        -------
        torch.Tensor
            Tensor of shape `[batch size, number of classes]`
        """

        x = self.feature_norm(x).mean(dim=1)

        return self.classifier(x)


class MedianClassifier(nn.Module):
    """Median classification module.

    Parameters
    ----------
    input_dim : int
        Input dimension
    n_classes : int
        Number of classes
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
    ) -> None:
        super().__init__()

        self.n_classes = n_classes

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, n_classes)
        )  # [B,F] -> [B, n_classes]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes tensor through module.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `[batch size, sequence length, features]`

        Returns
        -------
        torch.Tensor
            Tensor of shape `[batch size, number of classes]`
        """

        x = x.median(dim=1)[0]

        return self.classifier(x)


class MajorityClassifier(nn.Module):
    """Classification by majority vote.

    Parameters
    ----------
    input_dim : int
        Input dimension
    n_classes : int
        Number of classes
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
    ) -> None:
        super().__init__()

        self.n_classes = n_classes

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, n_classes)
        )  # [B,F] -> [B, n_classes]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes tensor through module.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `[batch size, sequence length, features]`

        Returns
        -------
        torch.Tensor
            Tensor of shape `[batch size, number of classes]`
        """
        B, N, F = x.size()

        x = x.view(-1, F)
        x = self.classifier(x)
        x = x.view(B, N, self.n_classes)

        x = x.mean(dim=1)

        return x
