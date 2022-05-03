import torch
from torch import nn


class MeanClassifier(nn.Module):
    """
    Mean Classification
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

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x : B x N x F
        """

        x = self.feature_norm(x).mean(dim=1)

        return self.classifier(x)


class MedianClassifier(nn.Module):
    """
    Median Classification
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

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x : B x N x F
        """

        x = x.median(dim=1)[0]

        return self.classifier(x)


class MajorityClassifier(nn.Module):
    """
    Majority Vote
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

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: B x N x F
        """
        B, N, F = x.size()

        x = x.view(-1, F)
        x = self.classifier(x)
        x = x.view(B, N, self.n_classes)

        x = x.mean(dim=1)

        return x
