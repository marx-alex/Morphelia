from typing import Optional, Tuple

import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl

from ..utils.helper import get_activation_fn


class ClusterDistance(nn.Module):
    """Cluster distance model that learns cluster centers.

    Parameters
    ----------
    n_classes : int, optional
        Number of classes
    latent_dim : int
        Dimensions of the latent space
    cluster_centers : torch.Tensor, optional
        Cluster centers that should be initialized.
        Should be a tensor of shape `[number of classes, latent dimensions]`.
    """

    def __init__(
        self,
        n_classes: int,
        latent_dim: int,
        cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.enc_shape = latent_dim
        self.n_classes = n_classes
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_classes, self.enc_shape, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward tensor through module.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
            Distances to cluster centers. This is a tensor of shape
            `[batch size, number of classes]`.
        """
        cdist = torch.cdist(x, self.cluster_centers)
        return cdist


class DEEPMAP(pl.LightningModule):
    """Implementation of the DeepMap model as
    as PyTorch Lighnting module.

    Parameters
    ----------
    encoder : torch.nn.Module
        Encoder module for dimensionality reduction
    n_classes : int
        Number of classes
    learning_rate : float
        Learning rate during training
    optimizer : str
        Optimizer for the training process.
        Can be `Adam` or `AdamW`.

    References
    ----------
    .. [1] Ren et al., 2021, bioRxiv
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        cdist_act: Optional[str] = None,
        learning_rate: float = 1e-3,
        optimizer: str = "Adam",
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.encoder = encoder
        self.n_classes = n_classes
        self.latent_dim = encoder.latent_dim

        self.cdist_act = get_activation_fn(cdist_act)
        self.softmax = nn.Softmax(dim=1)

        self.clustering = ClusterDistance(
            n_classes=n_classes, latent_dim=self.latent_dim
        )

        self.cross_entropy = nn.CrossEntropyLoss()

        metrics = torchmetrics.MetricCollection(
            dict(
                ACC=torchmetrics.Accuracy(num_classes=self.n_classes),
                F1=torchmetrics.F1Score(num_classes=self.n_classes),
            )
        )

        self.train_metric = metrics.clone("train/")
        self.valid_metric = metrics.clone("valid/")
        self.test_metric = metrics.clone("test/")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Passes tensor through encoder module.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
            Latent representation
        """
        return self.encoder(x)

    def forward_pred(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards tensor through model and transforms
        logits with the SoftMax function.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
            Predictions
        """
        logits, _ = self(x)
        return self.softmax(logits)

    def classifier(self, x: torch.Tensor) -> torch.Tensor:
        """Classification by a single classification layer.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
            Logits
        """
        cdist = self.clustering(x)
        if self.cdist_act is not None:
            cdist = self.cdist_act(cdist)
        return cdist

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forwards tensor through model.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor, torch.Tensor
            Logits and predictions
        """
        z = self.forward_features(x)
        logits = self.classifier(z)
        pred = self.softmax(logits)
        return logits, pred

    def training_step(self, batch: dict, batch_idx: torch.Tensor) -> torch.Tensor:
        """Training step.

        Parameters
        ----------
        batch : dict
            Dictionary with `x` and `target` as required keys.
            `t` for time is a optional key.
        batch_idx : torch.Tensor
            Batch index

        Returns
        -------
        torch.Tensor
            Loss
        """
        loss, pred, target = self._common_step(batch, prefix="train")
        self.train_metric(pred, target)
        self.log_dict(self.train_metric, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: torch.Tensor) -> torch.Tensor:
        """Validation step.

        Parameters
        ----------
        batch : dict
            Dictionary with `x` and `target` as required keys.
            `t` for time is a optional key.
        batch_idx : torch.Tensor
            Batch index

        Returns
        -------
        torch.Tensor
            Loss
        """
        loss, pred, target = self._common_step(batch, prefix="valid")
        self.valid_metric(pred, target)
        self.log_dict(self.valid_metric, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: dict, batch_idx: torch.Tensor) -> torch.Tensor:
        """Test step.

        Parameters
        ----------
        batch : dict
            Dictionary with `x` and `target` as required keys.
            `t` for time is a optional key.
        batch_idx : torch.Tensor
            Batch index

        Returns
        -------
        torch.Tensor
            Loss
        """
        loss, pred, target = self._common_step(batch, prefix="test")
        self.test_metric(pred, target)
        self.log_dict(self.test_metric, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer.

        Returns
        -------
        torch.optim.Adam or torch.optim.AdamW
        """
        if self.hparams.optimizer == "Adam":
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "AdamW":
            opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        else:
            assert False, f"{self.hparams.optimizer=} is not Adam or AdamW"

        return opt

    @staticmethod
    def _prepare_batch(batch):
        x = batch["x"]
        target = batch["target"]
        t = None
        if "t" in batch:
            t = batch["t"]
        return x, target, t

    def _common_step(self, batch, prefix="train"):
        x, target, t = self._prepare_batch(batch)

        logits, pred = self(x)
        loss = self.cross_entropy(logits, target)

        self.log(
            f"{prefix}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss, pred, target
