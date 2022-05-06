from typing import Optional, Tuple

import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl

from ..utils.helper import get_activation_fn


class ClusterDistance(nn.Module):
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cdist = torch.cdist(x, self.cluster_centers)
        return cdist


class DEEPMAP(pl.LightningModule):
    """
    DeepMap as described by Ren et al., 2021, bioRxiv
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        cdist_act: Optional[str] = None,
        learning_rate: float = 1e-3,
        optimizer: str = "Adam",
    ):
        super().__init__()

        self.save_hyperparameters("learning_rate", "optimizer")

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
        return self.encoder(x)

    def forward_pred(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self(x)
        return self.softmax(logits)

    def classifier(self, x: torch.Tensor) -> torch.Tensor:
        cdist = self.clustering(x)
        if self.cdist_act is not None:
            cdist = self.cdist_act(cdist)
        return cdist

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        z = self.forward_features(x)
        logits = self.classifier(z)
        pred = self.softmax(logits)
        return logits, pred

    def training_step(self, batch, batch_idx):
        loss, pred, target = self._common_step(batch, prefix="train")
        self.train_metric(pred, target)
        self.log_dict(self.train_metric, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, target = self._common_step(batch, prefix="valid")
        self.valid_metric(pred, target)
        self.log_dict(self.valid_metric, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, pred, target = self._common_step(batch, prefix="test")
        self.test_metric(pred, target)
        self.log_dict(self.test_metric, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
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
