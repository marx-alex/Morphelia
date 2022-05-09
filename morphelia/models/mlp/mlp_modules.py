from typing import Optional

import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl


class MLPModule(pl.LightningModule):
    """
    Multilayer Perceptron for Classification
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        t_max: Optional[int] = None,
        learning_rate: float = 1e-3,
        optimizer: str = "Adam",
    ):
        super().__init__()

        self.save_hyperparameters("learning_rate", "optimizer")

        self.encoder = encoder
        self.n_classes = n_classes
        self.latent_dim = encoder.latent_dim
        self.t_max = t_max

        self.softmax = nn.Softmax(dim=1)

        self.classifier = nn.Linear(self.latent_dim, n_classes)

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

    def forward_features(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.encoder(x, c=c)

    def forward_pred(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self(x, c=c)
        return self.softmax(x)

    def forward(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        z = self.forward_features(x, c=c)
        logits = self.classifier(z)
        return logits

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
        c = None
        if "c" in batch:
            c = batch["c"]
        return x, target, c

    def _common_step(self, batch, prefix="train"):
        x, target, c = self._prepare_batch(batch)

        logits = self(x, c=c)
        loss = self.cross_entropy(logits, target)
        pred = self.softmax(logits)

        self.log(
            f"{prefix}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss, pred, target
