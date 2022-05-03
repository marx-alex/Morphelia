from typing import Optional

import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl

from morphelia.models.modules._utils import add_condition
from morphelia.models.modules.vae import CondLayer


class MultiClassRegression(nn.Module):
    def __init__(
        self, in_features: int, n_classes: int = 0, latent_dim: int = 64
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.class_layer = nn.Sequential(
            CondLayer(
                in_features=in_features,
                out_features=latent_dim,
                n_conditions=self.n_classes,
                bias=True,
            ),
            nn.ReLU(),
        )
        self.regression = nn.Linear(latent_dim, 1)

    def forward(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch size, sequence length, feature size]
            c: Tensor of shape [batch size,]
        """
        # concatenate x and condition
        if c is not None:
            x = add_condition(x, c, self.n_classes)

        x = self.class_layer(x)

        return self.regression(x)


class TSEMModule(pl.LightningModule):
    """
    Time series evolutional model.
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        t_max: Optional[int] = None,
        ce_beta: float = 1.0,
        mse_beta: float = 1.0,
        learning_rate: float = 1e-3,
        optimizer: str = "Adam",
    ):
        super().__init__()

        self.save_hyperparameters("learning_rate", "optimizer")

        self.encoder = encoder
        self.n_classes = n_classes
        self.latent_dim = encoder.latent_dim
        self.t_max = t_max
        self.mse_beta = mse_beta
        self.ce_beta = ce_beta

        self.softmax = nn.Softmax(dim=1)

        self.classifier = nn.Sequential(nn.Linear(self.latent_dim, n_classes))

        self.regression = MultiClassRegression(
            in_features=self.latent_dim, n_classes=n_classes
        )

        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")
        self.mse = nn.MSELoss()

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
        _, pred, _ = self(x, c=c)
        return pred

    def forward(self, x, c: Optional[torch.Tensor] = None):
        z = self.forward_features(x, c=c)
        logits = self.classifier(z)
        pred = self.softmax(logits)
        pred_class = torch.argmax(pred, dim=1)
        y = self.regression(z, c=pred_class)
        return logits, pred, y

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
        c = None
        if "c" in batch:
            c = batch["c"]
        return x, target, t, c

    def _weighted_cross_entropy(
        self, loss: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if t is not None:
            t_scaled = t / self.t_max
            return torch.mean(torch.mul(loss, t_scaled))
        return torch.mean(loss)

    def _common_step(self, batch, prefix="train"):
        x, target, t, c = self._prepare_batch(batch)

        logits, pred, y = self(x, c=c)
        ce = self.cross_entropy(logits, target)
        ce = self._weighted_cross_entropy(ce, t) * self.ce_beta
        mse = self.mse(y, t.unsqueeze(1)) * self.mse_beta
        loss = ce + mse

        self.log(
            f"{prefix}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log_dict(
            {
                f"{prefix}/cross_entropy": ce,
                f"{prefix}/mse": mse,
            },
            on_step=False,
            on_epoch=True,
        )

        return loss, pred, target
