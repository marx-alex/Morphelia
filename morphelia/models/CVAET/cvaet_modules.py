from typing import Optional, Tuple, Union

import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl

from ..modules import sampling
from ..utils import MaskedMSELoss, ConditionalMMDLoss, geometric_noise, KLDLoss


class cVAET_us(pl.LightningModule):
    """
    Unsupervised part of Conditional Variational Autoencoder with a Time Series Transformer.
    """

    def __init__(
        self,
        encoder: nn.Module,
        transformer: nn.Module,
        decoder: nn.Module,
        kld_beta: float = 1.0,
        mmd_beta: float = 1.0,
        mse_beta: float = 1.0,
        mask: bool = True,
        masking_len=3,
        masking_ratio=0.15,
        learning_rate: float = 1e-4,
        optimizer: str = "Adam",
    ) -> None:
        super().__init__()

        self.save_hyperparameters("learning_rate", "optimizer")
        self.kld_beta = kld_beta
        self.mmd_beta = mmd_beta
        self.mse_beta = mse_beta
        self.mask = mask
        self.masking_len = masking_len
        self.masking_ratio = masking_ratio

        self.encoder = encoder  # [B x S x F] -> [B x S x latent_dim]
        self.transformer = transformer  # [B x S x latent_dim] -> [B x S x latent_dim]
        self.decoder = decoder  # [B x S x latent_dim] -> [B x S x F]

        # initiate loss functions
        self.mse = MaskedMSELoss()
        self.mmd = ConditionalMMDLoss(n_conditions=encoder.n_conditions)
        self.kld = KLDLoss()

    def forward_features(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_statistic: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:

        mean, log_var = self.encoder(x, c=c)
        z = sampling(mean, log_var)
        z = self.transformer(z, key_padding_mask=padding_mask)

        if return_statistic:
            return z, mean, log_var

        return z

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_length, feat_dim) Masked features (input).
            c: (batch_size,) Conditions.
            padding_mask: (batch_size, seq_length) boolean tensor,
                0s mean keep vector at this position, 1s means padding.
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        z, z_mean, z_log_var = self.forward_features(
            x, c=c, padding_mask=padding_mask, return_statistic=True
        )
        x_hat, y_hat = self.decoder(z, c=c)

        return z, z_mean, z_log_var, x_hat, y_hat

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, flag="train/")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, flag="valid/")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, flag="test/")
        return loss

    def configure_optimizers(self):
        if self.hparams.optimizer == "Adam":
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "AdamW":
            opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise NotImplementedError(f"{self.hparams.optimizer=} is not Adam or AdamW")

        return opt

    def _masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_mask = geometric_noise(x, self.masking_len, self.masking_ratio)
        x = x * x_mask
        return x, ~x_mask.type(torch.bool)

    @staticmethod
    def _prepare_batch(batch):
        x = batch["x"]
        c = None
        if "c" in batch:
            c = batch["c"]
        padding_mask = None
        if "padding_mask" in batch:
            padding_mask = batch["padding_mask"]
        return x, c, padding_mask

    def _common_step(self, batch, flag="train/"):
        x, c, padding_mask = self._prepare_batch(batch)
        target_x = x.clone()

        if self.mask:
            x, x_mask = self._masking(x)
            if padding_mask is not None:
                target_mask = x_mask * ~padding_mask.unsqueeze(-1)
            else:
                target_mask = x_mask
        else:
            if padding_mask is not None:
                target_mask = torch.ones_like(target_x) * ~padding_mask.unsqueeze(-1)
            else:
                target_mask = torch.ones_like(target_x)

        z, z_mean, z_log_var, x_hat, y_hat = self(x, c=c, padding_mask=padding_mask)

        b, s, f = z.shape
        if c is not None:
            c = torch.tile(c.reshape(-1, 1), (1, s))  # [B,] -> [B, S]
            c = c.reshape(b * s, -1)  # [B, S] -> [B x S,]
            y_hat = y_hat.reshape(b * s, -1)  # [B, S, F] -> [B x S, F]
            mmd = self.mmd(y_hat, c)
        else:
            mmd = torch.tensor(0.0, device=z.device)

        mse = self.mse(x_hat, target_x, mask=target_mask)
        kld = self.kld(z_mean, z_log_var)

        loss = (self.mse_beta * mse) + (self.kld_beta * kld) + (self.mmd_beta * mmd)

        self.log_dict(
            {
                f"{flag}loss": loss,
                f"{flag}mse": mse,
                f"{flag}kld": kld,
                f"{flag}mmd": mmd,
            }
        )

        return loss


class cVAET_s(pl.LightningModule):
    """
    Supervised part of Conditional Variational Autoencoder with a Time Series Transformer.
    """

    def __init__(
        self,
        model: pl.LightningModule,
        n_classes: int,
        learning_rate: float = 1e-4,
        optimizer: str = "Adam",
    ) -> None:
        super().__init__()

        self.save_hyperparameters("learning_rate", "optimizer")
        self.n_classes = n_classes

        self.encoder = model.encoder  # [B x S x F] -> [B x S x latent_dim]
        self.transformer = model.transformer
        self.classifier = nn.Linear(
            self.transformer.input_dim * self.transformer.seq_len, self.n_classes
        )

        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(-1)

        metrics = torchmetrics.MetricCollection(
            dict(
                ACC=torchmetrics.Accuracy(num_classes=n_classes),
                F1=torchmetrics.F1Score(num_classes=n_classes),
            )
        )

        self.train_metric = metrics.clone(prefix="train/")
        self.valid_metric = metrics.clone(prefix="valid/")
        self.test_metric = metrics.clone(prefix="test/")

    def forward_features(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mean, log_var = self.encoder(x, c=c)
        z = sampling(mean, log_var)

        z = self.transformer(z, key_padding_mask=key_padding_mask)
        return z

    def classify(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if padding_mask is not None:
            y = x * ~padding_mask.unsqueeze(-1)
        else:
            y = x

        y = x.reshape(y.shape[0], -1)  # [N, S x F]
        return self.classifier(y)  # [N, n_classes]

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_length, feat_dim) Masked features (input).
            c: (batch_size,) Conditions.
            padding_mask: (batch_size, seq_length) boolean tensor,
                0s mean keep vector at this position, 1s means padding.
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        z = self.forward_features(x, c=c, key_padding_mask=padding_mask)
        logits = self.classify(z, padding_mask=padding_mask)

        return logits

    def training_step(self, batch, batch_idx):
        loss, pred, target = self._common_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.train_metric.update(pred, target)
        self.log_dict(self.train_metric, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, target = self._common_step(batch)
        self.log("valid/loss", loss, on_step=True, on_epoch=True)
        self.valid_metric.update(preds, target)
        self.log_dict(self.valid_metric, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, target = self._common_step(batch)
        self.log("test/loss", loss, on_step=True, on_epoch=True)
        self.test_metric.update(preds, target)
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

    def _masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_mask = geometric_noise(x, self.masking_len, self.masking_ratio)
        x = x * x_mask
        return x, ~x_mask.type(torch.bool)

    @staticmethod
    def _prepare_batch(batch):
        x = batch["x"]
        c = None
        if "c" in batch:
            c = batch["c"]
        padding_mask = None
        if "padding_mask" in batch:
            padding_mask = batch["padding_mask"]
        target = batch["target"]
        return x, target, c, padding_mask

    def _common_step(self, batch):
        x, target, c, padding_mask = self._prepare_batch(batch)

        logits = self(x, c=c, padding_mask=padding_mask)
        pred = self.softmax(logits)

        loss = self.loss(logits, target)

        return loss, pred, target
