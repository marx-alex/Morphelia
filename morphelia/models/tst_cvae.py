from typing import Optional, Tuple

import torch
from torch import nn
import pytorch_lightning as pl

from . import sampling, MaskedMSELoss, MMDLoss, kl_divergence, geometric_noise


class CVAET(pl.LightningModule):
    """
    Conditional Variational Autoencoder with a Time Series Transformer.
    """

    def __init__(
        self,
        encoder: nn.Module,
        transformer: nn.Module,
        n_conditions: int,
        n_classes: Optional[int] = None,
        decoder: Optional[nn.Module] = None,
        kld_alpha: float = 1.0,
        mmd_alpha: float = 1.0,
        mask: bool = True,
        masking_len=3,
        masking_ratio=0.15,
        learning_rate: float = 1e-4,
        optimizer: str = "Adam",
    ) -> None:
        super(CVAET, self).__init__()

        self.save_hyperparameters("learning_rate", "optimizer")
        self.n_classes = n_classes
        self.kld_alpha = kld_alpha
        self.mmd_alpha = mmd_alpha
        self.mask = mask
        self.masking_len = masking_len
        self.masking_ratio = masking_ratio

        self.encoder = encoder  # [B x S x F] -> [B x S x latent_dim]
        self.transformer = transformer  # [B x S x latent_dim] -> [B x S x latent_dim]
        self.decoder = decoder  # [B x S x latent_dim] -> [B x S x F]

        layer_dims = encoder.layer_dims
        latent_dim = encoder.latent_dim
        self.mean_encoder = nn.Linear(layer_dims[-1], latent_dim)
        self.log_var_encoder = nn.Linear(layer_dims[-1], latent_dim)

        # initiate loss functions
        self.mse = MaskedMSELoss()
        self.mmd = MMDLoss(n_conditions=n_conditions)
        self.kld = kl_divergence

    def forward_enc(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s, f = x.shape
        if c is not None:
            c = torch.tile(c.reshape(-1, 1), (1, s))  # [B,] -> [B, S]
            c = c.reshape(b * s, -1)  # [B, S] -> [B x S,]
        x = x.reshape(b * s, -1)  # [B, S, F] -> [B x S, F]

        z = self.encoder(x, c=c)  # [B x S, latent_dim]
        z = z.reshape(b, s, -1)  # [B x S, latent_dim] -> [B, S, latent_dim]
        z = self.transformer(z, key_padding_mask=padding_mask)

        mean = self.mean_encoder(z)
        log_var = self.log_var_encoder(z)

        return mean, log_var

    def forward_dec(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s, f = z.shape
        if c is not None:
            c = torch.tile(c.reshape(-1, 1), (1, s))  # [B,] -> [B, S]
            c = c.reshape(b * s, -1)  # [B, S] -> [B x S,]
        z = z.reshape(b * s, -1)  # [B, S, latent_dim] -> [B x S, latent_dim]

        x_hat, y_hat = self.decoder(z, c=c)  # [B x S, F]
        x_hat = x_hat.reshape(b, s, -1)  # [B x S, F] -> [B, S, F]
        return x_hat, y_hat

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

        z_mean, z_log_var = self.forward_enc(x, c=c, padding_mask=padding_mask)
        z = sampling(z_mean, z_log_var)
        x_hat, y_hat = self.forward_dec(z, c=c)

        return z, z_mean, z_log_var, x_hat, y_hat

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch)
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
        c = batch["c"]
        padding_mask = batch["padding_mask"]
        return x, c, padding_mask

    def _common_step(self, batch):
        x, c, padding_mask = self._prepare_batch(batch)
        target_x = x.clone()
        if self.mask:
            x, x_mask = self._masking(x)
        z, z_mean, z_log_var, x_hat, y_hat = self(x, c=c, padding_mask=padding_mask)

        b, s, f = z.shape
        if c is not None:
            c = torch.tile(c.reshape(-1, 1), (1, s))  # [B,] -> [B, S]
            c = c.reshape(b * s, -1)  # [B, S] -> [B x S,]
            mmd = self.mmd(y_hat, c)
        else:
            mmd = torch.tensor(0.0, device=z.device)

        if self.mask:
            target_mask = x_mask * ~padding_mask.unsqueeze(-1)
        else:
            target_mask = torch.ones_like(target_x) * ~padding_mask.unsqueeze(-1)
        mse = self.mse(x_hat, target_x, mask=target_mask)
        kld = self.kld(z_mean, z_log_var)

        loss = mse + (self.kld_alpha * kld) + (self.mmd_alpha * mmd)

        self.log_dict({"loss": loss, "mse": mse, "kld": kld, "mmd": mmd})

        return loss
