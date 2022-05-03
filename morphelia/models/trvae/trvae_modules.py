from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch.distributions import Normal
import pytorch_lightning as pl

from ..utils import ConditionalMMDLoss, KLDLoss


class TRVAE(pl.LightningModule):
    """
    trVAE as described by Lotfollahi et al., 2020, Bioinformatics
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        kld_beta: float = 1.0,
        mmd_beta: float = 1.0,
        mse_beta: float = 1.0,
        learning_rate: float = 1e-3,
        optimizer: str = "Adam",
    ):
        super().__init__()

        self.save_hyperparameters("learning_rate", "optimizer")
        self.kld_beta = kld_beta
        self.mmd_beta = mmd_beta
        self.mse_beta = mse_beta

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = encoder.latent_dim

        # initiate loss functions
        self.mse = nn.MSELoss()
        self.mmd = ConditionalMMDLoss(n_conditions=encoder.n_conditions)
        self.kld = KLDLoss()

    @staticmethod
    def reparameterize(
        mu: torch.Tensor, log_sigma: torch.Tensor, eps: float = 1e-5
    ) -> torch.Tensor:
        sigma = torch.exp(log_sigma) + eps
        return Normal(mu, sigma.sqrt()).rsample()

    def forward_features(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        return_statistics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        z_mean, z_logvar = self.encoder(x, c=c)
        z = self.reparameterize(z_mean, z_logvar)

        if return_statistics:
            return z, z_mean, z_logvar
        return z

    def forward(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, z_mean, z_logvar = self.forward_features(x, c=c, return_statistics=True)
        x_hat, y_hat = self.decoder(z, c=c)
        return z, z_mean, z_logvar, x_hat, y_hat

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, prefix="valid")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, prefix="test")
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
        c = None
        if "c" in batch:
            c = batch["c"]
        return x, c

    def _common_step(self, batch, prefix="train"):
        x, c = self._prepare_batch(batch)

        z, z_mean, z_log_var, x_hat, y_hat = self(x, c=c)

        if c is not None:
            mmd = self.mmd(y_hat, c)
        else:
            mmd = torch.tensor(0.0, device=z.device)

        mse = self.mse(x_hat, x)
        kld = self.kld(z_mean, z_log_var)

        loss = (self.mse_beta * mse) + (self.kld_beta * kld) + (self.mmd_beta * mmd)

        self.log_dict(
            {
                f"{prefix}loss": loss,
                f"{prefix}mse": mse,
                f"{prefix}kld": kld,
                f"{prefix}mmd": mmd,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss
