from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


def kld(mu: torch.Tensor, log_var: torch.Tensor, eps: float = 1e-5):
    var = torch.exp(log_var) + eps
    kld = (
        kl_divergence(
            Normal(mu, var.sqrt()), Normal(torch.zeros_like(mu), torch.ones_like(var))
        )
        .sum(dim=1)
        .mean()
    )
    return kld


def sampling(
    mu: torch.Tensor, log_var: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Sampling from a normal distribution using the reparameterization trick.

    Args:
        mu: Means.
        log_var: Logarithmic variances.
        eps: Avoid zero variances.

    Returns:
        Samples
    """
    var = torch.exp(log_var) + eps
    return Normal(mu, var.sqrt()).rsample()


class CondLayer(nn.Module):
    """Implements a conditional layer."""

    def __init__(
        self, in_features: int, out_features: int, n_conditions: int, bias: bool
    ):
        super().__init__()
        self.n_conditions = n_conditions
        self.layer = nn.Linear(in_features, out_features, bias=bias)
        self.cond_layer = nn.Linear(self.n_conditions, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_conditions == 0:
            return self.layer(x)
        else:
            x, condition = torch.split(
                x, [x.shape[1] - self.n_cond, self.n_cond], dim=1
            )
            return self.layer(x) + self.cond_layer(condition)


class Encoder(nn.Module):
    """
    Implements the Encoder Module for a conditional VAE.

    Args:
        layer_dims: List with number of hidden features.
        latent_dim: Number of features in latent space.
        n_conditions: Number of conditions. Vanilla VAE if 0.
        batch_norm: Add batch normalization.
        layer_norm: Add layer normalization.
        dropout: Add dropout layer.
    """

    def __init__(
        self,
        layer_dims: list,
        latent_dim: int,
        n_conditions: int = 0,
        batch_norm: bool = False,
        layer_norm: bool = True,
        dropout: float = None,
    ) -> None:
        super().__init__()
        self.n_conditions = n_conditions

        # dynamically append modules
        self.encode = None
        if len(layer_dims) > 0:
            self.encode = []

            for i, (in_features, out_features) in enumerate(
                zip(layer_dims[:-1], layer_dims[1:])
            ):
                if i == 0:
                    self.encode.append(
                        CondLayer(
                            in_features=in_features,
                            out_features=out_features,
                            n_conditions=self.n_conditions,
                            bias=True,
                        )
                    )
                else:
                    self.encode.append(
                        nn.Linear(
                            in_features=in_features,
                            out_features=out_features,
                            bias=True,
                        )
                    )

                if batch_norm:
                    self.encode.append(nn.BatchNorm1d(out_features, affine=True))
                if layer_norm:
                    self.encode.append(
                        nn.LayerNorm(out_features, elementwise_affine=False)
                    )

                self.encode.append(nn.ReLU())

                if dropout is not None:
                    self.encode.append(nn.Dropout(dropout))

            self.encode = nn.Sequential(*self.encode)

        self.layer_dims = layer_dims
        self.latent_dim = latent_dim
        self.output = nn.Linear(layer_dims[-1], latent_dim)

    def forward(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape [batch size, feature size]
            c: Tensor of shape [batch size,]
        """
        # concatenate x and condition
        if c is not None:
            c = F.one_hot(c, num_classes=self.n_conditions)
            x = torch.cat((x, c), dim=-1)

        if self.encode is not None:
            x = self.encode(x)

        return self.output(x)


class VAEEncoder(Encoder):
    """
    Implements the Encoder Module for a Variational Autoencoder.

    Outputs means and log-variances instead of z.
    """

    def __init__(self, **kwargs):
        super(VAEEncoder, self).__init__(**kwargs)
        self.mean_encoder = nn.Linear(self.layer_dims[-1], self.latent_dim)
        self.logvar_encoder = nn.Linear(self.layer_dims[-1], self.latent_dim)

    def forward(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape [batch size x feature size]
            c: Tensor of shape [batch size,]
        """
        # concatenate x and condition
        if c is not None:
            c = F.one_hot(c, num_classes=self.n_conditions)
            x = torch.cat((x, c), dim=1)

        if self.encode is not None:
            x = self.encode(x)

        mean = self.mean_encoder(x)
        logvar = self.logvar_encoder(x)
        return mean, logvar


class MMDDecoder(nn.Module):
    """
    Implements the split Decoder Module for a conditional VAE,
    that can be used to calculate the MMD loss.

    Args:
        layer_dims: List with number of hidden features.
        latent_dim: Number of features in latent space.
        n_conditions: Number of conditions. Vanilla VAE if 0.
        batch_norm: Add batch normalization.
        layer_norm: Add layer normalization.
        dropout: Add dropout layer.
    """

    def __init__(
        self,
        layer_dims: list,
        latent_dim: int,
        n_conditions: int = 0,
        batch_norm: bool = False,
        layer_norm: bool = True,
        dropout: float = None,
    ):
        super().__init__()
        self.n_conditions = n_conditions
        layer_dims = [latent_dim] + layer_dims

        # create first layer (g1)
        self.decode1 = []
        self.decode1.append(
            CondLayer(
                in_features=layer_dims[0],
                out_features=layer_dims[1],
                n_conditions=self.n_conditions,
                bias=False,
            )
        )

        if batch_norm:
            self.decode1.append(nn.BatchNorm1d(layer_dims[1], affine=True))
        if layer_norm:
            self.decode1.append(nn.LayerNorm(layer_dims[1], elementwise_affine=False))
        self.decode1.append(nn.ReLU())
        if dropout is not None:
            self.decode1.append(nn.Dropout(dropout))
        self.decode1 = nn.Sequential(*self.decode1)

        # create all other layers (g2)
        self.decode2 = None
        if len(layer_dims) > 2:
            self.decode2 = []

            for i, (in_features, out_features) in enumerate(
                zip(layer_dims[1:-1], layer_dims[2:])
            ):
                if i + 3 < len(layer_dims):
                    self.decode2.append(
                        nn.Linear(
                            in_features=in_features,
                            out_features=out_features,
                            bias=False,
                        )
                    )

                    if batch_norm:
                        self.decode2.append(nn.BatchNorm1d(out_features, affine=True))
                    if layer_norm:
                        self.decode2.append(
                            nn.LayerNorm(out_features, elementwise_affine=False)
                        )

                    self.decode2.append(nn.ReLU())

                    if dropout is not None:
                        self.decode2.append(nn.Dropout(dropout))

            self.decode2 = nn.Sequential(*self.decode2)

        self.output = nn.Sequential(
            nn.Linear(layer_dims[-2], layer_dims[-1]), nn.ReLU()
        )

    def forward(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape [batch size x feature size]
            c: Tensor of shape [batch size,]
        """
        # concatenate x and condition
        if c is not None:
            c = F.one_hot(c, num_classes=self.n_conditions)
            x = torch.cat((x, c), dim=1)

        y_hat = self.decode1(x)

        if self.decode2 is not None:
            x_hat = self.decode2(y_hat)
        else:
            x_hat = y_hat

        x_hat = self.output(x_hat)
        return x_hat, y_hat
