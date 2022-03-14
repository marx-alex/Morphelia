from typing import Tuple, Optional

import torch
from torch import nn
from torch.distributions import Normal

from . import PermuteAxis
from ._utils import add_condition


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
    """Implements a conditional layer.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        n_conditions: Absolute number of conditions.
        bias: Learn bias for the layer. Default is True.
            This das not affect the conditional layer where no additive bias is learned.
    """

    def __init__(
        self, in_features: int, out_features: int, n_conditions: int, bias: bool = True
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
                x, [x.shape[-1] - self.n_conditions, self.n_conditions], dim=-1
            )
            return self.layer(x) + self.cond_layer(condition)


class Encoder(nn.Module):
    """
    Implements the Encoder Module for a conditional VAE.

    Args:
        layer_dims: List with number of hidden features.
            Last number is number of features in hidden space.
        latent_dim: Number of features in latent space.
        n_conditions: Number of conditions. Vanilla VAE if 0.
        sequential: Permute Axis before BatchNorm if sequential.
        batch_norm: Add batch normalization.
        layer_norm: Add layer normalization.
        dropout: Add dropout layer.
    """

    def __init__(
        self,
        layer_dims: list,
        latent_dim: int,
        n_conditions: int = 0,
        sequential: bool = False,
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
                    if sequential:
                        self.encode.append(PermuteAxis([0, 2, 1]))
                    self.encode.append(nn.BatchNorm1d(out_features, affine=True))
                    if sequential:
                        self.encode.append(PermuteAxis([0, 2, 1]))
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
            x = add_condition(x, c, self.n_conditions)

        if self.encode is not None:
            x = self.encode(x)

        return self.output(x)


class VAEEncoder(Encoder):
    """
    Implements the Encoder Module for a Variational Autoencoder.

    Outputs means and log-variances instead of z.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
            x = add_condition(x, c, self.n_conditions)

        if self.encode is not None:
            x = self.encode(x)

        mean = self.mean_encoder(x)
        log_var = self.logvar_encoder(x)
        return mean, log_var


class MMDDecoder(nn.Module):
    """
    Implements the split Decoder Module for a conditional VAE,
    that can be used to calculate the MMD loss.

    Args:
        layer_dims: List with number of hidden features.
        latent_dim: Number of features in latent space.
        n_conditions: Number of conditions. Vanilla VAE if 0.
        sequential: Permute Axis before BatchNorm if sequential.
        batch_norm: Add batch normalization.
        layer_norm: Add layer normalization.
        dropout: Add dropout layer.
    """

    def __init__(
        self,
        layer_dims: list,
        latent_dim: int,
        n_conditions: int = 0,
        sequential: bool = False,
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
            if sequential:
                self.decode1.append(PermuteAxis([0, 2, 1]))
            self.decode1.append(nn.BatchNorm1d(layer_dims[1], affine=True))
            if sequential:
                self.decode1.append(PermuteAxis([0, 2, 1]))
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
                        if sequential:
                            self.decode2.append(PermuteAxis([0, 2, 1]))
                        self.decode2.append(nn.BatchNorm1d(out_features, affine=True))
                        if sequential:
                            self.decode2.append(PermuteAxis([0, 2, 1]))
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
            x = add_condition(x, c, self.n_conditions)

        y_hat = self.decode1(x)

        if self.decode2 is not None:
            x_hat = self.decode2(y_hat)
        else:
            x_hat = y_hat

        x_hat = self.output(x_hat)
        return x_hat, y_hat


class Decoder(nn.Module):
    """
    Decoder Module.

    Args:
        layer_dims: List with number of hidden features.
        latent_dim: Number of features in latent space.
        n_conditions: Number of conditions. Vanilla VAE if 0.
        sequential: Permute Axis before BatchNorm if sequential.
        batch_norm: Add batch normalization.
        layer_norm: Add layer normalization.
        dropout: Add dropout layer.
    """

    def __init__(
        self,
        layer_dims: list,
        latent_dim: int,
        n_conditions: int = 0,
        sequential: bool = False,
        batch_norm: bool = False,
        layer_norm: bool = True,
        dropout: float = None,
    ):
        super().__init__()
        self.n_conditions = n_conditions
        layer_dims = [latent_dim] + layer_dims

        # dynamically append modules
        self.decode = None
        if len(layer_dims) > 0:
            self.decode = []

            for i, (in_features, out_features) in enumerate(
                zip(layer_dims[:-2], layer_dims[1:-1])
            ):
                if i == 0:
                    self.decode.append(
                        CondLayer(
                            in_features=layer_dims[0],
                            out_features=layer_dims[1],
                            n_conditions=self.n_conditions,
                            bias=False,
                        )
                    )

                else:

                    self.decode.append(
                        nn.Linear(
                            in_features=in_features,
                            out_features=out_features,
                            bias=False,
                        )
                    )

                if batch_norm:
                    if sequential:
                        self.decode.append(PermuteAxis([0, 2, 1]))
                    self.decode.append(nn.BatchNorm1d(out_features, affine=True))
                    if sequential:
                        self.decode.append(PermuteAxis([0, 2, 1]))
                if layer_norm:
                    self.decode.append(
                        nn.LayerNorm(out_features, elementwise_affine=False)
                    )

                self.decode.append(nn.ReLU())

                if dropout is not None:
                    self.decode.append(nn.Dropout(dropout))

            self.decode = nn.Sequential(*self.decode)

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
            x = add_condition(x, c, self.n_conditions)

        if self.decode is not None:
            x_hat = self.decode(x)
        else:
            x_hat = x

        x_hat = self.output(x_hat)
        return x_hat
