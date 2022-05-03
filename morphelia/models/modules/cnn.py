from typing import Union, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .transformation import PermuteAxis
from ._utils import add_condition, MultOutSequential, PosArgSequential


class ConvBlock(nn.Module):
    """Conditional convolutional Block with optional average pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        n_conditions: int = 0,
        padding: int = 0,
        norm: bool = True,
        pool: bool = True,
        pool_method: str = "avg",
        pooling_kernel_size: int = 2,
    ):
        super().__init__()

        self.n_conditions = n_conditions

        if pool_method == "avg":
            self.pooling = nn.AvgPool1d(pooling_kernel_size)
        elif pool_method == "max":
            self.pooling = nn.MaxPool1d(pooling_kernel_size, return_indices=True)
        else:
            NotImplementedError(
                f"pool_method should be 'max' or 'avg', got {pool_method}"
            )

        self.conv = nn.Conv1d(
            in_channels, out_channels, (kernel_size,), padding=padding
        )
        if self.n_conditions > 0:
            self.cond_conv = nn.Conv1d(
                n_conditions, out_channels, (kernel_size,), padding=padding, bias=False
            )

        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.norm = norm
        self.pool = pool
        self.pool_method = pool_method

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Tensor of shape [batch size, channel dimensions, sequence length].

        Returns:
            Returns the output tensor and indices if 'pool_method' is 'max'.
            Otherwise only the output is tensor is returned.
        """
        if self.n_conditions == 0:
            x = self.conv(x)
        else:
            x, condition = torch.split(
                x, [x.shape[1] - self.n_conditions, self.n_conditions], dim=1
            )
            x = self.conv(x) + self.cond_conv(condition)

        if self.norm:
            x = self.batch_norm(x)
        x = self.act(x)
        if self.pool and x.size(-1) > 1:
            x = self.pooling(x)
        return x


class DeconvBlock(nn.Module):
    """Conditional deconvolutional Block with optional average pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        n_conditions: int = 0,
        padding: int = 0,
        norm: bool = True,
        pool: bool = True,
        pool_method: str = "avg",
        pooling_kernel_size: int = 2,
    ):
        super().__init__()

        self.n_conditions = n_conditions

        if pool_method == "avg":
            self.pooling = nn.Upsample(
                scale_factor=pooling_kernel_size, mode="linear", align_corners=True
            )
        elif pool_method == "max":
            self.pooling = nn.MaxUnpool1d(kernel_size=pooling_kernel_size)
        else:
            NotImplementedError(
                f"pool_method should be 'max' or 'avg', got {pool_method}"
            )

        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, (kernel_size,), padding=(padding,)
        )
        if self.n_conditions > 0:
            self.cond_conv = nn.ConvTranspose1d(
                n_conditions,
                out_channels,
                (kernel_size,),
                padding=(padding,),
                bias=False,
            )

        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.norm = norm
        self.pool = pool
        self.pool_method = pool_method

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch size, channel dimensions, sequence length].
            kwargs: Passed to the pooling layer. Could be indices if 'pooling_method'
                is 'max'.
        """
        if self.n_conditions == 0:
            if self.pool:
                x = self.pooling(x, **kwargs)
            x = self.conv(x)
        else:
            x, condition = torch.split(
                x, [x.shape[1] - self.n_conditions, self.n_conditions], dim=1
            )
            if self.pool:
                x = self.pooling(x, **kwargs)
                pad_len = x.shape[-1] - condition.shape[-1]
                condition = F.pad(condition, (0, pad_len), mode="replicate")
            x = self.conv(x) + self.cond_conv(condition)

        if self.norm:
            x = self.batch_norm(x)
        x = self.act(x)
        return x


class ConvEncoder(nn.Module):
    """
    Sequentially added convolutional layers for Encoding.
    The first layer is an optional conditional layer and adds conditions to the model.
    """

    def __init__(
        self,
        channel_dims: list,
        latent_dim: int,
        seq_len: int,
        kernel_size: Union[int, list],
        pool_method: str = "avg",
        pooling_kernel_size: Union[int, list] = 2,
        n_conditions: int = 0,
        final_pool: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.permute = PermuteAxis([0, 2, 1])
        self.channel_dims = channel_dims
        self.n_conditions = n_conditions

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(channel_dims)
        assert len(kernel_size) == len(channel_dims), (
            f"Expected {len(channel_dims)} kernel sizes for {len(channel_dims)} channel dimensions, "
            f"got {len(kernel_size)}"
        )

        if isinstance(pooling_kernel_size, int):
            pooling_kernel_size = [pooling_kernel_size] * len(channel_dims)
        assert len(pooling_kernel_size) == len(channel_dims), (
            f"Expected {len(channel_dims)} pooling kernel sizes for {len(channel_dims)} channel dimensions, "
            f"got {len(pooling_kernel_size)}"
        )

        if final_pool:
            final_pooling_kernel_size = seq_len
            if len(pooling_kernel_size) > 1:
                for pk in pooling_kernel_size[:-1]:
                    final_pooling_kernel_size = final_pooling_kernel_size // pk

            pooling_kernel_size[-1] = final_pooling_kernel_size

        self.encoder = None
        if len(channel_dims) > 1:
            self.encoder = []

            for i, (in_channels, out_channels, ks, pks) in enumerate(
                zip(
                    channel_dims[:-1],
                    channel_dims[1:],
                    kernel_size,
                    pooling_kernel_size,
                )
            ):
                padding = (ks - 1) // 2

                if i == 0:
                    self.encoder.append(
                        ConvBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=ks,
                            n_conditions=self.n_conditions,
                            padding=padding,
                            pooling_kernel_size=pks,
                            pool_method=pool_method,
                            **kwargs,
                        )
                    )
                else:
                    self.encoder.append(
                        ConvBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=ks,
                            padding=padding,
                            pooling_kernel_size=pks,
                            pool_method=pool_method,
                            **kwargs,
                        )
                    )

            self.encoder = MultOutSequential(*self.encoder)

        self.output = nn.Linear(channel_dims[-1], latent_dim)
        self.pooling_kernel_size = pooling_kernel_size
        self.pool_method = pool_method
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        return_indices: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Tensor of shape [batch size, sequence length, feature size]
            c: Tensor of shape [batch size,]
            return_indices: Return indices if pooling method is 'max'.
        """
        # concatenate x and condition
        if c is not None:
            x = add_condition(x, c, self.n_conditions)

        x = self.permute(x)
        indices = None
        if self.encoder is not None:
            x = self.encoder(x)
            if self.pool_method == "max":
                x, indices = x
        x = self.permute(x)

        if return_indices and indices is not None:
            return self.output(x), indices
        return self.output(x)


class ConvVAEEncoder(ConvEncoder):
    """
    1d-Convolutional Neural Network as Variational Autoencoder.
    """

    def __init__(self, channel_dims: list, latent_dim: int, **kwargs):
        super().__init__(channel_dims=channel_dims, latent_dim=latent_dim, **kwargs)

        self.mean_encoder = nn.Linear(channel_dims[-1], latent_dim)
        self.logvar_encoder = nn.Linear(channel_dims[-1], latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        return_statistic: bool = True,
        return_indices: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Args:
            x: Tensor of shape [batch size, sequence length, feature size]
            c: Tensor of shape [batch size,]
            return_statistic: Return means ond log-variances together with z.
            return_indices: Return indices if pooling method is 'max'.
        """
        # concatenate x and condition
        if c is not None:
            x = add_condition(x, c, self.n_conditions)

        x = self.permute(x)
        indices = None
        if self.encoder is not None:
            x = self.encoder(x)
            if self.pool_method == "max":
                x, indices = x
        x = self.permute(x)

        mean = self.mean_encoder(x)
        log_var = self.logvar_encoder(x)
        if indices is not None and return_indices:
            return mean, log_var, indices
        return mean, log_var


class CNNClassifier(ConvEncoder):
    """
    CNN Classifier wit convolutional encodings.
    """

    def __init__(self, n_classes: int, channel_dims: list, **kwargs):
        super().__init__(channel_dims=channel_dims, final_pool=True, **kwargs)

        self.classify = nn.Linear(channel_dims[-1], n_classes)
        self.n_classes = n_classes

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        return_indices: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Args:
            x: Tensor of shape [batch size, sequence length, feature size]
            c: Tensor of shape [batch size,]
            return_indices: Return indices if pooling method is 'max'.

        Returns:
            Logits
        """
        # concatenate x and condition
        if c is not None:
            x = add_condition(x, c, self.n_conditions)

        x = self.permute(x)

        indices = None
        if self.encoder is not None:

            x = self.encoder(x)  # [B, 1, latent_dim]
            if self.pool_method == "max":
                x, indices = x
        self.permute(x)

        x = x.reshape(x.size(0), -1)  # [B, 1 x latent_dim]
        logits = self.classify(x)  # [B, n_classes]

        if indices is not None and return_indices:
            return logits, indices
        return logits


class DeconvDecoder(nn.Module):
    """
    Sequentially added deconvolution layers to reconstruct sequantial data.
    The first layer is an optional conditional layer and add conditions to the model.
    """

    def __init__(
        self,
        channel_dims: list,
        latent_dim: int,
        kernel_size: Union[int, list],
        pooling_kernel_size: Union[int, list] = 2,
        n_conditions: int = 0,
        pool_method: str = "avg",
        **kwargs,
    ):
        super().__init__()

        self.permute = PermuteAxis([0, 2, 1])
        self.n_conditions = n_conditions
        channel_dims = [latent_dim] + channel_dims

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * (len(channel_dims) - 1)
        assert len(kernel_size) == (len(channel_dims) - 1), (
            f"Expected {len(channel_dims) - 1} kernel sizes for {len(channel_dims)} channel dimensions, "
            f"got {len(kernel_size)}"
        )

        if isinstance(pooling_kernel_size, int):
            pooling_kernel_size = [pooling_kernel_size] * (len(channel_dims) - 1)
        assert len(pooling_kernel_size) == (len(channel_dims) - 1), (
            f"Expected {len(channel_dims) - 1} pooling kernel sizes for {len(channel_dims)} channel dimensions, "
            f"got {len(pooling_kernel_size)}"
        )

        self.decoder = None
        if len(channel_dims) > 2:
            self.decoder = []

            for i, (in_channels, out_channels, ks, pks) in enumerate(
                zip(
                    channel_dims[:-2],
                    channel_dims[1:-1],
                    kernel_size,
                    pooling_kernel_size,
                )
            ):
                padding = (ks - 1) // 2

                if i == 0:
                    self.decoder.append(
                        DeconvBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=ks,
                            n_conditions=self.n_conditions,
                            padding=padding,
                            pooling_kernel_size=pks,
                            pool_method=pool_method,
                            **kwargs,
                        )
                    )
                else:
                    self.decoder.append(
                        DeconvBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=ks,
                            padding=padding,
                            pooling_kernel_size=pks,
                            pool_method=pool_method,
                            **kwargs,
                        )
                    )

            self.decoder = PosArgSequential(*self.decoder)

        self.output = nn.Sequential(
            nn.Linear(channel_dims[-2], channel_dims[-1]), nn.ReLU()
        )
        self.pooling_kernel_size = pooling_kernel_size
        self.pool_method = pool_method
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

    def forward(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch size, sequence length, feature size]
            c: Tensor of shape [batch size,]
            kwargs: Passed to the pooling layer.
                Could be indices if 'pooling_method' is 'max'.
                In this case, kwargs should hold the keyword 'indices' with a list
                containing indices for every layer of the model.
        """
        # concatenate x and condition
        if c is not None:
            x = add_condition(x, c, self.n_conditions)

        x = self.permute(x)
        if self.decoder is not None:
            x = self.decoder(x, **kwargs)
        x = self.permute(x)
        return self.output(x)
