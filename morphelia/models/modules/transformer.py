from typing import Optional, Tuple, Union
import math

import torch
from torch import nn

from . import PermuteAxis, Reshape
from .vae import sampling
from ._utils import MultOutSequential, ArgSequential


def _get_pos_encoder(pos_encoding: str):
    pos_encoding = pos_encoding.lower()
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding
    raise NotImplementedError(
        f"Positional encoding can be either 'learnable' or 'fixed', instead got {pos_encoding}"
    )


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    """Implements positional encodings from sine and cosine functions of different frequencies.

    Parameters
    ----------
    dim : int
        Number of features
    dropout : float
        Dropout value
    seq_len : int
        Maximum length of the incoming sequence
    scale_factor : float
        Scaling factor
    """

    def __init__(
        self,
        dim: int,
        seq_len: int,
        dropout: float = 0.1,
        scale_factor: float = 1.0,
    ) -> None:
        super(FixedPositionalEncoding, self).__init__()

        # positional encoding
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `[N x S x F]`, where `S` is the sequence length,
            `N` is the batch size and `F` is the number of features
        Returns
        -------
        torch.Tensor
            Tensor of same shape as input tensor
        """
        x = x + self.pe
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Implements the learnable positional encoding Module.

    Parameters
    ----------
    dim : int
        Number of features
    dropout : float
        Dropout value
    seq_len : int
        Maximum length of the incoming sequence
    """

    def __init__(self, dim: int, seq_len: int, dropout: float = 0.1) -> None:
        super(LearnablePositionalEncoding, self).__init__()

        # positional encoding
        self.pe = self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `[N x S x F]`, where `S` is the sequence length,
            `N` is the batch size and `F` is the number of features
        Returns
        -------
        torch.Tensor
            Tensor of same shape as input tensor
        """
        x = x + self.pe
        return self.dropout(x)


class ResNormBlock(nn.Module):
    """Transformer block for sequential data.

    Parameters
    ----------
    dim : int
        Number of features
    fn : pytorch.nn.Module
        Module to wrap in the block
    dropout : float
        Include dropout
    norm : str
        `batch_norm` for batch normalization, `layer_norm` for layer normalization
    """

    def __init__(
        self, dim: int, fn: torch.nn.Module, dropout: float = 0.0, norm: str = "layer"
    ) -> None:
        super(ResNormBlock, self).__init__()

        norm = norm.lower()
        if norm == "layer":
            self.norm = nn.LayerNorm(dim)
        elif norm == "batch":
            self.norm = nn.Sequential(
                PermuteAxis([0, 2, 1]),  # [B, F, S]
                nn.BatchNorm1d(dim),
                PermuteAxis([0, 2, 1]),  # [B, S, F]
            )
        else:
            raise NotImplementedError(
                f"Normalization should be either 'batch' or 'layer', instead got {norm}"
            )
        self.dropout = nn.Dropout(dropout)
        self.fn = fn

    def forward(
        self, x: torch.Tensor, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Pass tensor through module.

        Parameters
        ----------
        x : torch.Tensor
        **kwargs
            Keyword arguments are passed to the layer

        Returns
        -------
        torch.Tensor, torch.Tensor
             Output tensor and attention if layer returns a tuple
        """
        z = self.norm(x)
        z = self.fn(z, **kwargs)
        z = self.dropout(z)
        # if tuple, pass attention
        if isinstance(z, tuple):
            z, attn = z
            return z + x, attn
        return z + x


class Attention(nn.Module):
    """This class implements an attention module.

    Parameters
    ----------
    dim : int
        Number of features
    heads : int
        Number of heads
    dropout : float
        Dropout rate

    References
    ----------
    .. [1] Attention is all you need. A. Vaswani et al.
       Advances in Neural Information Processing Systems , page 5998--6008. (2017)
    """

    def __init__(self, dim: int, heads: int, dropout=0.0) -> None:
        super(Attention, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(dim, 3 * dim),  # [B, S, 3*F]
            Reshape(3, dim, fixed=2),  # [B, S, 3, F]
            PermuteAxis(2, 0, 1, 3),  # [3, B, S, F]
        )  # [B, S, F] -> [3, B, S, F]

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )  # [3, B, S, F] -> [B, S, F]

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Pass tensor through module.

        Parameters
        ----------
            x : torch.Tensor
                Input tensor
            key_padding_mask : torch.Tensor
            attn_mask : torch.Tensor
            return_attention : bool

        Returns
        -------
        torch.Tensor, torch.Tensor
            Output tensor and attention if `return_attention` it True
        """
        if isinstance(x, tuple):
            q, k, v = x
        else:
            q, k, v = self.linear(x)  # create query, key and value
        x, w = self.multi_head_attention(
            q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )  # new sequence
        if return_attention:
            return x, w
        return x


class FeedForward(nn.Module):
    """This class implements the multi-layer perceptron, which is applied after
    the Multi-Head Attention.

    Parameters
    ----------
    dim : int
        Number of features
    hidden_dim : int
        Number of hidden features
    dropout : float
        Dropout rate
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )  # [B, S, F] -> [B, S, F]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass tensor through module.

        Parameters
        ----------
            x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer.

    Parameters
    ----------
    input_dim : int
        Number of input features
    dim_feedforward : int
        Dimensions of the MLP
    nhead : int
        attention heads (dim % heads == 0)
    ff_drop : float
        Feat forward drop
    att_drop : float
        Attention drop
    norm : str
        Normalization method. `layer` or `batch` - normalization.
    """

    def __init__(
        self,
        input_dim: int,
        dim_feedforward: int,
        nhead: int,
        ff_drop: float = 0.0,
        att_drop: float = 0.0,
        connection_drop: float = 0.0,
        norm: str = "layer",
    ) -> None:
        super().__init__()

        self.attn = ResNormBlock(
            input_dim,
            Attention(input_dim, nhead, att_drop),
            dropout=connection_drop,
            norm=norm,
        )
        self.ff = ResNormBlock(
            input_dim,
            FeedForward(input_dim, dim_feedforward, ff_drop),
            dropout=connection_drop,
            norm=norm,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Pass tensor through module.

        Parameters
        ----------
        x : torch.Tensor
        key_padding_mask : torch.Tensor
        attn_mask : torch.Tensor
        return_attention : bool

        Returns
        -------
        torch.Tensor, torch.Tensor
            Output tensor and attention tensor if `return_attention` is True
        """
        z = self.attn(
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            return_attention=return_attention,
        )
        if return_attention:
            z, attn = z
            return self.ff(z), attn
        return self.ff(z)


class TransformerEncoder(nn.Module):
    """Transformer Encoder.

    Implements the Transformer encoder with additional features.
    This is similar to pytorch's TransformerEncoder but with batch normalization
    and fixed as well as learnable positional embedding.

    Parameters
    ----------
    input_dim : int
        The number of expected features in the input
    seq_len : int
        Sequence length
    nhead : int
        The number of heads in the multihead attention layer.
        Embedded dimensions must be divisible by number of heads.
    dim_feedforward : int, optional
        The dimension of the feedforward network model.
        The default dimension is 2 times the input dimensions.
    dropout : float
        The dropout value for the feedforward and attention layer
    pos_dropout : float
        The dropout value for the positional encoding
    norm : str
        Normalization method (`batch` or `layer`)
    pos_encoding : str, optional
        Positional encoding. Can be `learnable` or `fixed`.
    num_layers : int
        Depth of encoder

    References
    ----------
    .. [1] George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning,
       in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21),
       August 14--18, 2021.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        nhead: int = 1,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        pos_dropout: float = 0.1,
        norm: str = "batch",
        pos_encoding: Optional[str] = "learnable",
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len

        att_params_default = dict(
            input_dim=input_dim,
            dim_feedforward=int(2 * input_dim),
            nhead=nhead,
            ff_drop=dropout,
            att_drop=dropout,
            norm=norm,
            connection_drop=dropout,
        )
        if dim_feedforward is not None:
            att_params_default.update(dict(dim_feedforward=dim_feedforward))
        self.transformers = MultOutSequential(
            *[TransformerEncoderLayer(**att_params_default) for _ in range(num_layers)]
        )

        # positional encoding
        self.forward_emb = None
        if pos_encoding is not None:
            self.forward_emb = _get_pos_encoder(pos_encoding)(
                input_dim, seq_len=seq_len, dropout=pos_dropout
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Passes the input through the encoder modules.

        Parameters
        ----------
        x : torch.Tensor
            The sequence to the encoder modules
        attn_mask : torch.Tensor, optional
            The mask for the src sequence
        key_padding_mask : torch.Tensor, optional
            The mask for the src keys per batch
        return_attention : bool, optional
            Return the attention weights

        Returns
        -------
        torch.Tensor, torch.Tensor
            Output tensor and attention weights if `return_attention` is True
        """
        if self.forward_emb is not None:
            z = self.forward_emb(x)
        else:
            z = x

        z = self.transformers(
            z,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            return_attention=return_attention,
        )

        return z


class TVAEEncoder(TransformerEncoder):
    """Variation Autoencoder using a transformer encoder.

    Parameters
    ----------
    input_dim : int
        Number of input features
    seq_len : int
        Sequence length
    latent_dim : int
        Number of latent features
    **kwargs
        Keyword arguments are passed to morphelia.dl.modules.TransformerEncoder
    """

    def __init__(self, input_dim: int, seq_len: int, latent_dim: int, **kwargs):
        super().__init__(input_dim=input_dim, seq_len=seq_len, **kwargs)

        self.means, self.logvar = nn.Linear(input_dim, latent_dim), nn.Linear(
            input_dim, latent_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_statistics: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Passes the input through the encoder modules.

        Parameters
        ----------
        x : torch.Tensor
            The sequence to the encoder modules
        attn_mask : torch.Tensor, optional
            The mask for the src sequence
        key_padding_mask : torch.Tensor, optional
            The mask for the src keys per batch
        return_attention : bool
            Return the attention weights
        return_statistics : bool
            Return the means and log-variances

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            Output tensor, attention weights (if `return_attention` is True),
            means and log-variance (if `return_statistics` is True).
        """
        if self.forward_emb is not None:
            z = self.forward_emb(x)
        else:
            z = x

        z = self.transformers(
            z,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            return_attention=return_attention,
        )

        if return_attention:
            z, w = z

        z_means, z_logvar = self.means(z), self.logvar(z)
        z = sampling(z_means, z_logvar)

        if return_attention:
            return z, w
        if return_statistics:
            return z, z_means, z_logvar
        return z


class TransformerClassifier(TransformerEncoder):
    """Transformer Classifier.

    Parameters
    ----------
    input_dim : int
        Number of input features
    seq_len : int
        Sequence length
    n_classes : int
        Number of classes
    **kwargs
        Keyword arguments are passed to morphelia.dl.modules.TransformerEncoder
    """

    def __init__(self, input_dim: int, seq_len: int, n_classes: int, **kwargs):
        super().__init__(input_dim=input_dim, seq_len=seq_len, **kwargs)

        self.n_classes = n_classes
        self.classifier = nn.Linear(input_dim * seq_len, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Passes the input through the encoder modules.

        Parameters
        ----------
        x : torch.Tensor
            The sequence to the encoder modules
        attn_mask : torch.Tensor, optional
            The mask for the src sequence
        key_padding_mask : torch.Tensor, optional
            The mask for the src keys per batch
        return_attention : bool
            Return the attention weights

        Returns
        -------
        torch.Tensor, torch.Tensor
            Logits and attention weights (if `return_attention` is True)
        """
        if self.forward_emb is not None:
            z = self.forward_emb(x)
        else:
            z = x

        z = self.transformers(
            z,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            return_attention=return_attention,
        )

        if return_attention:
            z, w = z

        if key_padding_mask is not None:
            z = z * ~key_padding_mask.unsqueeze(-1)

        z = z.reshape(z.shape[0], -1)  # [N, S x F]
        pred = self.classifier(z)

        if return_attention:
            return pred, w
        return pred


class TransformerTokenClassifier(TransformerEncoder):
    """Transformer token classifier.

    Parameters
    ----------
    input_dim : int
        Number of input features
    seq_len : int
        Sequence length
    n_classes : int
        Number of classes
    pos_dropout : float
        Dropout rate
    pos_encoding : str, optional
        `learnable` or `fixed` positional encoding
    **kwargs
        Keyword arguments are passed to morphelia.dl.modules.TransformerEncoder
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        n_classes: int,
        pos_dropout: float = 0.1,
        pos_encoding: Optional[str] = "learnable",
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            pos_dropout=pos_dropout,
            pos_encoding=pos_encoding,
            **kwargs,
        )

        self.n_classes = n_classes
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, n_classes)
        )  # [B, F] -> [B, n_classes]

        # overwrite positional encoding
        self.forward_emb = None
        if pos_encoding is not None:
            self.forward_emb = _get_pos_encoder(pos_encoding)(
                input_dim, seq_len=seq_len + 1, dropout=pos_dropout
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Passes the input through the encoder modules.

        Parameters
        ----------
        x : torch.Tensor
            The sequence to the encoder modules
        attn_mask : torch.Tensor, optional
            The mask for the src sequence
        key_padding_mask : torch.Tensor, optional
            The mask for the src keys per batch
        return_attention : bool
            Return the attention weights

        Returns
        -------
        torch.Tensor, torch.Tensor
            Logits and attention weights (if `return_attention` is True)
        """
        # introduce classification token
        b, n, f = x.shape
        x = torch.cat(
            (self.cls_token.repeat(b, 1, 1), x), dim=1
        )  # [B, N, F] -> [B, N+1, F]

        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                (
                    torch.ones(b, 1, dtype=torch.bool, device=key_padding_mask.device),
                    key_padding_mask,
                ),
                dim=1,
            )

        if self.forward_emb is not None:
            x = self.forward_emb(x)

        z = self.transformers(
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            return_attention=return_attention,
        )

        if return_attention:
            z, w = z

        z = z[:, 0]  # [B,N+1,F] -> [B,F]
        pred = self.classifier(z)

        if return_attention:
            return pred, w
        return pred


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer.

    Parameters
    ----------
    input_dim : int
        Number of input features
    dim_feedforward : int
        Dimensions of the MLP
    nhead : int
        attention heads (dim % heads == 0)
    ff_drop : float
        Feat forward drop
    att_drop : float
        Attention drop
    connection_drop : float
        Connection drop
    norm : str
        Normalization method. `layer` or `batch` - normalization.
    """

    def __init__(
        self,
        input_dim: int,
        dim_feedforward: int,
        nhead: int,
        ff_drop: float = 0.0,
        att_drop: float = 0.0,
        connection_drop: float = 0.0,
        norm: str = "layer",
    ) -> None:
        super().__init__()

        self.self_attn = ResNormBlock(
            input_dim,
            Attention(input_dim, nhead, att_drop),
            dropout=connection_drop,
            norm=norm,
        )
        self.src_attn = ResNormBlock(
            input_dim,
            Attention(input_dim, nhead, att_drop),
            dropout=connection_drop,
            norm=norm,
        )
        self.ff = ResNormBlock(
            input_dim,
            FeedForward(input_dim, dim_feedforward, ff_drop),
            dropout=connection_drop,
            norm=norm,
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        src_attn_mask: Optional[torch.Tensor] = None,
        tgt_attn_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Pass tensor through module.

        Parameters
        ----------
        x : torch.Tensor
        memory : torch.Tensor
        key_padding_mask : torch.Tensor
        src_attn_mask : torch.Tensor
        tgt_attn_mask : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        x = self.self_attn(
            x, key_padding_mask=key_padding_mask, attn_mask=tgt_attn_mask
        )
        x = self.src_attn(
            (x, memory, memory),
            key_paddinging_mask=key_padding_mask,
            attn_mask=src_attn_mask,
        )

        return self.ff(x)


class TransformerDecoder(nn.Module):
    """Transformer Decoder.

    Implements the Transformer Decoder with additional features.
    This is similar to pytorch's TransformerDecoder but with batch normalization
    and fixed as well as learnable positional embedding.

    Parameters
    ----------
    input_dim : int
        The number of expected features in the input
    seq_len : int
        Sequence length
    nhead : int
        The number of heads in the multihead attention layer.
        Embedded dimensions must be divisible by number of heads.
    dim_feedforward : int, optional
        The dimension of the feedforward network model.
        The default dimension is 2 times the input dimensions.
    dropout : float
        The dropout value for the feedforward and attention layer
    pos_dropout : float
        The dropout value for the positional encoding
    norm : str
        Normalization method (`batch` or `layer`)
    pos_encoding : str, optional
        Positional encoding. Can be `learnable` or `fixed`.
    num_layers : int
        Depth of encoder

    References
    ----------
    .. [1] George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning,
       in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21),
       August 14--18, 2021.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        nhead: int = 1,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        pos_dropout: float = 0.1,
        norm: str = "batch",
        pos_encoding: Optional[str] = "learnable",
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len

        att_params_default = dict(
            input_dim=input_dim,
            dim_feedforward=int(2 * input_dim),
            nhead=nhead,
            ff_drop=dropout,
            att_drop=dropout,
            norm=norm,
            connection_drop=dropout,
        )
        if dim_feedforward is not None:
            att_params_default.update(dict(dim_feedforward=dim_feedforward))
        self.transformers = ArgSequential(
            *[TransformerDecoderLayer(**att_params_default) for _ in range(num_layers)]
        )

        # positional encoding
        self.forward_emb = None
        if pos_encoding is not None:
            self.forward_emb = _get_pos_encoder(pos_encoding)(
                input_dim, seq_len=seq_len, dropout=pos_dropout
            )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        src_attn_mask: Optional[torch.Tensor] = None,
        tgt_attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pass tensor through module.

        Parameters
        ----------
        x : torch.Tensor
        memory : torch.Tensor
        key_padding_mask : torch.Tensor
        src_attn_mask : torch.Tensor
        tgt_attn_mask : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        if self.forward_emb is not None:
            z = self.forward_emb(x)
        else:
            z = x

        z = self.transformers(
            z,
            memory=memory,
            src_attn_mask=src_attn_mask,
            tgt_attn_mask=tgt_attn_mask,
            key_padding_mask=key_padding_mask,
        )

        return z
