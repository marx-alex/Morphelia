from typing import Optional, Tuple
import math

import torch
from torch import nn

from . import PermuteAxis, Reshape
from ._utils import ArgSequential


def get_pos_encoder(pos_encoding: str):
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
    """
    Implements positional encodings from sine and cosine functions of different frequencies.

    Args:
        dim: Number of features.
        dropout: Dropout value.
        seq_len: Maximum length of the incoming sequence.
        scale_factor: Scaling factor.
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
        """
        Add positional encoding to input tensor.

        Args:
            x: Tensor of shape [N x S x F], where S is the sequence length,
                N is the batch size and F is the number of features.
        Returns:
            (torch.Tensor): Tensor of same shape as input tensor.
        """
        x = x + self.pe
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Implements the learnable positional encoding Module.

    Args:
        dim: Number of features.
        dropout: Dropout value.
        seq_len: Maximum length of the incoming sequence.
    """

    def __init__(self, dim: int, seq_len: int = 1024, dropout: float = 0.1) -> None:
        super(LearnablePositionalEncoding, self).__init__()

        # positional encoding
        self.pe = self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Tensor of shape [N x S x F], where S is the sequence length,
                N is the batch size and F is the number of features.
        Returns:
            (torch.Tensor): Tensor of same shape as input tensor.
        """
        x = x + self.pe
        return self.dropout(x)


class ResNormBlock(nn.Module):
    def __init__(self, dim: int, fn, norm: str = "layer") -> None:
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
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        z = self.norm(x)
        z = self.fn(z, **kwargs)
        return z + x


class Attention(nn.Module):
    """
    This class implements an attention modules according to
        https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, dim, heads, dropout=0.0):
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
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, k, v = self.linear(x)  # create query, key and value
        x, _ = self.multi_head_attention(
            q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )  # new sequence
        return x

    def forward_with_attention(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        q, k, v = self.linear(x)  # create query, key and value
        x, w = self.multi_head_attention(
            q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )  # new sequence
        return x, w


class FeedForward(nn.Module):
    """
    This class implements the MLP, which is applied after
    the Multi-Head Attention.
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
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dim_feedforward: int,
        nhead: int,
        ff_drop: float = 0.0,
        att_drop: float = 0.0,
        norm: str = "layer",
    ) -> None:
        """
        MLP = multi modules perceptron

        Args:
            input_dim: feature dimension
            dim_feedforward: dim for the mlp
            nhead: attention heads (dim % heads == 0)
            ff_drop: feat forward drop
            att_drop: attention drop
            norm: Normalization method.
        """
        super(TransformerBlock, self).__init__()

        self.attn = ResNormBlock(
            input_dim, Attention(input_dim, nhead, att_drop), norm=norm
        )
        self.ff = ResNormBlock(
            input_dim, FeedForward(input_dim, dim_feedforward, ff_drop), norm=norm
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self.attn(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return self.ff(z)


class TransformerEncoder(nn.Module):
    """
    Implements the Transformer encoder.

    Args:
        input_dim: The number of expected features in the input.
        seq_len: Sequence length.
        nhead: The number of heads in the multihead attention layer.
            Embedded dimensions must be divisible by number of heads.
        dim_feedforward: The dimension of the feedforward network model.
            The default dimension is 2 times the input dimensions.
        dropout: The dropout value for the feedforward and attention layer.
        pos_dropout: The dropout value for the positional encoding.
        norm: Normalization method ('batch' or 'layer').
        pos_encoding: Positional encoding. Can be 'learnable' or 'fixed'
        num_layers: Depth of encoder.

    Reference:
        George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning,
        in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21),
        August 14--18, 2021.
        ArXiV version: https://arxiv.org/abs/2010.02803
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
        )
        if dim_feedforward is not None:
            att_params_default.update(dict(dim_feedforward=dim_feedforward))
        self.transformers = ArgSequential(
            *[TransformerBlock(**att_params_default) for _ in range(num_layers)]
        )

        # positional encoding
        self.forward_emb = None
        if pos_encoding is not None:
            self.forward_emb = get_pos_encoder(pos_encoding)(
                input_dim, seq_len=seq_len, dropout=pos_dropout
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes the input through the encoder modules.

        Args:
            x: the sequence to the encoder modules (required).
            attn_mask: the mask for the src sequence (optional).
            key_padding_mask: the mask for the src keys per batch (optional).
        """
        if self.forward_emb is not None:
            z = self.forward_emb(x)
        else:
            z = x

        z = self.transformers(z, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        return z
