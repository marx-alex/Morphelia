from typing import Optional, Tuple
import math

import torch
from torch import nn

from transformation import PermuteAxis, Reshape
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
        max_len: Maximum length of the incoming sequence.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        max_len: int = 1024,
        scale_factor: float = 1.0,
    ) -> None:
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # positional encoding
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        # store encoding in state dict
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Tensor of shape [S x N x F], where S is the sequence length,
                N is the batch size and F is the number of features.
        Returns:
            (torch.Tensor): Tensor of same shape as input tensor.
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


# From https://github.com/gzerveas/mvts_transformer/blob/master/src/models/ts_transformer.py
class LearnablePositionalEncoding(nn.Module):
    """
    Implements the learnable positional encoding Module.
    """

    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 1024) -> None:
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # positional encoding
        self.pe = nn.Parameter(torch.empty(max_len, 1, dim))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Tensor of shape [S x N x F], where S is the sequence length,
                N is the batch size and F is the number of features.
        Returns:
            (torch.Tensor): Tensor of same shape as input tensor.
        """

        x = x + self.pe[: x.size(0), :]
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
            nn.Linear(dim, 3 * dim),  # (B,S,3*F)
            Reshape(3, dim, fixed=2),  # (B,S,3,F),
            PermuteAxis(2, 0, 1, 3),  # (3,B,S,F)
        )  # (B,S,F) -> (3,B,S,F)

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )  # (3,B,S,F) -> (B,S,F)

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
        dim_feedforward: The dimension of the feedforward network model.
        dropout: The dropout value for the feedforward and attention layer.
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
        nhead: int = 10,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        norm: str = "batch",
        pos_encoding: Optional[str] = "learnable",
        num_layers: int = 1,
    ) -> None:
        super(TransformerEncoder, self).__init__()

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
            self.forward_emb = nn.Sequential(
                get_pos_encoder(pos_encoding)(
                    input_dim, dropout=dropout, max_len=seq_len
                ),
                nn.Dropout(dropout),
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
        if self.positional_encoder is not None:
            z = self.forward_emb(x)
        else:
            z = x

        z = self.transformers(z, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        return z


class TransformerClassifier(nn.Module):
    """Implements a classification layer on top of a Transformer encoder.

    Args:
        transformer: Transformer encoder module.
        n_classes: The number of expected classes.
    """

    def __init__(self, transformer: nn.Module, n_classes: int):
        super(TransformerClassifier, self).__init__()
        self.transformer = transformer
        input_dim = self.transformer.input_dim
        seq_len = self.transfromer.seq_len
        self.classifier = nn.Linear(input_dim * seq_len, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Passes the input through the encoder and classification modules.

        Args:
            x: the sequence to the encoder modules (required).
            attn_mask: the mask for the src sequence (optional).
            key_padding_mask: the mask for the src keys per batch (optional).
        """
        z = self.transformers(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # classification part
        y = z * ~key_padding_mask.unsqueeze(-1)
        y = y.reshape(y.shape[0], -1)  # [N, S x F]
        y = self.classifier(y)  # [N, n_classes]
        return y


class TransformerTokenClassifier(nn.Module):
    """
    This class transforms a sequence
    B x N x F -> B x F
    """

    def __init__(
        self,
        input_dim,
        seq_len,
        n_classes,
        att_params=None,
        depth=1,
        pos_drop=0.0,
        pos_emb=True,
    ):
        super(TransformerTokenClassifier, self).__init__()

        self.num_classes = n_classes

        att_params_default = dict(
            input_dim=input_dim, dim_feedforward=int(2 * input_dim), nhead=10
        )
        if att_params is not None:
            att_params_default.update(att_params)
        self.transformers = nn.Sequential(
            *[TransformerBlock(**att_params_default) for _ in range(depth)]
        )  # (B,N+1,F) -> (B,N+1,F) ... N + 1 because of the class token

        if pos_emb:
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, input_dim))
            self.pos_drop = nn.Dropout(pos_drop)

        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, n_classes)
        )  # (B,F) -> (B,n_classes)

    def forward_emb(self, x: torch.Tensor):
        """
        :param x:
        :return:
        """
        b, n, f = x.shape
        x = torch.cat(
            (self.cls_token.repeat(b, 1, 1), x), dim=1
        )  # (B,N,F) -> (B,N+1,F)

        if self.do_pos_emb:
            x += self.pos_embedding
            x = self.pos_drop(x)

        return x

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): B x N x F
        """
        x = self.forward_emb(x)  # (B,N,F) -> (B,N+1,F)
        x = self.transformers(x)  # (B,N+1,F) -> (B,N+1,F)
        x = x[:, 0]  # (B,N+1,F) -> (B,F)

        return self.classifier(x)
