from typing import List

import torch
from torch import nn


def get_activation_fn(activation: str):
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU
    elif activation == "gelu":
        return nn.GELU
    raise NotImplementedError(
        f"Activation can be either 'relu' or 'gelu', instead got {activation}"
    )


def partition(x: torch.Tensor, y: torch.Tensor, n_partitions) -> List[torch.Tensor]:
    partitions = []
    y = y.flatten()

    for i in range(n_partitions):
        mask = y == i
        partitions.append(x[mask, ...])

    return partitions


def geometric_noise(
    x: torch.Tensor, masking_len: int = 3, masking_ratio: float = 0.15
) -> torch.Tensor:
    """
    Creating geometric noise on a sequence with n features in a Markovian manner.
    The states of the Markov chain are masked or und masked with two transition probabilities
    defined by an average masking length and a masking ratio.
    This returns a binary attention mask to introduce random noise into the sequences.
    0s refer to masked values.

    Args:
        x: Tensor of shape batch size x sequence length x feature dimensions.
        masking_len: Average length of masks.
        masking_ratio: Average masking ratio for a feature.

    Returns:
        (torch.Tensor): Tensor of same shape as X.
            0s mean masked values.

    Reference:
        George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning,
        in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21),
        August 14-18, 2021. ArXiV version: https://arxiv.org/abs/2010.02803
    """
    mask = torch.ones_like(x)
    k, m, n = x.shape  # [B, N, S]

    # probabilities for masking and unmasking
    p_mask = 1 / masking_len
    p_unmask = p_mask * masking_ratio / (1 - masking_ratio)
    p = torch.Tensor([p_mask, p_unmask])

    for sample_ix in range(k):
        # start state
        state = (torch.rand(n) > masking_ratio).type(torch.long)
        for sequence_ix in range(m):
            mask[sample_ix, sequence_ix, :] = state
            state_prob = torch.rand(n)
            change_prob = torch.take(p, state)
            state = torch.where(state_prob < change_prob, 1 - state, state)

    return mask
