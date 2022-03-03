import torch
from torch import nn


class ArgSequential(nn.Sequential):
    """
    Pytorch Sequential Module. Optional keyword arguments are passed to every module.
    """

    def __init__(self, *args):
        super(ArgSequential, self).__init__(*args)

    def forward(self, x, **kwargs):
        for module in self:
            x = module(x, **kwargs)
        return x


def geometric_noise(x, masking_len=3, masking_ratio=0.15):
    """
    Creating geometric noise on a sequence with n features in a Markovian manner.
    The states of the Markov chain are masked or und masked with two transition probabilities
    defined by an average masking length and a masking ratio.
    This returns a binary attention mask to introduce random noise into the sequences.
    0s refer to masked values.

    Args:
        x (torch.Tensor): sequence length x feature dimensions
        masking_len (int): Average length of masks.
        masking_ratio (float): Average masking ratio for a feature.

    Returns:
        (torch.Tensor): Tensor of same shape as X.
            0s mean masked values.

    Reference:
        George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning,
        in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21),
        August 14-18, 2021. ArXiV version: https://arxiv.org/abs/2010.02803
    """
    mask = torch.ones_like(x)
    m, n = x.shape  # sequence x features

    # probabilities for masking and unmasking
    p_mask = 1 / masking_len
    p_unmask = p_mask * masking_ratio / (1 - masking_ratio)
    p = torch.Tensor([p_mask, p_unmask])

    # start state
    state = (torch.rand(n) > masking_ratio).type(torch.long)
    for sequence_ix in range(m):
        mask[sequence_ix, :] = state
        state_prob = torch.rand(n)
        change_prob = torch.take(p, state)
        state = torch.where(state_prob < change_prob, 1 - state, state)

    return mask
