import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl


class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data, return_index=True):
        self.return_index = return_index
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        if self.return_index:
            return self.X_data[index], self.y_data[index], index
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 X_train: np.ndarray = None, y_train: np.ndarray = None,
                 X_test: np.ndarray = None, y_test: np.ndarray = None,
                 X_val: np.ndarray = None, y_val: np.ndarray = None,
                 weighted=False,
                 batch_size: int = 32,
                 num_workers=0):
        """
        Data Module.

        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :param X_val:
        :param y_val:
        :param batch_size:
        :param num_workers:
        """
        super().__init__()

        self.train_dataset = None
        if X_train is not None:
            self.train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(),
                                                   torch.from_numpy(y_train).long())

        self.test_dataset = None
        if X_test is not None:
            self.test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(),
                                                  torch.from_numpy(y_test).long())

        self.val_dataset = None
        if X_val is not None:
            self.val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(),
                                                 torch.from_numpy(y_val).long())
        self.weighted = weighted
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_shape = X_train.shape[-1]
        self.n_classes = len(np.unique(y_train))

    @staticmethod
    def weighted_sampler(y):
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        values, counts = np.unique(y, return_counts=True)
        weight = 1 / counts
        samples_weight = np.array([weight[v] for v in y])

        samples_weight = torch.from_numpy(samples_weight).double()
        return WeightedRandomSampler(samples_weight, len(samples_weight))

    def train_dataloader(self):
        sampler = None
        shuffle = True
        if self.weighted:
            sampler = self.weighted_sampler(self.train_dataset.y_data)
            shuffle = False
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=shuffle,
                          sampler=sampler)

    def val_dataloader(self):
        sampler = None
        if self.weighted:
            sampler = self.weighted_sampler(self.val_dataset.y_data)
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=sampler,
                          shuffle=False)

    def test_dataloader(self):
        sampler = None
        if self.weighted:
            sampler = self.weighted_sampler(self.test_dataset.y_data)
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=sampler,
                          shuffle=False)

    def predict_dataloader(self):
        sampler = None
        if self.weighted:
            sampler = self.weighted_sampler(self.test_dataset.y_data)
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=sampler,
                          shuffle=False)
    
    
def collate_superv(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples.
    
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels, ixs = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks, ixs


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class ImputationDataset(Dataset):
    """Dynamically computes missingness (noise) mask for each sample from an anndata object.
    As described by:

    Reference:

        George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning,
        in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21),
        August 14--18, 2021.
        ArXiV version: https://arxiv.org/abs/2010.02803

        https://github.com/gzerveas/mvts_transformer/blob/master/src/datasets/dataset.py
    """
    def __init__(self, adata, indices,
                 mean_mask_length=3, masking_ratio=0.15,
                 mode='separate', distribution='geometric', exclude_feats=None):
        self.data = adata
        self.idxs = indices
        self.features = self.data[self.idxs, :].X.copy()

        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length
        self.mode = mode
        self.distribution = distribution
        self.exclude_feats = exclude_feats

    def __getitem__(self, index):
        X = self.features[index]
        mask = noise_mask(X, self.masking_ratio, self.mean_mask_length, self.mode, self.distribution,
                          self.exclude_feats)  # (seq_length, feat_dim) boolean array

        return torch.from_numpy(X), torch.from_numpy(mask), index

    def update(self):
        self.mean_mask_length = min(20, self.mean_mask_length + 1)
        self.masking_ratio = min(1, self.masking_ratio + 0.05)

    def __len__(self):
        return len(self.idxs)


# from https://github.com/gzerveas/mvts_transformer/blob/master/src/datasets/dataset.py
def noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)
    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask


# from https://github.com/gzerveas/mvts_transformer/blob/master/src/datasets/dataset.py
def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked
    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask
