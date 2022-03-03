from collections import abc

import torch
import numpy as np
import pandas as pd


def collate_concat(batch):
    """
    Collate a list of of samples.
    Samples might be tensors of dictionaries.
    Values of dictionaries can be list, numpy.ndarray, torch.tensor or pandas.Series.

    Args:
        batch (list): List of dictionaries.

    Returns:
        (dict): Dictionary with single tensors as values.
    """
    elem = batch[0]

    # handle list of tensor
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # concatenate directly into a shared memory tensor
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)

    # handle numpy array
    elif isinstance(elem, np.ndarray):
        return collate_concat([torch.from_numpy(b) for b in batch])

    # handle lists
    elif isinstance(elem, list):
        return collate_concat([torch.as_tensor(b) for b in batch])

    # handle pandas series
    elif isinstance(elem, pd.Series):
        if elem.dtype.name == "category":
            return collate_concat([torch.as_tensor(b.cat.codes) for b in batch])
        elif np.issubdtype(elem.dtype, np.number):
            return collate_concat([torch.from_numpy(b.to_numpy()) for b in batch])

    # handle dictionaries recursively
    elif isinstance(elem, abc.Mapping):
        output = {key: collate_concat([d[key] for d in batch]) for key in elem}
        return output
