from collections import abc
from typing import Optional

import torch
import numpy as np
import pandas as pd


def collate_concat(batch: list) -> dict:
    """Concatenate batch elements.

    This function collates a list of of samples.
    Samples might be tensors of dictionaries.
    Values of dictionaries can be `list`, `numpy.ndarray`, `torch.tensor` or `pandas.Series`.

    Parameters
    ----------
    batch : list
        List of dictionaries

    Returns
    -------
    dict
        Dictionary with single tensors as values

    Examples
    --------
    >>> import morphelia as mp

    >>> batch = [
    >>>     {'x': [0, 1, 2, 3],
    >>>      'y': [0, 0, 1, 1]},
    >>>     {'x': [4, 5, 6, 7],
    >>>      'y': [0, 0, 1, 1]}
    >>> ]
    >>> mp.dl.data.collate_concat(batch)
    {'x': tensor([[0, 1, 2, 3],
             [4, 5, 6, 7]]),
     'y': tensor([[0, 0, 1, 1],
             [0, 0, 1, 1]])}
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


class AdataCollator:
    """Collator class for the AnnLoader module.

    This class contains the pytorch collate logic, which
    can be initiated once and called for every batch.

    Parameters
    ----------
    target_key : str, optional
        Target variable
    condition_key : str, optional
        Condition variable
    time_key : str, optional
        Time variable (for time-series experiments)
    """

    def __init__(
        self,
        target_key: Optional[str] = "Metadata_Treatment",
        condition_key: Optional[str] = None,
        time_key: Optional[str] = None,
    ) -> None:
        self.target_key = target_key
        self.condition_key = condition_key
        self.time_key = time_key

    def __call__(self, batch) -> dict:
        """Called for every batch.

        If AdataCollator is called, `x`, `target`, `c` (condition) and `t` (time)
        is returned as a dictionary.

        Parameters
        ----------
        batch : AnnCollectionView
            Batch given by anndata.experimental.AnnLoader

        Returns
        -------
        dict
            Converted batch to dictionary
        """
        out = dict()

        out["x"] = batch.X
        if self.target_key is not None:
            out["target"] = batch.obs[self.target_key].long()
        out["ids"] = torch.Tensor(batch.oidx).int()
        if self.condition_key is not None:
            out["c"] = batch.obs[self.condition_key].long()
        if self.time_key is not None:
            out["t"] = batch.obs[self.time_key].to(torch.float)
        return out
