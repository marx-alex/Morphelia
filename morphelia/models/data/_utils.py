import torch
import numpy as np
from scipy.sparse import issparse


def data_converter(arr):
    """Convert array to torch.Tensor."""
    if isinstance(arr, torch.Tensor):
        return arr
    elif arr.dtype.name == "category":
        return torch.Tensor(arr.cat.codes)
    elif np.issubdtype(arr.dtype, np.number):
        if issparse(arr):
            arr = arr.toarray()
        return torch.Tensor(arr)
    else:
        raise ValueError(f"Can not convert target dtype: {type(arr)}")
