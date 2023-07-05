from typing import Union, List, Callable, Optional, Sequence

import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm

import morphelia as mp


def _diff(a: np.ndarray, diff: int = 1) -> np.ndarray:
    assert diff >= 1, f"Difference must be greater than 0, instead got {diff}"

    out_shape = [*a.shape]
    out_shape[0] = out_shape[0] - diff
    out = np.zeros(out_shape).astype(a.dtype)

    for i in range(out.shape[0]):
        out[i, ...] = a[i + diff, ...] - a[i, ...]

    return out


def v_mean(x: np.ndarray, fpu: int = 1) -> np.ndarray:
    return np.mean(_diff(x, diff=fpu), axis=0)


def v_std(x: np.ndarray, fpu: int = 1) -> np.ndarray:
    return np.std(_diff(x, diff=fpu), axis=0)


def v_max(x: np.ndarray, fpu: int = 1) -> np.ndarray:
    return np.max(_diff(x, diff=fpu), axis=0)


def v_min(x: np.ndarray, fpu: int = 1) -> np.ndarray:
    return np.min(_diff(x, diff=fpu), axis=0)


def diff(x: np.ndarray, **kwargs) -> np.ndarray:
    return x[-1, :] - x[0, :]


TrajectoryReductionMethods = {
    'mean': v_mean,
    'std': v_std,
    'max': v_max,
    'min': v_min,
    'diff': diff
}


def ts_aggregate(
    adata: ad.AnnData,
    method: Union[str, List[Union[Callable, str]]] = "mean",
    track_id: str = "Metadata_Track",
    time_var: str = "Metadata_Time",
    use_rep: Optional[str] = None,
    fpu: int = 1,
    min_len: int = 30,
    var_names: Optional[Sequence] = None,
    store_vars: Optional[Union[str, list]] = None,
) -> pd.DataFrame:
    """
    Collates all trajectories of a time series experiments by given methods to reduce the whole
    trajectory to one dimension.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    method : list, str
        List of callable functions to apply on tracks to aggregate them.
        Can be one of mean, std, max, min, diff
        Default is to take the mean change.
    track_id : str
        Name of track identifiers in '.obs'
    time_var : str
        Name of time variable in '.obs'
    use_rep : str, optional
        Use a representation from '.obsm'
    fpu : int
        Frames per unit
    min_len : int
        Minimum length of track to consider. Only used if `add_mot` is False.
    var_names : sequence, list
        Names for variables two override names from the AnnData object
    store_vars : str, list
        Store additional variables for every track

    Returns
    -------
    pandas.DataFrame
        New motility parameters are stored in '.obs'
    """

    if isinstance(store_vars, str):
        store_vars = [store_vars]

    if isinstance(method, str) or callable(method):
        method = [method]

    # assign variable names
    if use_rep is None:
        n_vars = adata.n_vars
    else:
        n_vars = adata.obsm[use_rep].shape[1]

    if var_names is not None:
        assert (
            len(var_names) == n_vars
        ), f"Number of variables names ({len(var_names)}) must match number of features ({n_vars})"
    elif use_rep is None:
        var_names = adata.var_names
    else:
        var_names = np.array([f"{use_rep}_{i}" for i in range(n_vars)])

    # assign variable names for multiple methods
    if len(method) > 1:
        multiple_var_names = []
        for i, m in enumerate(method):
            if isinstance(m, str):
                c = m
            else:
                c = f"method_{i}"
            for v in var_names:
                multiple_var_names.append(f"{v}_{c}")
        var_names = multiple_var_names

    output = []

    for track, sdata in tqdm(adata.obs.groupby(track_id)):
        ts_len = len(sdata)

        if ts_len >= min_len:  # only continue with trajectories of minimum length
            sdata = sdata.sort_values(time_var)
            index = sdata.index

            results = pd.Series({'Track': track, 'Length': ts_len})
            if store_vars is not None:
                results = pd.concat((results, sdata.iloc[0, :][store_vars]))

            # get representation
            if use_rep is not None:
                X = adata[index, :].obsm[use_rep]
            else:
                X = adata[index, :].X

            # apply methods
            reductions = []
            for m in method:
                if isinstance(m, str):
                    assert m in TrajectoryReductionMethods.keys(), f"Method {m} not implemented"
                    m = TrajectoryReductionMethods[m]

                reduction = m(X.copy(), fpu=fpu)
                reduction = reduction.flatten()

                reductions.append(reduction)

            reductions = np.concatenate(reductions, axis=None)
            reductions = pd.Series(reductions, index=var_names)

            results = pd.concat((results, reductions))
            output.append(results)

    return pd.DataFrame(output)
