from typing import Union, Callable
from collections import defaultdict

import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm


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
    "mean": v_mean,
    "std": v_std,
    "max": v_max,
    "min": v_min,
    "diff": diff,
}


def ts_aggregate(
    adata: ad.AnnData,
    method: Union[str, Callable, str] = "mean",
    track_id: str = "Metadata_Track",
    time_var: str = "Metadata_Time",
    aggregate_reps: bool = False,
    fpu: int = 1,
    min_len: int = 30,
) -> ad.AnnData:
    """
    Collates all trajectories of a time series experiments by given methods to reduce the whole
    trajectory to one dimension.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    method : str, callable
        List of callable functions to apply on tracks to aggregate them.
        Can be one of mean, std, max, min, diff
        Default is to take the mean change.
    track_id : str
        Name of track identifiers in '.obs'
    time_var : str
        Name of time variable in '.obs'
    aggregate_reps : bool
        Aggregate representations in `.obsm` similar to `adata.X`
    fpu : int
        Frames per unit
    min_len : int
        Minimum length of track to consider. Only used if `add_mot` is False.

    Returns
    -------
    anndata.AnnData
        Aggregated data
    """
    if isinstance(method, str):
        assert (
            method in TrajectoryReductionMethods.keys()
        ), f"Method {method} not implemented"
        method = TrajectoryReductionMethods[method]

    # store aggregated data
    X_agg = []
    obs_agg = defaultdict(list)
    X_reps = defaultdict(list)

    for track, sdata in tqdm(adata.obs.groupby(track_id)):
        ts_len = len(sdata)

        if ts_len >= min_len:  # only continue with trajectories of minimum length
            sdata = sdata.sort_values(time_var)
            index = sdata.index

            # cache annotations from first element
            for key, val in sdata.iloc[0, :].to_dict().items():
                obs_agg[key].append(val)
            # Add length to annotations
            obs_agg["Length"].append(ts_len)

            agg = method(adata[index, :].X.copy(), fpu=fpu)
            X_agg.append(agg.reshape(1, -1))
            if aggregate_reps:
                for rep in adata.obsm.keys():
                    agg_rep = method(adata[index, :].obsm[rep].copy(), fpu=fpu)
                    X_reps[rep].append(agg_rep.reshape(1, -1))

    # make anndata object from aggregated data
    X_agg = np.concatenate(X_agg, axis=0)
    obs_agg = pd.DataFrame(obs_agg)

    # concatenate chunks of representations
    if len(list(X_reps.keys())) > 0:
        for _key, _val in X_reps.items():
            if len(_val) > 1:
                _val = np.vstack(_val)
            else:
                _val = _val[0]
            X_reps[_key] = _val

        adata = ad.AnnData(X=X_agg, obs=obs_agg, var=adata.var, obsm=X_reps)
    else:
        adata = ad.AnnData(X=X_agg, obs=obs_agg, var=adata.var)

    return adata
