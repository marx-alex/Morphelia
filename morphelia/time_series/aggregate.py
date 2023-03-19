from typing import Union, List, Callable, Optional

import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm

import morphelia as mp


def v_mean(X: np.ndarray, fpu: int = 1) -> np.ndarray:
    return np.mean(np.diff(X, axis=0) * fpu, axis=0)


def v_std(X: np.ndarray, fpu: int = 1) -> np.ndarray:
    return np.std(np.diff(X, axis=0) * fpu, axis=0)


def v_max(X: np.ndarray, fpu: int = 1) -> np.ndarray:
    return np.max(np.diff(X, axis=0) * fpu, axis=0)


def v_min(X: np.ndarray, fpu: int = 1) -> np.ndarray:
    return np.min(np.diff(X, axis=0) * fpu, axis=0)


method_dict = {"mean": v_mean, "std": v_std, "max": v_max, "min": v_min}


def ts_aggregate(
    adata: ad.AnnData,
    x_loc: str,
    y_loc: str,
    method: Union[str, List[Union[Callable, str]]] = "mean",
    add_mot: bool = True,
    obs_cols: Optional[Union[str, list]] = None,
    track_id: str = "Metadata_Track",
    time_var: str = "Metadata_Time",
    fpu: int = 1,
    msd_max_tau: Optional[int] = 30,
    kurtosis_max_tau: Optional[int] = 3,
    autocorr_max_tau: Optional[int] = 10,
    min_len: int = 30,
) -> ad.AnnData:
    """
    Collates all trajectories of a time series experiments by given methods to reduce the whole
    trajectory to one dimension.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    x_loc : str
        Location of cell on x-coordinate
    y_loc : str
        Location of cell on y-coordinate
    method : list, str
        List of callable functions to apply on tracks to aggregate them.
        Default is to take the mean change.
    add_mot : bool
        Add motility measurements to `.X`
    obs_cols : str, list
        Only keep specified columns for new AnnData object
    track_id : str
        Name of track identifiers in '.obs'
    time_var : str
        Name of time variable in '.obs'
    fpu : int
        Frames per unit
    msd_max_tau : int
        Maximal tau for Mean Squared Displacement
    kurtosis_max_tau : int
        Maximal tau for the calculation of the kurtosis of the displacement distribution
    autocorr_max_tau : int
        Maximal tau for Autocorrelation
    min_len : int
        Minimum length of track to consider. Only used if `add_mot` is False.

    Returns
    -------
    adata : anndata.AnnData
        New motility parameters are stored in '.obs'
    .uns['motility']
        Averaged values per track
    """
    if add_mot:
        min_track_len = max(msd_max_tau, autocorr_max_tau) + 1
    else:
        min_track_len = min_len

    obs_list = []
    Z_list = []
    mot_list = []

    if isinstance(obs_cols, str):
        obs_cols = [obs_cols]
    elif obs_cols is None:
        obs_cols = list(adata.obs.columns)

    if isinstance(method, str):
        method = [method]

    var_names = []
    for i, m in enumerate(method):
        if isinstance(m, str):
            c = m
        else:
            c = f"method_{i}"
        for v in adata.var_names:
            var_names.append(f"{v}_{c}")

    for track, sdata in tqdm(adata.obs.groupby(track_id)):
        ts_len = len(sdata)

        if ts_len >= min_track_len:  # only continue with trajectories of minimum length
            sdata = sdata.sort_values(time_var)
            sdata["Length"] = ts_len  # add trajectory length to annotations
            index = sdata.index
            obs_list.append(sdata.iloc[0, :][obs_cols])
            path = sdata[[x_loc, y_loc]].values

            # get motility of trajectory
            if add_mot:
                mot = mp.ts.CellMotility(
                    path=path,
                    fpu=fpu,
                    msd_max_tau=msd_max_tau,
                    kurtosis_max_tau=kurtosis_max_tau,
                    autocorr_max_tau=autocorr_max_tau,
                )
                mot_list.append(mot.result())

            # collate X
            Zs = []
            for m in method:
                if m in method_dict.keys():
                    m = method_dict[m]

                Z = m(adata[index, :].X.copy())
                Z = Z.flatten()

                assert (
                    len(Z.shape) == 1
                ), f"Method {m} does not collate trajectory to shape (1, x)"
                Zs.append(Z)

            Z_list.append(np.concatenate(Zs))

    # concatenate cached elements
    obs = pd.DataFrame(obs_list)
    X = np.vstack(Z_list)
    if add_mot:
        mot = pd.DataFrame(mot_list)
        X = np.hstack((X, mot.values))
        var_names = var_names + list(mot.columns)
    var = pd.DataFrame(index=var_names)

    return ad.AnnData(X=X, obs=obs, var=var)
