from dtaidistance import dtw_ndim, dtw, clustering
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad

import os
from typing import Optional, Union, Tuple, List

from morphelia.tools.utils import choose_representation


def linkage_tree(
    adata: ad.AnnData,
    time_var: str = "Metadata_Time",
    group_vars: Union[str, List[str]] = "Metadata_Treatment",
    dtw_method: str = "dependent",
    use_rep: Optional[str] = None,
    n_pcs: int = 50,
    show: bool = False,
    save: Optional[str] = None,
    plot_kwargs: dict = {},
    **kwargs,
) -> Union[clustering.LinkageTree, Tuple[clustering.LinkageTree, plt.Figure, plt.Axes]]:
    """Wrapper for scipy.cluster.hierarchy.linkage with a DTW distance matrix.

    This function calculates the Dynamic Time Warping distance between groups.
    Based on the distance, the groups are hierarchical clustered and connected through a linkage tree.

    Parameters
    ----------
    adata :anndata.AnnData): Multidimensional morphological data.
    time_var : str
        Variable in .obs that stores time points
    group_vars : str or list of str
        Variables in .obs that define conditions the distance
        matrix is to be calculated for.
    dtw_method : str
        `dependent` or `independent` DTW distance between two series
    use_rep : str, optional
        Calculate similarity/distance representation of X in `.obsm`
    n_pcs : int
        Number principal components to use if use_pcs is `X_pca`
    show : bool
        Show plot
    save : str, optional
        Save plot to a specified location
    plot_kwargs : dict
        Keyword arguments passed to plot
    **kwargs : dict
        Keyword arguments passed to dtaidistance.clustering.LinkageTree

    Returns
    -------
    clustering.LinkageTree or tuple
        Linkage tree with clustered groups

    Raises
    -------
    AssertionError
        If `time_var` or `group_vars` are not in `.obs`
    AssertionError
        If `dtw_method` is unknown
    AssertionError
        If time series are of different lengths
    OSError
        If figure can not be saved at specified location
    """
    # check variables
    assert time_var in adata.obs, f"time_var not in .obs: {time_var}"

    if isinstance(group_vars, str):
        group_vars = [group_vars]
    elif isinstance(group_vars, tuple):
        group_vars = list(group_vars)

    assert all(
        gv in adata.obs for gv in group_vars
    ), f"group_vars not in .obs: {group_vars}"

    avail_dtw_methods = ["dependent", "independent"]
    dtw_method = dtw_method.lower()
    assert dtw_method in avail_dtw_methods, (
        f"method not one of {avail_dtw_methods}, " f"instead got {dtw_method}"
    )

    # get series of groups
    s = []
    groups = []

    for group, group_df in adata.obs.groupby(group_vars):
        # sort group by time
        group_df = group_df.sort_values(time_var)
        # get group indices
        group_ixs = group_df.index

        adata_sub = adata[group_ixs, :].copy()
        if use_rep is not None:
            X = choose_representation(adata_sub, rep=use_rep, n_pcs=n_pcs)
        else:
            X = adata_sub.X

        # convert and append to series
        s.append(X.astype(np.double))
        groups.append(group)

    # y-dims should be all equal
    y_dims = [X.shape[0] for X in s]
    assert len(set(y_dims)) == 1, (
        "Data seems to be incorrectly aggregated. "
        "Assert that time points for groups are equal."
        f"Group lengths: {y_dims}"
    )

    # compute distance matrix
    ndim = X.shape[1]
    if dtw_method == "dependent":
        dists_fun = dtw_ndim.distance_matrix_fast
    elif dtw_method == "independent":
        dists_fun = _distance_matrix_fast_indipendent

    model = clustering.LinkageTree(
        dists_fun=dists_fun, dists_options={"ndim": ndim}, **kwargs
    )
    model.fit(s)

    if show:
        plot_kwargs.setdefault("show_tr_label", True)
        # plot_kwargs.setdefault('ts_label_margin', -10)
        # plot_kwargs.setdefault('ts_left_margin', 10)
        # plot_kwargs.setdefault('ts_sample_length', 1)
        # plot_kwargs.setdefault('ts_height', 30)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
        groups = [
            "-".join([str(elem) for elem in group]) if type(group) == tuple else group
            for group in groups
        ]

        def show_ts_label(idx):
            return str(groups[idx])

        plot_kwargs.setdefault("show_ts_label", show_ts_label)
        model.plot(axes=ax, **plot_kwargs)

        if save:
            try:
                plt.savefig(os.path.join(save, "feature_correlation.png"))
            except OSError:
                print(f"Can not save figure to {save}.")

        return model, fig, ax
    return model


def _distance_matrix_fast_indipendent(s: List[np.ndarray], ndim: int) -> np.ndarray:
    """Indipendent DTW distance matrix.

    Parameters
    ----------
    s : list of numpy.ndarray
        List of arrays with shape `[time-points, features]` representing different conditions
    ndim : int
        Feature dimensions

    Returns
    -------
    numpy.ndarray
        Distance matrix
    """
    dist = dtw.distance_matrix_fast(_series_sep_dim(s, 0))
    for dim in range(1, ndim):
        dist += dtw.distance_matrix_fast(_series_sep_dim(s, dim))

    return dist


def _series_sep_dim(s: List[np.ndarray], dim: int) -> list:
    """Return only one dimension from a series of arrays.

    Parameters
    ----------
    s : list of numpy.ndarray
        Series of multivariate arrays
    dim : int
        Feature dimensions

    Returns
    -------
    numpy.ndarray
        List of arrays
    """
    return [arr[:, dim] for arr in s]
