import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
from dtaidistance import dtw_ndim, dtw

from morphelia.time_series import HMMSimilarity
from morphelia.tools.utils import choose_representation


def dist_matrix(
    adata,
    method="pearson",
    group_var="Metadata_Treatment",
    other_group_vars=None,
    use_rep=None,
    n_pcs=50,
    make_plot=False,
    show=False,
    save=None,
    return_array=False,
):
    """Computes a similarity or distance matrix between different treatments and doses if given.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        method (str): Method for similarity/ distance computation.
            Should be one of: pearson, spearman, kendall, euclidean, mahalanobis.
        group_var (str): Find similarity between groups. Could be treatment conditions for example.
        other_group_vars (list, str): Other variables that define groups that are similar.
        use_rep (str): Calculate similarity/distance representation of X in .obsm.
        n_pcs (int): Number principal components to use if use_pcs is 'X_pca'.
        make_plot (bool): Plot similarity as heatmap.
        show (bool): Show plot and return figure and axis.
        save (str): Save plot to a specified location.
        return_array (bool): Return array instead of dataframe.

    Returns:
        pandas.DataFrame: similarity/ distance matrix
        numpy.ndarray: Only if return_array is True
    """
    # check variables
    assert group_var in adata.obs.columns, f"treat_var not in observations: {group_var}"
    if other_group_vars is not None:
        if isinstance(other_group_vars, str):
            other_group_vars = [other_group_vars]
        assert isinstance(other_group_vars, list), (
            "Expected type for other_group_vars is string or list, "
            f"instead got {type(other_group_vars)}"
        )
        assert all(var in adata.obs.columns for var in other_group_vars), (
            f"other_group_vars not in " f"observations: {other_group_vars}"
        )

    # check method
    avail_methods = [
        "pearson",
        "spearman",
        "kendall",
        "euclidean",
        "mahalanobis",
    ]
    method = method.lower()
    assert method in avail_methods, (
        f"method should be in {avail_methods}, " f"instead got {method}"
    )

    # get representation of data
    if use_rep is None:
        use_rep = "X"
    X = choose_representation(adata, rep=use_rep, n_pcs=n_pcs)

    # load profiles to dataframe and transpose
    if other_group_vars is not None:
        group_vars = [group_var] + other_group_vars
        dist_df = pd.DataFrame(X, index=[adata.obs[var] for var in group_vars]).T
    else:
        dist_df = pd.DataFrame(X, index=adata.obs[group_var]).T

    # calculate similarity
    if method == "pearson":
        dist_df = dist_df.corr(method="pearson")

    if method == "spearman":
        dist_df = dist_df.corr(method="spearman")

    if method == "kendall":
        dist_df = dist_df.corr(method="kendall")

    if method == "euclidean":
        sim = squareform(pdist(dist_df.T, metric="euclidean"))
        dist_df = pd.DataFrame(sim, columns=dist_df.columns, index=dist_df.columns)

    if method == "mahalanobis":
        sim = squareform(pdist(dist_df.T, metric="manhattan"))
        dist_df = pd.DataFrame(sim, columns=dist_df.columns, index=dist_df.columns)

    if return_array:
        return dist_df.to_numpy()

    # sort index
    dist_df.sort_index(axis=0, inplace=True)
    dist_df.sort_index(axis=1, inplace=True)

    if make_plot:
        sns.set_theme()
        cmap = matplotlib.cm.plasma

        fig = plt.figure(figsize=(7, 5))
        ax = sns.heatmap(dist_df, cmap=cmap)
        plt.suptitle(f"distance/ similarity: {method}", fontsize=16)

        if save:
            try:
                plt.savefig(os.path.join(save, "feature_correlation.png"))
            except OSError:
                print(f"Can not save figure to {save}.")

        if show:
            plt.show()
            return dist_df, fig, ax

    return dist_df


def dtw_dist_matrix(
    adata,
    time_var="Metadata_Time",
    group_vars="Metadata_Treatment",
    method="dependent",
    use_rep=None,
    n_pcs=50,
    make_plot=False,
    show=False,
    save=None,
    return_array=False,
    **kwargs,
):
    """
    Compute distance matrix with time-series data using multivariate dynamic time warping.

    This functions wraps the DTW implementation by Shokoohi-Yekta et al.:
    M. Shokoohi-Yekta, B. Hu, H. Jin, J. Wang, and E. Keogh.
    Generalizing dtw to the multi-dimensional case requires an adaptive approach.
    Data Mining and Knowledge Discovery, 31:1–31, 2016.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        time_var (str): Variable in .obs that stores time points.
        group_vars (str or list of str): Variables in .obs that define conditions the distance
            matrix is to be calculated for.
        method (str): 'dependent' or 'independent' DTW distance between two series.
        use_rep (str): Calculate similarity/distance representation of X in .obsm.
        n_pcs (int): Number principal components to use if use_pcs is 'X_pca'.
        make_plot (bool): Plot distance as heatmap.
        show (bool): Show plot and return figure and axis.
        save (str): Save plot to a specified location.
        return_array (bool): Return array instead of dataframe.
        **kwargs (dict): Keyword arguments passed to dtaidistance.dtw_ndim.distance_matrix_fast
            or dtaidistance.dtw.distance_matrix_fast based on method.

    Returns:
        pandas.DataFrame: similarity/ distance matrix
        numpy.ndarray: only if return_array is True
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

    avail_methods = ["dependent", "independent"]
    method = method.lower()
    assert method in avail_methods, (
        f"method not one of {avail_methods}, " f"instead got {method}"
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

    # compute distance matrix
    if method == "dependent":
        ndim = X.shape[1]
        dist = dtw_ndim.distance_matrix_fast(s, ndim=ndim, **kwargs)
    elif method == "independent":
        dist = dtw.distance_matrix_fast(_series_sep_dim(s, 0), **kwargs)
        for dim in range(1, X.shape[1]):
            dist += dtw.distance_matrix_fast(_series_sep_dim(s, dim), **kwargs)

    if return_array:
        return dist

    # annotation to dataframe
    groups = np.array(groups)
    if len(groups.shape) == 1:
        groups = groups[:, None]
    annotations = [
        pd.Series(groups[:, ix], name=group_var)
        for ix, group_var in enumerate(group_vars)
    ]

    # distance matrix to dataframe
    dist_df = pd.DataFrame(dist, index=annotations, columns=annotations)

    if make_plot:
        sns.set_theme()
        cmap = matplotlib.cm.plasma

        fig = plt.figure(figsize=(7, 5))
        ax = sns.heatmap(dist_df, cmap=cmap)
        plt.suptitle("DTW distances", fontsize=16)

        if save:
            try:
                plt.savefig(os.path.join(save, "feature_correlation.png"))
            except OSError:
                print(f"Can not save figure to {save}.")

        if show:
            plt.show()
            return dist_df, fig, ax

    return dist_df


def _series_sep_dim(s, dim):
    """Return only one dimension from a series of arrays.

    Args:
        s (list of np.ndarray): Series of multivariate arrays.
        dim (int): Dimension to return.

    Returns:
        numpy.ndarray
    """
    return [arr[:, dim] for arr in s]


def hmm_sim_matrix(
    adata,
    time_var="Metadata_Time",
    group_vars="Metadata_Treatment",
    comp_range=(1, 10),
    use_rep=None,
    n_pcs=50,
    make_plot=False,
    show=False,
    save=None,
    return_array=False,
):
    """
    Compute distance matrix with time-series data using hidden Markov models.

    This functions wraps the HMM distance metric by Sahraeian et al., 2011:
    S. M. E. Sahraeian and B. Yoon, "A Novel Low-Complexity HMM Similarity Measure,"
    in IEEE Signal Processing Letters, vol. 18, no. 2, pp. 87-90, Feb. 2011, doi: 10.1109/LSP.2010.2096417

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        time_var (str): Variable in .obs that stores time points.
        group_vars (str or list of str): Variables in .obs that define conditions the distance
            matrix is to be calculated for.
        comp_range (tuple): Range with number of components to fit HMMs.
        use_rep (str): Calculate similarity/distance representation of X in .obsm.
        n_pcs (int): Number principal components to use if use_pcs is 'X_pca'.
        make_plot (bool): Plot similarity as heatmap.
        show (bool): Show plot and return figure and axis.
        save (str): Save plot to a specified location.
        return_array (bool): Return array instead of dataframe.

    Returns:
        pandas.DataFrame: similarity/ distance matrix
        numpy.ndarray: only if return_array is True
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
        s.append(X)
        groups.append(group)

    # compute distance matrix
    hmm = HMMSimilarity(state_range=comp_range)
    hmm.fit(s)
    sim = hmm.similarity()

    if return_array:
        return sim

    # annotation to dataframe
    groups = np.array(groups)
    annotations = [
        pd.Series(groups[:, ix], name=group_var)
        for ix, group_var in enumerate(group_vars)
    ]

    # distance matrix to dataframe
    sim_df = pd.DataFrame(sim, index=annotations, columns=annotations)

    if make_plot:
        sns.set_theme()
        cmap = matplotlib.cm.plasma

        fig = plt.figure(figsize=(7, 5))
        ax = sns.heatmap(sim_df, cmap=cmap)
        plt.suptitle("HMM similarity", fontsize=16)

        if save:
            try:
                plt.savefig(os.path.join(save, "feature_correlation.png"))
            except OSError:
                print(f"Can not save figure to {save}.")

        if show:
            plt.show()
            return sim_df, fig, ax

    return sim_df
