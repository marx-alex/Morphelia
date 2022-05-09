# external libraries
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import silhouette_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad
import pandas as pd
from tqdm import tqdm

# internal libraries
import os
import warnings
from typing import Tuple, Optional, Union, List

from morphelia.tools.utils import get_subsample


def feature_agglo(
    adata: ad.AnnData,
    k: Union[str, int] = "estimate",
    cluster_range: Union[List[int], Tuple[int]] = (2, 100),
    subsample: bool = False,
    sample_size: int = 1000,
    seed: int = 0,
    make_plot: bool = False,
    show: bool = False,
    save: Optional[str] = None,
    **kwargs,
) -> ad.AnnData:
    """Wrapper for scikit's FeatureAgglomeration.

    Calculates clusters by hierarchical clustering and replaces feature clusters
    by their aggregates. This function expects scaled (z-transformed) data.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    k : str or int
        Number of clusters to compute. If `estimate`, the
        ideal number of k is estimated.
    cluster_range : tuple of int or list of int
        Minimum and maximum numbers of clusters to compute
    subsample : bool
        If True, fit models on subsample of data
    sample_size : int
        Size of subsample. Only if subsample is True.
    seed : int
        Seed for subsample calculation. Only if subsample is True.
    make_plot : bool
        Make plot to see results
    show : bool
        Show plot
    save : str, optional
        Path where to save figure
    **kwargs
        Keyword arguments passed to sklearn.cluster.FeatureAgglomeration

    Returns
    -------
    anndata.AnnData
        `.uns['agglo_feats']`: Agglomerated features and their original features

    Raises
    -------
    AssertionError
        If `cluster_range` is not a tuple (or list) of lenght two
    AssertionError
        If `k` is not of type integer

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np

    >>> data = np.random.rand(6, 5)
    >>> adata = ad.AnnData(data)
    >>> mp.ft.feature_agglo(adata, k=2)
    AnnData object with n_obs × n_vars = 6 × 2
        uns: 'agglo_feats'
    """
    # check cluster_range
    if isinstance(cluster_range, tuple):
        cluster_range = list(cluster_range)
    assert isinstance(
        cluster_range, list
    ), f"type list or tuple expected for cluster_range instead got: {type(cluster_range)}"
    assert (
        len(cluster_range) == 2
    ), f"cluster_range should be a tuple or list of length 2, instead got: {cluster_range}"
    if cluster_range[1] > adata.shape[1]:
        warnings.warn(
            "Maximal expected k is larger than number of features", UserWarning
        )
        cluster_range[1] = adata.shape[1]
    if cluster_range[0] < 2:
        warnings.warn(f"Minimum k is 2, changes {cluster_range[0]} to 2", UserWarning)
        cluster_range[0] = 2

    # check for nan values
    if np.isnan(adata.X).any():
        warnings.warn("Array contains NaN.", UserWarning)

    # get subsample
    if subsample:
        adata_ss = get_subsample(adata, sample_size=sample_size, seed=seed)
    else:
        adata_ss = adata.copy()

    if k == "estimate":
        k = estimate_k(
            adata_ss,
            cluster_range=cluster_range,
            make_plot=make_plot,
            return_fig=False,
            show=show,
            save=save,
            **kwargs,
        )

    # calculate feature agglomeration
    assert isinstance(k, int), f"k should be an integer, instead got {type(k)}"
    model = FeatureAgglomeration(n_clusters=k, **kwargs)

    agglo = model.fit(adata_ss.X)

    X_agglo = model.transform(adata.X)

    # add unstructured data:
    # dict of labels from new features with old features as values
    feat_lst = list(zip(agglo.labels_, adata.var_names))
    agglo_feats = {}
    for pair in feat_lst:
        agglo_feats.setdefault(f"agglo_{pair[0]}", []).append(pair[1])

    # create new anndata object
    var = pd.DataFrame(index=[f"agglo_{i}" for i in set(agglo.labels_)])
    adata = ad.AnnData(X=X_agglo, obs=adata.obs, var=var, uns=adata.uns)
    adata.uns["agglo_feats"] = agglo_feats

    return adata


def estimate_k(
    adata: ad.AnnData,
    cluster_range: Union[List[int], Tuple[int]] = (2, 100),
    min_k: int = 3,
    make_plot: bool = False,
    return_fig: bool = False,
    show: bool = False,
    save: Optional[str] = None,
    **kwargs,
) -> Union[int, Tuple[int, plt.Figure]]:
    """Estimates k clusters with highest silhouette coefficient
    on subset of data for better performance.

    The silhouette coefficient is defined as:

    .. math::

        s = \\frac{b - a}{max(a, b)}

    with:
        a: The mean distance between a sample and all other points in the same class.
        b: The mean distance between a sample and all other points in the next nearest cluster.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    cluster_range : tuple of int or list of int
        Minimum and maximum numbers of clusters to compute
    min_k : int
        Minimum number of feature clusters.
        Two clusters tend to have very high silhouette coefficients.
    make_plot : bool
        Make plot to see results
    return_fig : bool
        Return best k and figure
    show : bool
        Show plot
    save : str, optional
        Path where to save figure
    **kwargs
        Keyword arguments passed to sklearn.cluster.FeatureAgglomeration

    Returns
    -------
    anndata.AnnData
        `.uns['agglo_feats']`: Agglomerated features and their original features

    Returns
    -------
    int, plt.Figure
        Best number of clusters and figure if `return_fig` is True

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np

    >>> data = np.random.rand(20, 100)
    >>> adata = ad.AnnData(data)
    >>> mp.ft.estimate_k(adata, cluster_range=(2, 10))
    Testing 2 to 10 ks: 100%|██████████| 8/8 [00:00<00:00, 273.32it/s]
    9
    """
    min_cluster, max_cluster = cluster_range

    # cache silhouette coefficients
    sil_coeffs = []
    ks = []

    # calculate silhouette scores for all ks
    iterator = range(min_cluster, max_cluster)
    for k in tqdm(iterator, desc=f"Testing {min_cluster} to {max_cluster} ks"):

        model = FeatureAgglomeration(n_clusters=k, **kwargs)
        agglo = model.fit(adata.X)

        # get silhouette score
        label = agglo.labels_
        sil_coeff = silhouette_score(adata.X.T, label, metric="euclidean")
        sil_coeffs.append(sil_coeff)
        ks.append(k)

    # get k maximum silhouette coefficient
    assert min_k in ks, f"min_k {min_k} is not in cluster_range: {cluster_range}"
    min_k_ix = ks.index(min_k)
    max_ix = np.argmax(sil_coeffs[min_k_ix:])
    max_k = ks[min_k_ix:][max_ix]

    # show
    if make_plot:
        # lineplot
        sns.set_theme()
        fig = plt.figure()
        sns.lineplot(x=ks, y=sil_coeffs)
        plt.xlabel("k (Number of clusters)")
        plt.axvline(
            max_k,
            color="firebrick",
            linestyle="dotted",
            label=f"Best k: {max_k}",
        )
        plt.title("Silhouette Score")
        plt.legend()

        # save
        if save is not None:
            try:
                plt.savefig(os.path.join(save, "estimate_k.png"))
            except OSError:
                print(f"Can not save figure to {save}.")

        if show:
            plt.show()

        if return_fig:
            return max_k, fig

    return max_k
