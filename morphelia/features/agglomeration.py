# external libraries
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import silhouette_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import anndata as ad
import pandas as pd
from tqdm import tqdm

# internal libraries
import os
import warnings

from morphelia.tools._utils import _get_subsample


def feature_agglo(adata,
                  k='estimate',
                  cluster_range=(2, 100),
                  subsample=False,
                  sample_size=1000,
                  seed=0,
                  group_by=None,
                  show=False,
                  save=None,
                  **kwargs):
    """Wrapper for scikits FeatureAgglomeration.
    Calculates clusters by hierarchical clustering and replaces clusters by centers.
    Expects z-tranformed data.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        k ('estimate' or int): Number of clusters to compute. If 'estimate'
            estimates ideal number of k with subset of data.
        cluster_range (list): Minimum and maximum numbers of clusters to compute.
        subsample (bool): If True, fit models on subsample of data.
        sample_size (int): Size of supsample.
            Only if subsample is True.
        seed (int): Seed for subsample calculation.
            Only if subsample is True.
        group_by (pd.Series): Color groups of cells.
        show (bool): True to get figure.
        save (str): Path where to save figure.

    Returns:
        anndata.AnnData
        .uns['agglo_feats']: Agglomerated features and their original features.
    """
    # check cluster_range
    try:
        cluster_range = list(cluster_range)
    except TypeError:
        print("list or tuple expected for cluster_range "
              f"instead got: {type(cluster_range)}")
    if not len(cluster_range) == 2:
        raise ValueError("cluster_range should be a tuple or list of length 2, "
                         f"instead got: {cluster_range}")
    if cluster_range[1] > adata.shape[1]:
        warnings.warn("Maximal expected k is larger than number of features", UserWarning)
        cluster_range[1] = adata.shape[1]
    if cluster_range[0] < 2:
        warnings.warn(f"Minimum k is 2, changes {cluster_range[0]} to 2", UserWarning)
        cluster_range[0] = 2

    # check for z-transformed data
    norm_mean = np.nanmean(adata.X)
    if norm_mean > 0.1 or norm_mean < -0.1:
        warnings.warn("Array does not seem to be normally distributed.", UserWarning)

    # check for nan values
    if np.isnan(adata.X).any():
        warnings.warn("Array contains NaN.", UserWarning)

    # get subsample
    if subsample:
        adata_ss = _get_subsample(adata,
                                  sample_size=sample_size,
                                  seed=seed)
    else:
        adata_ss = adata.copy()

    if k == 'estimate':
        k = estimate_k(adata_ss,
                       cluster_range=cluster_range,
                       group_by=group_by,
                       show=show,
                       save=save,
                       **kwargs)

    # calculate feature agglomeration
    if isinstance(k, int):
        model = FeatureAgglomeration(n_clusters=k, **kwargs)

        agglo = model.fit(adata_ss.X)

        X_agglo = model.transform(adata.X)
    else:
        raise TypeError(f"k should be an integer, instead got {type(k)}")

    # add unstructured data:
    # dict of labels from new features with old features as values
    feat_lst = list(zip(agglo.labels_, adata.var_names))
    agglo_feats = {}
    for pair in feat_lst:
        agglo_feats.setdefault(f"agglo_{pair[0]}", []).append(pair[1])

    # create new anndata object
    var = pd.DataFrame(index=[f"agglo_{i}" for i in set(agglo.labels_)])
    adata = ad.AnnData(X=X_agglo, obs=adata.obs, var=var, uns=adata.uns)
    adata.uns['agglo_feats'] = agglo_feats

    return adata


def estimate_k(adata,
               cluster_range,
               group_by,
               min_k=3,
               show=False,
               save=None,
               **kwargs):
    """Estimates k clusters with highest silhouette coefficient
     on subset of data for better performance.

     The silhouette coefficient is defined as:

     s = (b - a) / max(a, b)

     with:
        a: The mean distance between a sample and all other points in the same class.
        b: The mean distance between a sample and all other points in the next nearest cluster.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        cluster_range (list): Minimum and maximum numbers of clusters to compute.
        group_by (np.array): Color groups of cells.
        min_k (int): Minimum number of feature clusters.
            Two clusters tend to have very high silhouette coefficients.
        show (bool): Plots MAD score and number of clusters.
        save (str): Path where to save figure.

    Returns:
        k (int): Number of clusters.
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
        sil_coeff = silhouette_score(adata.X.T, label, metric='euclidean')
        sil_coeffs.append(sil_coeff)
        ks.append(k)

    # get k maximum silhouette coefficient
    assert min_k in ks, f"min_k {min_k} is not in cluster_range: {cluster_range}"
    min_k_ix = ks.index(min_k)
    max_ix = np.argmax(sil_coeffs[min_k_ix:])
    max_k = ks[min_k_ix:][max_ix]

    # show
    if show:
        # lineplot
        sns.set_theme()
        plt.figure(1)
        p = sns.lineplot(x=ks, y=sil_coeffs)
        plt.xlabel('k (Number of clusters)')
        plt.ylabel('Silhouette Coefficient')
        plt.axvline(max_k, color='firebrick', linestyle='dotted', label=f'Est. k: {max_k}')
        plt.title(f"Estimation of k on subsample of N = {adata.X.shape[0]}")
        plt.legend()

        # save
        if save is not None:
            try:
                plt.savefig(os.path.join(save, "estimate_k.png"))
            except OSError:
                print(f'Can not save figure to {save}.')

        # heatmap
        # get cell groups
        row_colors = None
        handles = None
        if group_by is not None:
            try:
                group_by = adata.obs[group_by].tolist()
            except KeyError:
                print(f"{group_by} not in anndata.AnnData.obs")
            unique_groups = set(group_by)
            colors = sns.color_palette("Set2", len(unique_groups))
            col_map = dict(zip(unique_groups, colors))
            row_colors = list(map(col_map.get, group_by))
            handles = [Patch(facecolor=col_map[name]) for name in col_map]

        plt.figure(2)
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        g = sns.clustermap(adata.X, row_cluster=True,
                           row_colors=row_colors,
                           metric='euclidean', method='ward',
                           cmap=cmap, vmin=-3, vmax=3)
        ax = g.ax_heatmap
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_yticks([])
        lgd = None
        if handles is not None:
            lgd = plt.legend(handles, col_map, bbox_to_anchor=(1, 1),
                             bbox_transform=plt.gcf().transFigure, loc='upper left',
                             frameon=False)

        # save
        if save is not None:
            try:
                if handles is not None:
                    plt.savefig(os.path.join(save, "heatmap_feat_agglo.png"),
                                bbox_extra_artists=[lgd], bbox_inches='tight')
                else:
                    plt.savefig(os.path.join(save, "heatmap_feat_agglo.png"))
            except OSError:
                print(f'Can not save figure to {save}.')

    return max_k
