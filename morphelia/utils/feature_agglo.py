# external libraries
from sklearn.cluster import FeatureAgglomeration
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scipy.stats as st
import anndata as ad
import pandas as pd

# internal libraries
import os
import warnings


def feature_agglo(md, k='estimate', cluster_range=(2, 100), cutoff_mad=0.25,
                  subsample=1000, seed=0, group_by=None,
                  show=False, save=None, **kwargs):
    """Wrapper for scikits FeatureAgglomeration.
    Calculates clusters by hierarchical clustering and replaces clusters by centers.
    Expects z-tranformed data.

    Args:
        md (anndata.AnnData): Multidimensional morphological data.
        k ('estimate' or int): Number of clusters to compute. If 'estimate'
            estimates ideal number of k with subset of data.
        cluster_range (list): Minimum and maximum numbers of clusters to compute.
        cutoff_mad (int): Maximum sum of intra-cluster median absolute deviation.
        subsample (int): If given, retrieves subsample of all cell data for speed.
        seed (int): Seed for subsample calculation.
        group_by (pd.Series): Color groups of cells.
        show (bool): True to get figure.
        save (str): Path where to save figure.

    Returns:
        anndata.AnnData
    """
    # check cluster_range
    if not len(cluster_range) == 2:
        raise ValueError("cluster_range should be a tuple or list of length 2, "
                         f"instead got: {cluster_range}")
    if cluster_range[1] > md.shape[1]:
        warnings.warn("Maximal expected k is larger than number of features", UserWarning)
        cluster_range = list(cluster_range)
        cluster_range[1] = md.shape[1]

    # initiate subsample
    X_ss = None

    # get subsample
    if k == 'estimate':
        assert isinstance(subsample, int), f"Define a subsample size (int) if k is 'estimate': {subsample}"
        # get samples
        np.random.seed(seed)
        X_len = md.shape[0]
        sample_ix = np.random.randint(X_len, size=subsample)
        try:
            X_ss = md.X.copy()[sample_ix]
            obs_ss = md.obs.copy().iloc[sample_ix]
        except:
            X_ss = md.X.copy()
            obs_ss = md.obs.copy()

        # estimate k
        k = _estimate_k(X_ss, obs_ss, cluster_range=cluster_range, group_by=group_by,
                        show=show, save=save, cutoff=cutoff_mad, **kwargs)

    # calculate feature agglomeration
    if isinstance(k, int):
        model = FeatureAgglomeration(n_clusters=k, **kwargs)

        if X_ss is not None:
            agglo = model.fit(X_ss)
        else:
            agglo = model.fit(md.X)

        X_red = model.transform(md.X)
    else:
        raise TypeError(f"k should be an integer, instead got {type(k)}")

    # add unstructured data:
    # dict of labels from new features with old features as values
    feat_lst = list(zip(agglo.labels_, md.var_names))
    agglo_feats = {}
    for pair in feat_lst:
        agglo_feats.setdefault(f"agglo_{pair[0]}", []).append(pair[1])

    # create new anndata object
    var = pd.DataFrame(index=[f"agglo_{i}" for i in set(agglo.labels_)])
    md = ad.AnnData(X=X_red, obs=md.obs, var=var)
    md.uns['agglo_feats'] = agglo_feats

    return md


def _estimate_k(X_ss, obs_ss, cluster_range, group_by,
                show=False, save=None, cutoff=None, **kwargs):
    """Estimates k clusters with low mean intra-cluster median
    absolute deviation on subset of data for better perfomance.

    Args:
        X_ss (np.ndarray): cells x features.
        obs_ss (pd.DataFrame): Subset of observations.
        cluster_range (list): Minimum and maximum numbers of clusters to compute.
        row_colors (np.array): Color groups of cells.
        show (bool): Plots MAD score and number of clusters.
        save (str): Path where to save figure.
        cutoff (int): Cutoff for MAD score.

    Returns:
        k (int): Number of clusters.
    """
    min_cluster, max_cluster = cluster_range

    # cache dispersion index sums
    mad_means = []
    ks = []

    # calculate intra-cluster dispersion factor for all ks
    for k in range(min_cluster, max_cluster):

        model = FeatureAgglomeration(n_clusters=k, **kwargs)
        agglo = model.fit(X_ss)
        # X_red = model.transform(X_ss)

        # get mean of intra-cluster MADs
        mads = []
        for k_ix in range(k):
            mask = agglo.labels_ == k_ix
            cluster_merge = X_ss[:, mask]
            mad = np.mean(st.median_abs_deviation(cluster_merge, axis=1))
            mads.append(mad)
        mad_mean = np.mean(mads)
        mad_means.append(mad_mean)
        ks.append(k)

    # get k with minimum MAD
    min_ix = None
    if cutoff is not None:
        min_ix = next((ix for ix, mad in enumerate(mad_means) if mad < cutoff), None)
    if min_ix is None:
        warnings.warn("MAD cutoff not reached, global minimum taken", UserWarning)
        min_ix = np.argmin(mad_means)
    min_k = ks[min_ix]

    # show
    if show:
        # lineplot
        sns.set_theme()
        plt.figure(1)
        p = sns.lineplot(x=ks, y=mad_means)
        plt.xlabel('Clusters')
        plt.ylabel('Mean intra-cluster MAD')
        plt.axvline(min_k, color='k', linestyle='dotted', label=f'Min k: {min_k}')
        if cutoff is not None:
            plt.axhline(cutoff, color='r', linestyle='dotted', label=f'Cutoff: {cutoff}')
        plt.title(f"Estimation of k on subsample of {X_ss.shape[0]} cells")
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
                group_by = obs_ss[group_by].tolist()
            except KeyError:
                print(f"{group_by} not in anndata.AnnData.obs")
            unique_groups = set(group_by)
            colors = sns.color_palette("Set2", len(unique_groups))
            col_map = dict(zip(unique_groups, colors))
            row_colors = list(map(col_map.get, group_by))
            handles = [Patch(facecolor=col_map[name]) for name in col_map]

        plt.figure(2)
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        g = sns.clustermap(X_ss, row_cluster=True,
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
                plt.savefig(os.path.join(save, "heatmap_feat_agglo.png"),
                            bbox_extra_artists=[lgd], bbox_inches='tight')
            except OSError:
                print(f'Can not save figure to {save}.')

    return min_k
