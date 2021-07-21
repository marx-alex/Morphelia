# import internal libraries
import os
import warnings

# import external libraries
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def drop_highly_correlated(adata, thresh=0.95, show=False, save=False,
                           subsample=1000, seed=0, verbose=False,
                           neg_corr=False, **kwargs):
    """Drops features that have a Pearson correlation coefficient
    with another feature above a certain threshold.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        thresh (float): Correlated features with Pearson correlation coefficient
            above threshold get dropped.
        show (bool): True to get figure.
        save (str): Path where to save figure.
        subsample (int): If given, retrieves subsample of all cell data for speed.
        seed (int): Seed for subsample calculation.
        verbose (bool)
        neg_corr (bool): Include negative correlated features.
        **kwargs: Keyword arguments for sns.clustermap

    Returns:
        (anndata.Anndata)
    """

    # get subsample
    if subsample is not None:
        assert isinstance(subsample, int), f"expected type for subsample is int, instead got {type(subsample)}"
        # get samples
        np.random.seed(seed)
        X_len = adata.shape[0]
        if subsample > X_len:
            subsample = X_len
        sample_ix = np.random.randint(X_len, size=subsample)
        try:
            adata_ss = adata[sample_ix, :]
        except:
            adata_ss = adata.X.copy()
    else:
        adata_ss = adata.X.copy()

    # calculate correlation coefficients
    corr_matrix = np.corrcoef(adata_ss.X.T)

    # cache features with only nan correlations
    nan_mask = np.all(np.isnan(corr_matrix), axis=0)
    nan_feats = adata.var_names[nan_mask]

    # triangular matrix
    tri = corr_matrix.copy()
    tri[~np.triu(np.ones(tri.shape), k=1).astype(bool)] = np.nan

    # get features above threshold
    if neg_corr:
        drop_ix = np.argwhere((tri > thresh) | (tri < -thresh))
    elif neg_corr is False:
        drop_ix = np.argwhere(tri > thresh)
    else:
        raise TypeError(f"neg_corr expected to be a boolean, instead got {type(neg_corr)}")

    if len(drop_ix) > 0:
        drop_ix = list(set(drop_ix[:, 1].tolist()))
    else:
        print(f'No highly correlated features found with threshold: {thresh}.')
        drop_ix = None

    # drop highly correlated features
    if drop_ix is not None:
        all_vars = adata.var_names
        drop_vars = all_vars[drop_ix]

        if verbose:
            print(f"Dropped features: {drop_vars}")
        keep_vars = [var for var in all_vars if var not in drop_vars]
        adata = adata[:, keep_vars]

    # drop nan features
    if len(nan_feats) > 0:
        non_nan_feats = [feat for feat in adata.var_names if feat not in nan_feats]

        if verbose:
            print(f"Dropped uniform features: {nan_feats}")

        adata = adata[:, non_nan_feats]

    if show:
        # do not show features with only nan
        corr_matrix = np.nan_to_num(corr_matrix)

        kwargs.setdefault('cmap', 'viridis')
        kwargs.setdefault('method', 'ward')
        kwargs.setdefault('vmin', -1)
        kwargs.setdefault('vmax', 1)

        # get group labels
        row_colors = None
        handles = None
        if drop_ix is not None:
            groups = ['other features' if var in keep_vars else 'higly correlated features' for var in all_vars]
            if len(nan_feats) > 0:
                groups = ['uniform features' if var in nan_feats else label for label, var in zip(groups, all_vars)]
            unique_groups = set(groups)
            colors = sns.color_palette("Set2", len(unique_groups))
            col_map = dict(zip(unique_groups, colors))
            row_colors = list(map(col_map.get, groups))
            handles = [Patch(facecolor=col_map[name]) for name in col_map]

        sns.set_theme()
        plt.figure()
        cm = sns.clustermap(corr_matrix, row_colors=row_colors, **kwargs)
        plt.suptitle('Correlation between features', y=1.05, fontsize=16)
        cm.ax_row_dendrogram.set_visible(False)
        ax = cm.ax_heatmap
        ax.get_yaxis().set_visible(False)
        if adata.shape[1] > 50:
            warnings.warn("Labels are hidden with more than 50 features", UserWarning)
            ax.get_xaxis().set_visible(False)
        if handles is not None:
            lgd = plt.legend(handles, col_map, bbox_to_anchor=(1, 1),
                             bbox_transform=plt.gcf().transFigure, loc='upper left',
                             frameon=False)

        # save
        if save:
            try:
                plt.savefig(os.path.join(save, "feature_correlation.png"),
                            bbox_extra_artists=[lgd], bbox_inches='tight')
            except OSError:
                print(f'Can not save figure to {save}.')

    return adata
