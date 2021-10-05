import numpy as np
from morphelia.plotting import plot_corr_matrix


def drop_highly_correlated(adata, thresh=0.95, show=False, save=False,
                           subsample=1000, seed=0, verbose=False,
                           neg_corr=True, drop=True, **kwargs):
    """Drops features that have a Pearson correlation coefficient
    with another feature above a certain threshold.
    Only one feature in a highly correlated group is kept.

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
        drop (bool): Drop features. If false add information to .var.
        **kwargs: Keyword arguments for sns.clustermap

    Returns:
        anndata.Anndata
        .uns['highly_correlated']: Highly correlated features.
        .uns['nan_feats']: Features with nan values.
        .var['highly_correlated']: True for highly correlated features.
            Only if drop is False.
        .var['contains_nan']: True for features that contain nan values.
            Only if drop is Fals.
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
            print(f"Dropped {len(drop_vars)} features: {drop_vars}")
        keep_vars = [var for var in all_vars if var not in drop_vars]

        if drop:
            adata = adata[:, keep_vars].copy()
            # store info in .uns
            adata.uns['highly_correlated'] = drop_vars
        else:
            mask = [True if var in drop_vars else False for var in all_vars]
            adata.var['highly_correlated'] = mask
    else:
        if drop:
            adata.uns['highly_correlated'] = []
        else:
            mask = [False for var in adata.var_names]
            adata.var['highly_correlated'] = mask

    # drop nan features
    if len(nan_feats) > 0:
        non_nan_feats = [feat for feat in adata.var_names if feat not in nan_feats]

        if verbose:
            print(f"Dropped uniform features: {nan_feats}")

        if drop:
            adata = adata[:, non_nan_feats].copy()

            if 'nan_feats' not in adata.uns:
                adata.uns['nan_feats'] = nan_feats
            else:
                adata.uns[nan_feats].append(nan_feats)

        else:
            mask = [True if feat in nan_feats else False for feat in adata.var_names]
            adata.var['contains_nan'] = mask
    else:
        if not drop:
            mask = [False for var in adata.var_names]
            adata.var['contains_nan'] = mask

    if show:
        if drop_ix is not None:
            groups = ['other features' if var in keep_vars else 'higly correlated features' for var in all_vars]
            if len(nan_feats) > 0:
                groups = ['invariant features' if var in nan_feats else label for label, var in zip(groups, all_vars)]
        else:
            groups = None

        # plot
        plot_corr_matrix(corr_matrix, groups, save=save, **kwargs)

    return adata

