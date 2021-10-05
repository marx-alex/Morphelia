import numpy as np


def drop_outlier(adata,
                 thresh=15,
                 drop=True,
                 verbose=False):
    """Drop all features with a min or max absolute value that is greater than a threshold.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        thresh (int): Threshold for outlier identification.
        drop (bool): Drop features with outliers if True.
        verbose (bool)

    Returns:
        anndata.AnnData
        .uns['outlier_feats']: Dropped features with outliers.
        .var['outlier_feats']: True for features that contain outliers.
            Only if drop is False.
    """
    max_feature_values = np.abs(np.max(adata.X, axis=0))
    min_feature_values = np.abs(np.min(adata.X, axis=0))

    assert isinstance(thresh, (int, float)), f"thresh expected to be of type(int) or type(float), " \
                                             f"instead got {type(thresh)}"

    mask = np.logical_and((max_feature_values <= thresh), (min_feature_values <= thresh))

    dropped_feats = adata.var_names[~mask]
    if verbose:
        print(f"Drop {len(dropped_feats)} features with outlier values: {dropped_feats}")

    # drop features
    if drop:
        adata = adata[:, mask].copy()
        adata.uns['outlier_feats'] = dropped_feats
    else:
        adata.var['outlier_feats'] = ~mask

    return adata