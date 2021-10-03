import numpy as np


def drop_outlier(adata,
                 thresh=15,
                 verbose=False):
    """Drop all features with a min or max absolute value that is greater than a threshold.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        thresh (int): Threshold for outlier identification.
        verbose (bool)
    """
    max_feature_values = np.abs(np.max(adata.X, axis=0))
    min_feature_values = np.abs(np.max(adata.X, axis=0))

    assert isinstance(thresh, (int, float)), f"thresh expected to be of type(int) or type(float), " \
                                             f"instead got {type(thresh)}"

    mask = np.logical_or((max_feature_values <= thresh), (min_feature_values <= thresh))

    if verbose:
        dropped_feats = adata.var_names[~mask]
        print(f"Drop {len(dropped_feats)} features with outlier values: {dropped_feats}")

    # drop features
    adata = adata[:, mask]

    return adata