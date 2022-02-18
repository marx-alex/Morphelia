import numpy as np
import warnings
import logging

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def drop_outlier(adata,
                 thresh=15,
                 axis=0,
                 drop=True,
                 verbose=False):
    """Drop all features or cells with a min or max absolute value that is greater than a threshold.

    Only use with normally distributed data.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        thresh (int): Threshold for outlier identification.
        axis (int): 0 means along features, 1 means along cells.
        drop (bool): Drop features/ cells with outliers if True.
        verbose (bool)

    Returns:
        anndata.AnnData
        Only if axis is 0:
        .uns['outlier_feats']: Dropped features with outliers.
        .var['outlier_feats']: True for features that contain outliers.
            Only if drop is False.
    """
    assert axis in [0, 1], f"axis has to be either 0 (features) or 1 (cells), instead got {axis}"

    max_values = np.abs(np.max(adata.X, axis=axis))
    min_values = np.abs(np.min(adata.X, axis=axis))

    assert isinstance(thresh, (int, float)), f"thresh expected to be of type(int) or type(float), " \
                                             f"instead got {type(thresh)}"

    mask = np.logical_and((max_values <= thresh), (min_values <= thresh))

    if axis == 0:
        dropped_feats = adata.var_names[~mask]
        if verbose:
            logger.info(f"Drop {len(dropped_feats)} features with outlier values: {dropped_feats}")

        # drop features
        if drop:
            adata = adata[:, mask].copy()
            adata.uns['outlier_feats'] = dropped_feats
        else:
            adata.var['outlier_feats'] = ~mask

    else:
        n_before = len(adata)

        if drop:
            adata = adata[mask, :].copy()
        if verbose:
            logger.info(f"{n_before - len(adata)} cells removed with feature values >= or <= {thresh}")

    return adata
