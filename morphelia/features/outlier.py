import numpy as np
import logging
from typing import Union

import anndata as ad

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def thresh_outlier(
    adata: ad.AnnData,
    thresh: Union[int, float] = 3.5,
    axis: int = 0,
    drop: bool = True,
    verbose: bool = False,
) -> ad.AnnData:
    """Drop outlier features or cells.

    This function drops all features or cells with
    a min or max absolute value that is greater than a threshold.

    Scale the data beforehand.

    The following information is stored if `axis` is 0:
        .uns['outlier_feats']: Dropped features with outliers.

        .var['outlier_feats']: True for features that contain outliers. Only if drop is False.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data.
    thresh : int
        Threshold for outlier identification.
    axis : int
        0 means along features, 1 means along cells.
    drop : bool
        Drop features/ cells with outliers if True.
    verbose : bool

    Returns
    -------
    anndata.AnnData
        AnnData object without dropped features if `drop` is True

    Raises
    -------
    AssertionError
        If `axis` is neither 0 nor 1
    AssertionError
        If `thresh` is not of type int or float

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np

    >>> data = np.random.rand(10, 5)
    >>> adata = ad.AnnData(data)
    >>> mp.ft.thresh_outlier(adata, thresh=3)
    AnnData object with n_obs × n_vars = 10 × 5
        uns: 'outlier_feats'
    """
    assert axis in [
        0,
        1,
    ], f"axis has to be either 0 (features) or 1 (cells), instead got {axis}"

    max_values = np.max(np.abs(adata.X), axis=axis)

    assert isinstance(thresh, (int, float)), (
        f"thresh expected to be of type int or float, " f"instead got {type(thresh)}"
    )

    mask = max_values <= thresh

    if axis == 0:
        dropped_feats = adata.var_names[~mask]
        if verbose:
            logger.info(
                f"Drop {len(dropped_feats)} features with outlier values: {dropped_feats}"
            )

        # drop features
        if drop:
            adata = adata[:, mask].copy()
            adata.uns["outlier_feats"] = dropped_feats
        else:
            adata.var["outlier_feats"] = ~mask

    else:
        n_before = len(adata)

        if drop:
            adata = adata[mask, :].copy()
        if verbose:
            logger.info(f"{n_before - len(adata)} cells removed with outlier values")

    return adata
