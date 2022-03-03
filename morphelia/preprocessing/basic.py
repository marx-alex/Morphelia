# import external libraries
import numpy as np

import logging

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def drop_nan(
    adata,
    axis=0,
    drop_inf=True,
    drop_dtype_max=True,
    min_nan_frac=None,
    verbose=False,
):
    """Drop rows or columns that contain invalid values.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        axis (int): Drop features (0) or cells (1).
        drop_inf (bool): Drop also infinity values.
        drop_dtype_max (bool): Drop values to large for dtype.
        min_nan_frac (float): Minimum fraction of column/ or row that has to be invalid to be dropped.
        verbose (bool)

    Returns:
        anndata.AnnData
        .uns['nan_feats']: Dropped features that contain nan values.
    """
    assert axis in [0, 1], f"axis should be either 0 or 1, but got {axis}"
    n_vals = adata.X.shape[axis]

    if drop_inf:
        adata.X[(adata.X == np.inf) | (adata.X == -np.inf)] = np.nan

    if drop_dtype_max:
        adata.X[adata.X > np.finfo(adata.X.dtype).max] = np.nan

    n_nan = np.isnan(adata.X).sum(axis=axis)

    # nan values must be a minimum fraction of all values
    if min_nan_frac is not None:
        assert (
            0 <= min_nan_frac <= 1
        ), "min_nan_frac should be of type float and between 0 and 1, instead got {min_nan_frac}"
        frac = n_nan / n_vals
        mask = frac >= min_nan_frac
    else:
        mask = n_nan > 0

    # --> filter features
    if axis == 0:
        dropped = list(adata.var_names[mask])

        if verbose:
            logger.info(
                f"Dropped {len(dropped)} features with missing values: {dropped}"
            )

        adata = adata[:, ~mask].copy()

        # store dropped features in adata.uns
        if "nan_feats" in adata.uns:
            adata.uns["nan_feats"].append(dropped)
        else:
            adata.uns["nan_feats"] = dropped

    # if axis 1 --> filter cells
    else:
        dropped = list(adata.obs.index[mask])

        if verbose:
            logger.info(f"Dropped {len(dropped)} cells with missing values: {dropped}")

        adata = adata[~mask, :].copy()

        # store dropped features in adata.uns
        if "nan_cells" in adata.uns:
            adata.uns["nan_cells"].append(dropped)
        else:
            adata.uns["nan_cells"] = dropped

    return adata


def filter_std(adata, var, std_thresh=3, side="both"):
    """Filter cells by identifying outliers in a distribution
    of a single value.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        var (str): Variable to use for filtering.
        std_thresh (int): Threshold to use for outlier identification.
            x-fold of standard deviation in both or one directions.
        side (str): 'left', 'right' or 'both'

    Returns:
        anndata.AnnData
    """
    # get standard deviation and mean
    std = np.nanstd(adata[:, var].X)
    mean = np.nanmean(adata[:, var].X)

    # do filtering
    if side not in ["left", "right", "both"]:
        raise ValueError(f"side should be 'left', 'right' or 'both': {side}")
    if side == "both" or side == "left":
        adata = adata[adata[:, var].X > (mean - (std_thresh * std)), :]
    if side == "both" or side == "right":
        adata = adata[adata[:, var].X < (mean + (std_thresh * std)), :]

    return adata


def filter_thresh(adata, var, thresh, side="right"):
    """Filter cells by thresholding.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        var (str): Variable to use for filtering.
        thresh (int): Threshold.
        side (str):
            'right': Drop values above threshold.
            'left': Drop values below threshold.

    Returns:
        anndata.AnnData
    """
    sides = ["right", "left"]
    assert (
        side in sides
    ), f"expected side to be either 'right' or 'left', instead got {side}"

    # do filtering
    if var in adata.var_names:
        if side == "right":
            adata = adata[adata[:, var].X < thresh, :]
        elif side == "left":
            adata = adata[adata[:, var].X > thresh, :]
    elif var in adata.obs.columns:
        if side == "right":
            adata = adata[adata.obs[var] < thresh, :]
        elif side == "left":
            adata = adata[adata.obs[var] > thresh, :]
    else:
        raise KeyError(f"Variable not in AnnData object: {var}")

    return adata


def drop_duplicates(adata, axis=1, verbose=False):
    """Drops all feature vectors or cells that are duplicates.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        axis (int): Drop duplicated features (1) or cells (0).
        verbose (bool)

    Returns:
        anndata.AnnData
        .uns['duplicated_feats']: Dropped duplicated features.
    """
    assert axis in [0, 1], f"axis should be either 0 or 1, but got {axis}"

    _, index = np.unique(adata.X, axis=axis, return_index=True)
    # index to mask
    mask = np.zeros(adata.X.shape[axis])
    mask[index] = 1
    mask = mask.astype(bool)

    if axis == 1:
        dropped = list(adata.var_names[~mask])

        if verbose:
            logger.info(f"Dropped {len(dropped)} duplicated features: {dropped}")

        # apply filter
        adata = adata[:, mask].copy()

        # store dropped features in adata.uns
        if "duplicated_feats" in adata.uns:
            adata.uns["duplicated_feats"].append(dropped)
        else:
            adata.uns["duplicated_feats"] = dropped

    else:
        dropped = list(adata.obs.index[~mask])

        if verbose:
            logger.info(f"Dropped {len(dropped)} duplicated cells: {dropped}")

        adata = adata[mask, :].copy()

        # store dropped cells in adata.uns
        if "duplicated_cells" in adata.uns:
            adata.uns["duplicated_cells"].append(dropped)
        else:
            adata.uns["duplicated_cells"] = dropped

    return adata


def drop_invariant(adata, axis=0, verbose=False):
    """Drops rows (cells) or columns (features)
    if all values are equal.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        axis (int): 0 for columns, 1 for rows.
        verbose (bool)

    Returns:
        anndata.AnnData
        .uns['invariant_feats']: Dropped invariant features.
    """
    # check if axis exists
    assert (
        axis < adata.X.ndim
    ), f"adata.X with {adata.X.ndim} dimensions has no axis {axis}"

    # get mask and apply it to adata
    if axis == 0:
        comp = adata.X[0, :]
        mask = np.all(adata.X == comp[None, :], axis=axis)
        dropped = list(adata.var_names[mask])

        if verbose:
            logger.info(f"Dropped {len(dropped)} invariant features: {dropped}")

        # apply mask
        adata = adata[:, ~mask].copy()

        adata.uns["invariant_feats"] = dropped

    elif axis == 1:
        comp = adata.X[:, 0]
        mask = np.all(adata.X == comp[:, None], axis=axis)
        dropped = list(adata.obs.index[mask])

        if verbose:
            logger.info(f"Dropped {len(dropped)} invariant cells: {dropped}")

        # apply mask
        adata = adata[~mask, :].copy()

        adata.uns["invariant_cells"] = dropped

    return adata
