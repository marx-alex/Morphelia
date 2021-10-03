# import external libraries
import numpy as np


def drop_nan(adata, drop_inf=True, drop_dtype_max=True, verbose=False):
    """Drop variables that contain invalid variables.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        drop_inf (bool): Drop also infinity values.
        drop_dtype_max (bool): Drop values to large for dtype.
        verbose (bool)
    """
    if np.isnan(adata.X).any():
        if drop_inf:
            adata.X[(adata.X == np.inf) | (adata.X == -np.inf)] = np.nan

        if drop_dtype_max:
            adata.X[adata.X > np.finfo(adata.X.dtype).max] = np.nan

        mask = ~np.isnan(adata.X).any(axis=0)
        masked_vars = adata.var[mask].index.tolist()

        if verbose:
            print(f"Dropped {adata.shape[1] - len(masked_vars)} features with missing values:"
                  f" {adata.var[~mask].index.tolist()}")

        adata = adata[:, masked_vars]

    return adata


def filter_std(adata, var, std_thresh=3, side='both'):
    """Filter cells by identifying outliers from a distribution
    of a single value.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        var (str): Variable to use for filtering.
        std_thresh (int): Threshold to use for outlier identification.
            x-fold of standard deviation in both or one directions.
        side (str): 'left', 'right' or 'both'
    """
    # get standard deviation and mean
    std = np.nanstd(adata[:, var].X)
    mean = np.nanmean(adata[:, var].X)

    # do filtering
    if side not in ['left', 'right', 'both']:
        raise ValueError(f"side should be 'left', 'right' or 'both': {side}")
    if side == 'both' or side == 'left':
        adata = adata[adata[:, var].X > (mean - (std_thresh * std)), :]
    if side == 'both' or side == 'right':
        adata = adata[adata[:, var].X < (mean + (std_thresh * std)), :]

    return adata


def filter_thresh(adata, var, thresh, side='right'):
    """Filter cells by thresholding.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        var (str): Variable to use for filtering.
        thresh (int): Threshold.
        side (str):
            'right': Cap values above threshold.
            'left': Cap values under threshold.
    """
    sides = ['right', 'left']
    assert side in sides, f"expected side to be either 'right' or 'left', instead got {side}"

    # do filtering
    if var in adata.var_names:
        if side == 'right':
            adata = adata[adata[:, var].X < thresh, :]
        elif side == 'left':
            adata = adata[adata[:, var].X > thresh, :]
    elif var in adata.obs.columns:
        if side == 'right':
            adata = adata[adata.obs[var] < thresh, :]
        elif side == 'left':
            adata = adata[adata.obs[var] > thresh, :]
    else:
        raise ValueError(f"Variable not in AnnData object: {var}")

    return adata


def drop_duplicates(adata, verbose=False):
    """Drops all feature vectors that are duplicates.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        verbose (bool)
    """
    _, index = np.unique(adata.X, axis=1, return_index=True)

    if verbose:
        print(f"Dropped {adata.shape[1] - len(index)} duplicated features:"
              f" {adata.var.index[[ix for ix in range(adata.X.shape[1]) if ix not in index]]}")

    # apply filter
    adata = adata[:, index]

    return adata


def drop_invariant(adata, axis=0, verbose=False):
    """Drops rows (cells) or columns (features)
    if all values are equal.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        axis (int): 0 for columns, 1 for rows.
        verbose (bool)
    """
    # check if axis exists
    if axis >= adata.X.ndim:
        raise ValueError(f"adata.X with {adata.X.ndim} dimensions"
                         f"has no axis {axis}")

    # get mask and apply it to adata
    if axis == 0:
        mask = np.all(adata.X == adata.X[0, :], axis=axis)

        if verbose:
            dropped = adata.var_names[mask]
            print(f"Dropped {len(dropped)} invariant features: {dropped}")

        # apply mask
        adata = adata[:, ~mask]

    elif axis == 1:
        mask = np.all(adata.X == adata.X[:, 0], axis=axis)

        if verbose:
            dropped = adata.obs.index[mask]
            print(f"Dropped {len(dropped)} invariant cells: {dropped}")

        # apply mask
        adata = adata[~mask, :]
    else:
        raise ValueError(f"axis should be 0 for columns and 1 for rows, "
                         f"instead got {axis}")

    return adata
