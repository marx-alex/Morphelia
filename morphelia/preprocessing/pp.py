# import external libraries
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


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
            print(f"Dropped NaN containing variables: {adata.var[~mask].index.tolist()}")

        adata = adata[:, masked_vars]

    return adata


def min_max_scaler(adata, min=0, max=1):
    """Wraper for sklearns MinMaxScaler.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        min, max (int): Desired range of transformed data.
    """
    scaler = MinMaxScaler(feature_range=(min, max))
    adata.X = scaler.fit_transform(adata.X)

    return adata


def filter_quant(adata, var, std_thresh=1.96, side='both'):
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
        print(f"Dropped duplicated features: {adata.var.index[[ix for ix in range(adata.X.shape[1]) if ix not in index]]}")

    # apply filter
    adata = adata[:, index]

    return adata


def z_transform(adata, by=("BatchNumber", "PlateNumber"), robust=False,
                clip=None, **kwargs):
    """Wrapper for sklearns StandardScaler to scale morphological data
    to unit variance by groups.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        by (iterable, str or None): Groups to apply function to.
            If None, apply to whole anndata.AnnData object.
        robust (bool): If true, use sklearn.preprocessing.RobustScaler
        clip (int): Clip (truncate) to this value after scaling. If None, do not clip.
        ** kwargs: Arguments passed to scaler.
    """
    # check that variables in by are in anndata
    if not all(var in adata.obs.columns for var in by):
        raise KeyError(f"Variables defined in 'by' are not in annotations: {by}")

    if isinstance(by, str):
        by = [by]
    elif isinstance(by, tuple):
        by = list(by)

    if robust:
        scaler = RobustScaler(unit_variance=True, **kwargs)
    else:
        scaler = StandardScaler(**kwargs)

    # iterate over md with grouping variables
    for groups, sub_df in adata.obs.groupby(by):
        # cache indices of group
        group_ix = sub_df.index
        # transform group with scaler
        adata[group_ix, :].X = scaler.fit_transform(adata[group_ix, :].X)

    if clip is not None:
        assert (clip > 0), f'Value for clip should be above 0, instead got {clip}'
        adata.X[adata.X > clip] = clip
        adata.X[adata.X < -clip] = -clip

    return adata


def drop_all_equal(adata, axis=0, verbose=False):
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
            print(f"Dropped uniform features: {dropped}")

        # apply mask
        adata = adata[:, ~mask]

    elif axis == 1:
        mask = np.all(adata.X == adata.X[:, 0], axis=axis)

        if verbose:
            dropped = adata.obs.index[mask]
            print(f"Dropped cells: {dropped}")

        # apply mask
        adata = adata[~mask, :]
    else:
        raise ValueError(f"axis should be 0 for columns and 1 for rows, "
                         f"instead got {axis}")

    return adata
