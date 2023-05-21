# import external libraries
import numpy as np
import anndata as ad

import logging
from typing import Optional, Union

import morphelia as mp

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def drop_nan(
    adata: ad.AnnData,
    axis: int = 0,
    drop_inf: bool = True,
    drop_dtype_max: bool = True,
    min_nan_frac: Optional[Union[float, int]] = None,
    obsm: Optional[str] = None,
    verbose: bool = False,
):
    """Drop rows or columns that contain invalid values.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    axis : int
        Drop cells (0) or features (1)
    drop_inf : bool
        Drop infinite values
    drop_dtype_max : bool
        Drop values to large for dtype
    min_nan_frac : float or int, optional
        Minimum fraction of column/ or row that has to be invalid to be dropped
    obsm : str, optional
        If provided, which element of obsm to scale
    verbose : bool

    Returns
    -------
    anndata.AnnData
        Filtered AnnData object
        .uns['nan_feats']: Dropped features that contain nan values

    Raises
    ------
    AssertionError
        If axis if neither `0` nor `1`
    AssertionError
        If min_nan_frac is not between `0` and `1`

    Examples
    --------
    >>> import morphelia as mp
    >>> import anndata as ad
    >>> import numpy as np

    >>> data = np.random.rand(5, 5)
    >>> data[0, 0] = np.nan

    >>> adata = ad.AnnData(data)
    >>> mp.pp.drop_nan(adata)
    AnnData object with n_obs × n_vars = 5 × 4
        uns: 'nan_feats'
    """
    assert axis in [0, 1], f"axis should be either 0 or 1, but got {axis}"

    x = mp.tl.choose_layer(adata, obsm=obsm)

    if drop_inf:
        x[(x == np.inf) | (x == -np.inf)] = np.nan

    if drop_dtype_max:
        x[x > np.finfo(x.dtype).max] = np.nan

    n_nan = np.isnan(x).sum(axis=1 if axis == 0 else 0)

    # nan values must be a minimum fraction of all values
    if min_nan_frac is not None:
        m, n = x.shape
        assert (
            0 <= min_nan_frac <= 1
        ), "min_nan_frac should be of type float and between 0 and 1, instead got {min_nan_frac}"
        if axis == 0:
            frac = n_nan / n
        else:
            frac = n_nan / m
        mask = frac >= min_nan_frac
    else:
        mask = n_nan > 0

    # if axis 0 --> filter cells
    if axis == 0:
        dropped = list(adata.obs.index[mask])

        if verbose:
            logger.info(f"Dropped {len(dropped)} cells with missing values: {dropped}")

        adata = adata[~mask, :].copy()

        # store dropped features in adata.uns
        if "nan_cells" in adata.uns:
            if isinstance(adata.uns["nan_cells"], list):
                adata.uns["nan_cells"].extend(dropped)
            else:
                adata.uns["nan_cells"] = dropped
        else:
            adata.uns["nan_cells"] = dropped

    # --> filter features
    else:
        dropped = list(adata.var_names[mask])

        if verbose:
            logger.info(
                f"Dropped {len(dropped)} features with missing values: {dropped}"
            )

        adata = adata[:, ~mask].copy()

        # store dropped features in adata.uns
        if "nan_feats" in adata.uns:
            if isinstance(adata.uns["nan_feats"], list):
                adata.uns["nan_feats"].extend(dropped)
            else:
                adata.uns["nan_feats"] = dropped
        else:
            adata.uns["nan_feats"] = dropped

    return adata


def filter_std(
    adata: ad.AnnData, var: str, std_thresh: Union[int, float] = 3, side: str = "both"
):
    """Filter by standard deviation.

    Filter cells that are below or above a multiple of the standard deviation.
    Any variable can be used.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    var : str
        Variable to use for filtering
    std_thresh : int or float
        Threshold to use for outlier identification.
        x-fold of standard deviation on both or a single side.
    side : str
        `left`, `right` or `both`

    Returns
    -------
    anndata.AnnData
        Filtered AnnData object

    Raises
    -------
    ValueError
        If side is neither `left`, `right` nor `both`.

    Examples
    --------
    >>> import morphelia as mp
    >>> import anndata as ad
    >>> import numpy as np

    >>> data = np.random.rand(5, 5)
    >>> adata = ad.AnnData(data)
    >>> mp.pp.filter_std(adata, var='0', std_thresh=2)
    AnnData object with n_obs × n_vars = 3 × 5
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


def filter_thresh(
    adata: ad.AnnData, var: str, thresh: Union[float, int], side: str = "right"
):
    """Filter cells by thresholding.

    Parameters
    ----------
    adata : anndata.AnnData)
        Multidimensional morphological data
    var : str
        Variable in .var to use for filtering
    thresh : int or float
        Threshold value
    side : str
        `right`: Drop values above threshold
        `left`: Drop values below threshold

    Returns
    -------
    anndata.AnnData
        Filtered AnnData object

    Raises
    -------
    AssertionError
        If `side` is neither `right` nor `left`
    KeyError
        If `var` is not in .var_names

    Examples
    --------
    >>> import morphelia as mp
    >>> import anndata as ad
    >>> import numpy as np

    >>> data = np.random.rand(5, 5)
    >>> adata = ad.AnnData(data)
    >>> mp.pp.filter_thresh(adata, var='0', thresh=2)
    AnnData object with n_obs × n_vars = 3 × 5
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


def drop_duplicates(
    adata: ad.AnnData, axis: Union[int, float] = 1, verbose: bool = False
):
    """Drop duplicated features.

    Drops all feature vectors or cells that are duplicates.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    axis : int or float
        Drop duplicated features (1) or cells (0)
    verbose : bool

    Returns
    -------
    anndata.AnnData
        Filtered AnnData object
        .uns['duplicated_feats']: Dropped duplicated features

    Raises
    -------
    AssertionError
        If `axis` is neither `0` nor `1`

    Examples
    --------
    >>> import morphelia as mp
    >>> import anndata as ad
    >>> import numpy as np

    >>> data = np.random.rand(5, 5)
    >>> data[:, 4] = data[:, 0]  # last column is a duplicate of the first column
    >>> adata = ad.AnnData(data)
    >>> mp.pp.drop_duplicates(adata)
    AnnData object with n_obs × n_vars = 5 × 4
        uns: 'duplicated_feats'
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


def drop_invariant(adata: ad.AnnData, axis: int = 0, verbose: bool = False):
    """Drop invariant features.

    Drops rows (cells) or columns (features)
    if all values are equal.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    axis : int
        `0` for columns, `1` for rows
    verbose : bool

    Returns
    -------
    anndata.AnnData
        Filtered AnnData object
        .uns['invariant_feats']: Dropped invariant features

    Raises
    -------
    AssertionError
        If `axis` is neither `0` nor `1`

    Examples
    --------
    >>> import morphelia as mp
    >>> import anndata as ad
    >>> import numpy as np

    >>> data = np.random.rand(5, 5)
    >>> data[:, 4] = 0  # last column is invariant
    >>> adata = ad.AnnData(data)
    >>> mp.pp.drop_invariant(adata)
    AnnData object with n_obs × n_vars = 5 × 4
        uns: 'invariant_feats'
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
