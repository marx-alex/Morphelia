import matplotlib.pyplot as plt
import numpy as np
import anndata as ad
from morphelia.plotting import plot_corr_matrix
from morphelia.tools.utils import get_subsample

import warnings
import logging
from typing import Optional, Union, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def drop_highly_correlated(
    adata: ad.AnnData,
    thresh: float = 0.95,
    subsample: bool = False,
    sample_size: int = 1000,
    seed: int = 0,
    verbose: bool = False,
    neg_corr: bool = False,
    drop: bool = True,
    make_plot: bool = True,
    show: bool = True,
    save: Optional[bool] = None,
    **kwargs,
) -> Union[ad.AnnData, Tuple[ad.AnnData, plt.Figure]]:
    """Drop highly correlated features

    This function calculates correlation coefficients
    between all features. Highly correlated features with a coefficient above
    are certain threshold are dropped (only the first feature is kept).

    The following information is stored:
        .uns['highly_correlated']: Highly correlated features

        .uns['nan_feats']: Features with nan values

        .var['highly_correlated']: True for highly correlated features
            Only if drop is False

        .var['contains_nan']: True for features that contain nan values
            Only if drop is False

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    thresh : float
        Correlated features with Pearson correlation coefficient
        above threshold are dropped
    save : str, optional
        Path where to save figure
    subsample : bool
        If True, fit models on subsample of data
    sample_size : int
        Size of supsample. Only if subsample is True.
    seed : int
        Seed for subsample calculation. Only if subsample is True.
    verbose : bool
    neg_corr : bool
        Drop negative correlated features
    drop : bool
        Drop features. If false add information to `.var`
    make_plot : bool
        Make plot to test results
    show : bool
        Show figure if True, else return AnnData object and figure
    **kwargs
        Keyword arguments for `seaborn.clustermap`

    Returns
    -------
    anndata.Anndata
        AnnData object without dropped features and information stored at `.uns` or `.var`

    Raises
    -------
    TypeError
        If `neg_corr` is not of type bool

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np

    >>> data = np.random.rand(6, 5)
    >>> adata = ad.AnnData(data)
    >>> mp.ft.drop_highly_correlated(adata)
    AnnData object with n_obs × n_vars = 6 × 5
        uns: 'highly_correlated'
    """

    # get subsample
    if subsample:
        adata_ss = get_subsample(adata, sample_size=sample_size, seed=seed)
    else:
        adata_ss = adata.copy()

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
        raise TypeError(
            f"neg_corr expected to be a boolean, instead got {type(neg_corr)}"
        )

    if len(drop_ix) > 0:
        drop_ix = list(set(drop_ix[:, 1].tolist()))
    else:
        warnings.warn(f"No highly correlated features found with threshold: {thresh}.")
        drop_ix = None

    # drop highly correlated features
    if drop_ix is not None:
        all_vars = adata.var_names
        drop_vars = all_vars[drop_ix]

        if verbose:
            logger.info(f"Dropped {len(drop_vars)} features: {drop_vars}")
        keep_vars = [var for var in all_vars if var not in drop_vars]

        if drop:
            adata = adata[:, keep_vars].copy()
            # store info in .uns
            adata.uns["highly_correlated"] = drop_vars
        else:
            mask = [True if var in drop_vars else False for var in all_vars]
            adata.var["highly_correlated"] = mask
    else:
        if drop:
            adata.uns["highly_correlated"] = []
        else:
            mask = [False for var in adata.var_names]
            adata.var["highly_correlated"] = mask

    # drop nan features
    if len(nan_feats) > 0:
        non_nan_feats = [feat for feat in adata.var_names if feat not in nan_feats]

        if verbose:
            logger.info(f"Dropped uniform features: {nan_feats}")

        if drop:
            adata = adata[:, non_nan_feats].copy()

            if "nan_feats" not in adata.uns:
                adata.uns["nan_feats"] = nan_feats
            else:
                adata.uns[nan_feats].append(nan_feats)

        else:
            mask = [True if feat in nan_feats else False for feat in adata.var_names]
            adata.var["contains_nan"] = mask
    else:
        if not drop:
            mask = [False for var in adata.var_names]
            adata.var["contains_nan"] = mask

    if make_plot:
        if drop_ix is not None:
            groups = [
                "other features" if var in keep_vars else "higly correlated features"
                for var in all_vars
            ]
            if len(nan_feats) > 0:
                groups = [
                    "invariant features" if var in nan_feats else label
                    for label, var in zip(groups, all_vars)
                ]
        else:
            groups = None

        # plot
        _ = plot_corr_matrix(corr_matrix, groups, save=save, show=show, **kwargs)

        if not show:
            fig, ax = _
            return adata, fig

    return adata
