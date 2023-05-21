from typing import Union, Optional

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import anndata as ad

from morphelia.tools import RobustMAD, choose_layer
from morphelia.preprocessing import drop_nan as drop_nan_feats


def normalize(
    adata: ad.AnnData,
    by: Optional[Union[str, list, tuple]] = ("BatchNumber", "PlateNumber"),
    method: str = "standard",
    pop_var: str = "Metadata_Treatment",
    norm_pop: Optional[str] = None,
    drop_nan: bool = False,
    obsm: Optional[str] = None,
    verbose: bool = False,
    **kwargs,
):
    """Feature normalization.

    Normalizes features of an experiment with one ore more batches and
    one or more plates.
    Several methods are available for normalization such as standard scaling,
    robust scaling, robust MAD scaling and min-max scaling.
    If a normalization population is given, scaling statistics are calculated only
    in this population, i.g. negative controls.

    Parameters
    ----------
    adata : anndata.AnnData)
        Multidimensional morphological data
    by : str or tuple or list
        Groups to apply function to.
        If None, apply to whole anndata.AnnData object
    method : str
        One of the following method to use for scaling:
        `standard`: removing the mean and scaling to unit variance
        `robust`: removing the median and scaling according to the IQR (interquartile range)
        `mad_robust`: removing the median and scaling to MAD (meand absolute deviation)
        `min_max`: scaling features to given range (typically 0 to 1)
    pop_var : str
        Population variable
    norm_pop : str, optional
        Normalization population to use to calculate statistics
        This is not used if norm_pop is None
    drop_nan : bool
        Drop feature containing nan values after transformation
    obsm : str, optional
        If provided, which element of obsm to scale
    verbose : bool
    **kwargs
        Arguments passed to scaler

    Returns
    -------
    anndata.AnnData
        Normalized AnnData object

    Raises
    -------
    AssertionError
        If any variable in `by` is not in `.var`
    AssertionError
        If `pop_var` is not in `.obs.columns`
    AssertionError
        If method is not valid

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np
    >>> import pandas as pd

    >>> data = np.random.rand(5, 5)
    >>> obs = pd.DataFrame({'group': [0, 0, 1, 1, 1]})
    >>> adata = ad.AnnData(data, obs=obs)

    >>> mp.pp.normalize(adata, by='group')
    AnnData object with n_obs × n_vars = 5 × 5
        obs: 'group'
        uns: 'nan_feats'
    """
    # check that variables in by are in anndata
    if by is not None:
        if isinstance(by, str):
            by = [by]
        elif isinstance(by, tuple):
            by = list(by)

        assert all(
            var in adata.obs.columns for var in by
        ), f"Variables defined in 'by' are not in annotations: {by}"

    if norm_pop is not None:
        assert (
            pop_var in adata.obs.columns
        ), f"Population variable not found in annotations: {pop_var}"

    # define scaler
    method = method.lower()

    avail_methods = ["standard", "robust", "mad_robust", "min_max"]
    assert method in avail_methods, (
        f"Method must be one of {avail_methods}, " f"instead got {method}"
    )

    scaler_method = None
    if method == "standard":
        scaler_method = StandardScaler
    elif method == "robust":
        scaler_method = RobustScaler
    elif method == "mad_robust":
        scaler_method = RobustMAD
    elif method == "min_max":
        scaler_method = MinMaxScaler

    # iterate over adata with grouping variables
    if by is not None:
        x = choose_layer(adata, obsm=obsm, copy=True)
        for groups, sub_df in adata.obs.groupby(by):
            scaler = scaler_method(**kwargs)
            # cache indices of group
            group_ix = sub_df.index
            mask = adata.obs.index.isin(group_ix)
            # transform group with scaler
            if norm_pop is not None:
                norm_ix = sub_df[sub_df[pop_var] == norm_pop].index
                norm_pop_mask = adata.obs.index.isin(norm_ix)
                scaler.fit(x[norm_pop_mask, :])
            else:
                scaler.fit(x[mask, :])

            # transform
            x[mask, :] = scaler.transform(x[mask, :])
        if obsm is not None:
            adata.obsm[obsm] = x
        else:
            adata.X = x

    else:
        scaler = scaler_method(**kwargs)
        if obsm is None:
            scaler.fit(adata.X)
            adata.X = scaler.transform(adata.X)
        else:
            scaler.fit(choose_layer(adata, obsm=obsm, copy=True))
            adata.obsm[obsm] = scaler.transform(
                choose_layer(adata, obsm=obsm, copy=True)
            )

    if drop_nan:
        adata = drop_nan_feats(adata, verbose=verbose, obsm=obsm, axis=0)

    return adata
