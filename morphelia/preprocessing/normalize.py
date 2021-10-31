import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from morphelia.tools import RobustMAD
from morphelia.preprocessing import drop_nan as drop_nan_feats


def normalize(adata,
              by=("BatchNumber", "PlateNumber"),
              method="standard",
              pop_var="Metadata_Treatment",
              norm_pop=None,
              clip=None,
              drop_nan=True,
              verbose=False,
              **kwargs):
    """
    Normalizes features of an experiment with one ore more batches and
    one or more plates.
    Several methods are available for normalization such as standard scaling,
    robust scaling, robust MAD scaling and min-max scaling.
    If a normalization population is given scaling statistics are calculated only
    in this population, i.g. negative controls.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        by (iterable, str or None): Groups to apply function to.
            If None, apply to whole anndata.AnnData object.
        method (str): One of the following method to use for scaling:
            standard: removing the mean and scaling to unit variance.
            robust: removing the median and scaling according to the IQR (interquartile range).
            mad_robust: removing the median and scaling to MAD (meand absolute deviation).
            min_max: scaling features to given range (typically 0 to 1).
        pop_var (str): Variable that denotes populations.
        norm_pop (str): Population to use for calculation of statistics.
            This is not used if norm_pop is None.
        drop_nan (bool): Drop feature containing nan values after transformation.
        clip (int): Clip (truncate) to this value after scaling. If None, do not clip.
        verbose (bool)
        ** kwargs: Arguments passed to scaler.
    """
    # check that variables in by are in anndata
    if by:
        if isinstance(by, str):
            by = [by]
        elif isinstance(by, tuple):
            by = list(by)

        if not all(var in adata.obs.columns for var in by):
            raise KeyError(f"Variables defined in 'by' are not in annotations: {by}")

    if norm_pop is not None:
        if pop_var not in adata.obs.columns:
            raise KeyError(f"Population variable not found in annotations: {pop_var}")

    # define scaler
    method = method.lower()

    avail_methods = ['standard', 'robust', 'mad_robust', 'min_max']
    assert method in avail_methods, f"Method must be one of {avail_methods}, " \
                                    f"instead got {method}"

    if method == "standard":
        scaler = StandardScaler(**kwargs)
    elif method == "robust":
        scaler = RobustScaler(**kwargs)
    elif method == "mad_robust":
        scaler = RobustMAD()
    elif method == "min_max":
        scaler = MinMaxScaler(**kwargs)

    # iterate over adata with grouping variables
    if by is not None:
        for groups, sub_df in adata.obs.groupby(by):
            # cache indices of group
            group_ix = sub_df.index
            # transform group with scaler
            if norm_pop is not None:
                norm_ix = sub_df[sub_df[pop_var] == norm_pop].index
                scaler.fit(adata[norm_ix, :].X.copy())
            else:
                scaler.fit(adata[group_ix, :].X.copy())
            # transform
            adata[group_ix, :].X = scaler.transform(adata[group_ix, :].X.copy())

    else:
        scaler.fit(adata.X.copy())
        adata.X = scaler.transform(adata.X.copy())

    if clip is not None:
        assert (clip > 0), f'Value for clip should be above 0, instead got {clip}'
        adata.X[adata.X > clip] = clip
        adata.X[adata.X < -clip] = clip

    if drop_nan:
        adata = drop_nan_feats(adata, verbose=verbose)

    return adata
