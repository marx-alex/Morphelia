# import internal libraries
from collections import defaultdict

# import external libraries
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def aggregate(md, by=("BatchNumber", "PlateNumber", "Metadata_Well")):
    """Aggregate multidimensional morphological data by populations.

    Args:
        md (anndata.AnnData): Annotated data object.
        by (list of str): Variables to use for aggregation.

    Returns:
        anndata.AnnData
    """
    # check that variables in by are in anndata
    if not all(var in md.obs.columns for var in by):
        raise KeyError(f"Variables defined in 'by' are not in annotations: {by}")

    # store aggregated data
    X_agg = []
    obs_agg = defaultdict(list)

    # iterate over md with grouping variables
    for groups, sub_df in md.obs.groupby(list(by)):
        # cache annotations
        for key, val in sub_df.iloc[0, :].to_dict().items():
            obs_agg[key].append(val)
        # add object number to observations
        obs_agg['CellNumber'].append(len(sub_df))

        # cache indices of group
        group_ix = sub_df.index

        # aggregate group
        agg = np.mean(md[group_ix, :].X, axis=0).reshape(1, -1)
        # concatenate aggregated groups
        X_agg.append(agg)

    # make anndata object from aggregated data
    X_agg = np.concatenate(X_agg, axis=0)
    obs_agg = pd.DataFrame(obs_agg)

    return ad.AnnData(X=X_agg, obs=obs_agg, var=md.var)


def subsample(md, perc=0.1, by=("BatchNumber", "PlateNumber", "Metadata_Well"),
              seed=0):
    """Gives a subsample of the data by selecting objects from given groups.

    Args:
        md (anndata.AnnData): Annotated data object.
        perc (float): Percentage of objects to store in subsample.
        by (list of str): Variables to use for aggregation.
        seed (int): Seed for initialization.

    Returns:
        anndata.AnnData
    """
    # check that variables in by are in anndata
    if not all(var in md.obs.columns for var in by):
        raise KeyError(f"Variables defined in 'by' are not in annotations: {by}")
    if perc < 0 or perc > 1:
        raise ValueError(f"Use a float between 0 and 1 for perc: {perc}")

    # store subsample data
    X_ss = []
    obs_ss = defaultdict(list)

    # iterate over md with grouping variables
    for groups, sub_df in md.obs.groupby(list(by)):

        n = round(perc * len(sub_df))
        # cache indices subsample
        group_ix = sub_df.sample(n=n, random_state=seed).index

        if len(group_ix) > 0:
            # cache annotations
            for key, val in sub_df.loc[group_ix, :].to_dict('list').items():
                obs_ss[key].extend(val)

            # subsample group
            agg = md[group_ix, :].X

            # concatenate aggregated groups
            X_ss.append(agg)

    # make anndata object from aggregated data
    X_ss = np.concatenate(X_ss, axis=0)
    obs_ss = pd.DataFrame(obs_ss)

    return ad.AnnData(X=X_ss, obs=obs_ss, var=md.var)


def drop_nan(md, drop_inf=True, drop_dtype_max=True, verbose=False):
    """Drop variables that contain invalid variables.

    Args:
        md (anndata.AnnData): Multidimensional morphological data.
        drop_inf (bool): Drop also infinity values.
        drop_dtype_max (bool): Drop values to large for dtype.
        verbose (bool)
    """
    if drop_inf:
        md.X[(md.X == np.inf) | (md.X == -np.inf)] = np.nan

    if drop_dtype_max:
        md.X[md.X > np.finfo(md.X.dtype).max] = np.nan

    mask = ~np.isnan(md.X).any(axis=0)
    masked_vars = md.var[mask].index.tolist()

    if verbose:
        print(f"Dropped variables: {md.var[~mask].index.tolist()}")

    md = md[:, masked_vars].copy()

    return md


def min_max_scaler(md, min=0, max=1):
    """Wraper for sklearns MinMaxScaler.

    Args:
        md (anndata.AnnData): Multidimensional morphological data.
        min, max (int): Desired range of transformed data.
    """
    scaler = MinMaxScaler(feature_range=(min, max))
    md.X = scaler.fit_transform(md.X)

    return md


def filter_cells(md, var, std_thresh=1.96, side='both'):
    """Filter cells by identifying outliers from a distribution
    of a single value.

    Args:
        md (anndata.AnnData): Multidimensional morphological data.
        var (str): Variable to use for filtering.
        std_thresh (int): Threshold to use for outlier identification.
            x-fold of standard deviation in both or one directions.
        side (str): 'left', 'right' or 'both'
    """
    # get standard deviation and mean
    std = np.nanstd(md[:, var].X)
    mean = np.nanmean(md[:, var].X)

    # do filtering
    if side not in ['left', 'right', 'both']:
        raise ValueError(f"side should be 'left', 'right' or 'both': {side}")
    if side == 'both' or side == 'left':
        md = md[md[:, var].X > (mean - (std_thresh*std)), :]
    if side == 'both' or side == 'right':
        md = md[md[:, var].X < (mean + (std_thresh * std)), :]

    return md


def unique_feats(md, verbose=False):
    """Drops all feature vectors that are duplicates.

    Args:
        md (anndata.AnnData): Multidimensional morphological data.
        verbose (bool)
    """
    _, index = np.unique(md.X, axis=1, return_index=True)

    if verbose:
        print(f"Dropped features: {md.var.index[[ix for ix in range(md.X.shape[1]) if ix not in index]]}")

    # apply filter
    md = md[:, index]

    return md


def z_transform(md, by=("BatchNumber", "PlateNumber"), robust=False,
                clip=None, **kwargs):
    """Wrapper for sklearns StandardScaler to scale morphological data
    to unit variance by groups.

    Args:
        md (anndata.AnnData): Multidimensional morphological data.
        by (iterable, str or None): Groups to apply function to.
            If None, apply to whole anndata.AnnData object.
        robust (bool): If true, use sklearn.preprocessing.RobustScaler
        clip (int): Clip (truncate) to this value after scaling. If None, do not clip.
        ** kwargs: Arguments passed to scaler.
    """
    # check that variables in by are in anndata
    if not all(var in md.obs.columns for var in by):
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
    for groups, sub_df in md.obs.groupby(by):

        # cache indices of group
        group_ix = sub_df.index
        # transform group with scaler
        md[group_ix, :].X = scaler.fit_transform(md[group_ix, :].X)

    if clip is not None:
        assert (clip > 0), f'Value for clip should be above 0, instead got {clip}'
        md.X[md.X > clip] = clip
        md.X[md.X < -clip] = -clip

    return md
