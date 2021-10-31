# import internal libraries
from collections import defaultdict

# import external libraries
import numpy as np
import pandas as pd
import anndata as ad


def aggregate(adata,
              by=("BatchNumber", "PlateNumber", "Metadata_Well"),
              method='median',
              keep_obs=None,
              count=True,
              qc=False,
              min_cells=300,
              verbose=False,
              **kwargs):
    """Aggregate multidimensional morphological data by populations.

    Args:
        adata (anndata.AnnData): Annotated data object.
        by (list of str): Variables to use for aggregation.
        method (str): Method of aggregation.
            Should be one of: Mean, median, modz.
        keep_obs (list of str): Identifiers for observations to keep.
            Keep all if None.
        count (bool): Add population count to observations if True.
        qc (bool): True for quality control.
        min_cells (int): Minimum number of cells per population.
            Population is deleted from data if below threshold.
        verbose (bool)
        **kwargs: Keyword arguments passed to methods.

    Returns:
        anndata.AnnData
    """
    # check that variables in by are in anndata
    if isinstance(by, str):
        by = [by]
    else:
        by = list(by)
    if not all(var in adata.obs.columns for var in by):
        raise KeyError(f"Variables defined in 'by' are not in annotations: {by}")

    # delete observations not needed for aggregation
    if keep_obs is not None:
        if isinstance(keep_obs, str):
            keep_obs = [keep_obs]
        if isinstance(keep_obs, list):
            drop_obs = [obs for obs in adata.obs.columns if not any(identifier in obs for identifier in keep_obs)]
            for elem in by:
                if elem in drop_obs:
                    drop_obs.remove(elem)
            adata.obs.drop(drop_obs, axis=1, inplace=True)
        else:
            raise TypeError(f"obs_ids is expected to be string or list, instead got {type(keep_obs)}")

    # check if cellnumber is already in adata
    cn_var = 'Metadata_Cellnumber'
    if cn_var in adata.obs.columns:
        count = False

    # check method
    avail_methods = ['mean', 'median', 'modz']
    method = method.lower()
    assert method in avail_methods, f'method not supported, choose one of {avail_methods}'

    # store aggregated data
    X_agg = []
    obs_agg = defaultdict(list)

    # iterate over adata with grouping variables
    for groups, sub_df in adata.obs.groupby(list(by)):
        # cache annotations from first element
        for key, val in sub_df.iloc[0, :].to_dict().items():
            if key != cn_var:
                obs_agg[key].append(val)
            else:
                # sum counts from preaggregated data
                sum_count = sub_df[cn_var].sum()
                obs_agg[key].append(sum_count)
        # add object number to observations
        if count:
            obs_agg[cn_var].append(len(sub_df))

        # cache indices of group
        group_ix = sub_df.index

        # aggregate group
        if method == 'mean':
            agg = np.nanmean(adata[group_ix, :].X.copy(), axis=0, **kwargs).reshape(1, -1)
        elif method == 'median':
            agg = np.nanmedian(adata[group_ix, :].X.copy(), axis=0, **kwargs).reshape(1, -1)
        elif method =='modz':
            agg = modz(adata[group_ix, :], **kwargs).reshape(1, -1)

        # concatenate aggregated groups
        X_agg.append(agg)

    # make anndata object from aggregated data
    X_agg = np.concatenate(X_agg, axis=0)
    obs_agg = pd.DataFrame(obs_agg)

    adata = ad.AnnData(X=X_agg, obs=obs_agg, var=adata.var)

    # quality control
    if qc:
        if min_cells is not None:
            if verbose:
                dropped_pops = adata[adata.obs[cn_var] < min_cells, :].obs[by].values.tolist()
                print(f"Dropped populations: {dropped_pops}")
            adata = adata[adata.obs[cn_var] >= min_cells, :]

    return adata


def modz(adata,
         method='spearman',
         min_weight=0.01,
         precision=4):
    """Performs a modified z score transformation.
    This code is modified from pycytominer:
    https://github.com/cytomining/pycytominer/blob/master/pycytominer/cyto_utils/modz.py

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        method (str): Correlation method. One of pearson, spearman or kendall.
        min_weight (float): Minimum correlation to clip all non-negative values lower to.
        precision (int): Number of digits to round weights to.

    Returns:
        numpy.array: Modz transformed aggregated data.
    """
    # check variables
    assert adata.shape[0] > 0, "AnnData object must include at least one sample"

    avail_methods = ["pearson", "spearman", "kendall"]
    method = method.lower()
    assert method in avail_methods, f"method must be one of {avail_methods}, " \
                                    f"instead got {method}"

    # adata to pandas dataframe
    adata = adata.to_df()

    # Step 1: Extract pairwise correlations of samples
    # Transpose so samples are columns
    adata = adata.transpose()
    corr_df = adata.corr(method=method)

    # Step 2: Identify sample weights
    # Fill diagonal of correlation_matrix with np.nan
    np.fill_diagonal(corr_df.values, np.nan)

    # Remove negative values
    corr_df = corr_df.clip(lower=0)

    # Get average correlation for each profile (will ignore NaN)
    raw_weights = corr_df.mean(axis=1)

    # Threshold weights (any value < min_weight will become min_weight)
    raw_weights = raw_weights.clip(lower=min_weight)

    # normalize raw_weights so that they add to 1
    weights = raw_weights / sum(raw_weights)
    weights = weights.round(precision)

    # Step 3: Normalize
    if adata.shape[1] == 1:
        # There is only one sample (note that columns are now samples)
        modz_df = adata.sum(axis=1)
    else:
        modz_df = adata * weights
        modz_df = modz_df.sum(axis=1)

    # convert series back to array
    modz = modz_df.to_numpy()

    return modz
