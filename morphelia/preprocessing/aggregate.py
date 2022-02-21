# import internal libraries
from collections import defaultdict
import logging

# import external libraries
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def aggregate(
    adata,
    by=("BatchNumber", "PlateNumber", "Metadata_Well"),
    method="median",
    keep_obs=None,
    count=True,
    aggregate_reps=True,
    qc=False,
    drop_qc=False,
    min_cells=300,
    verbose=False,
    **kwargs,
):
    """Aggregate multidimensional morphological data by populations.

    Args:
        adata (anndata.AnnData): Annotated data object.
        by (list, str): Variables to use for aggregation.
        method (str): Method of aggregation.
            Should be one of: Mean, median, modz.
        keep_obs (list of str): Identifiers for observations to keep.
            Keep all if None.
        count (bool): Add population count to observations if True.
        aggregate_reps (bool): Aggregate representations similar to adata.X
        qc (bool): Quality control based on cell counts in the aggregated populations.
            If True, returns aggregated anndata object and dropped populations.
        drop_qc (bool): Drop wells after quality control.
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
    assert all(
        var in adata.obs.columns for var in by
    ), f"Variables defined in 'by' are not in annotations: {by}"

    # delete observations not needed for aggregation
    if keep_obs is not None:
        if isinstance(keep_obs, str):
            keep_obs = [keep_obs]
        if isinstance(keep_obs, list):
            drop_obs = [
                obs
                for obs in adata.obs.columns
                if not any(identifier in obs for identifier in keep_obs)
            ]
            for elem in by:
                if elem in drop_obs:
                    drop_obs.remove(elem)
            adata.obs.drop(drop_obs, axis=1, inplace=True)
        else:
            raise TypeError(
                f"obs_ids is expected to be string or list, instead got {type(keep_obs)}"
            )

    # check if cellnumber is already in adata
    cn_var = "Metadata_Cellnumber"
    if cn_var in adata.obs.columns:
        count = False

    # check method
    avail_methods = ["mean", "median", "modz"]
    method = method.lower()
    assert (
        method in avail_methods
    ), f"method not supported, choose one of {avail_methods}"

    # store aggregated data
    X_agg = []
    obs_agg = defaultdict(list)
    X_reps = defaultdict(list)

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

        if method == "mean":
            agg = np.nanmean(adata[group_ix, :].X.copy(), axis=0, **kwargs).reshape(
                1, -1
            )
            if aggregate_reps:
                for rep in adata.obsm.keys():
                    agg_rep = np.nanmean(
                        adata[group_ix, :].obsm[rep].copy(), axis=0, **kwargs
                    ).reshape(1, -1)
                    X_reps[rep].append(agg_rep)
        elif method == "median":
            agg = np.nanmedian(adata[group_ix, :].X.copy(), axis=0, **kwargs).reshape(
                1, -1
            )
            if aggregate_reps:
                for rep in adata.obsm.keys():
                    agg_rep = np.nanmedian(
                        adata[group_ix, :].obsm[rep].copy(), axis=0, **kwargs
                    ).reshape(1, -1)
                    X_reps[rep].append(agg_rep)
        elif method == "modz":
            agg = modz(adata[group_ix, :].X.copy(), **kwargs).reshape(1, -1)
            if aggregate_reps:
                for rep in adata.obsm.keys():
                    agg_rep = modz(
                        adata[group_ix, :].obsm[rep].copy(), **kwargs
                    ).reshape(1, -1)
                    X_reps[rep].append(agg_rep)

        # concatenate aggregated groups
        X_agg.append(agg)

    # make anndata object from aggregated data
    X_agg = np.concatenate(X_agg, axis=0)
    obs_agg = pd.DataFrame(obs_agg)

    # concatenate chunks of representations
    if len(list(X_reps.keys())) > 0:
        for _key, _val in X_reps.items():
            if len(_val) > 1:
                _val = np.vstack(_val)
            else:
                _val = _val[0]
            X_reps[_key] = _val

        adata = ad.AnnData(X=X_agg, obs=obs_agg, var=adata.var, obsm=X_reps)
    else:
        adata = ad.AnnData(X=X_agg, obs=obs_agg, var=adata.var)

    # quality control
    if qc:
        dropped_pops = None
        if min_cells is not None:
            dropped_pops = adata.obs.loc[adata.obs[cn_var] < min_cells, by].values
            if verbose:
                logger.info(f"Dropped populations: {dropped_pops}")
            if drop_qc:
                adata = adata[adata.obs[cn_var] >= min_cells, :]
        return adata, dropped_pops

    return adata


def modz(arr, method="spearman", min_weight=0.01, precision=4):
    """Performs a modified z score transformation.
    This code is modified from pycytominer:
    https://github.com/cytomining/pycytominer/blob/master/pycytominer/cyto_utils/modz.py

    Args:
        arr (np.array): Representation of data.
        method (str): Correlation method. One of pearson, spearman or kendall.
        min_weight (float): Minimum correlation to clip all non-negative values lower to.
        precision (int): Number of digits to round weights to.

    Returns:
        numpy.array: Modz transformed aggregated data.
    """
    # check variables
    assert arr.shape[0] > 0, "array object must include at least one sample"

    avail_methods = ["pearson", "spearman", "kendall"]
    method = method.lower()
    assert method in avail_methods, (
        f"method must be one of {avail_methods}, " f"instead got {method}"
    )

    # adata to pandas dataframe
    arr = pd.DataFrame(data=arr)

    # Step 1: Extract pairwise correlations of samples
    # Transpose so samples are columns
    arr = arr.transpose()
    corr_df = arr.corr(method=method)

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
    if arr.shape[1] == 1:
        # There is only one sample (note that columns are now samples)
        modz_df = arr.sum(axis=1)
    else:
        modz_df = arr * weights
        modz_df = modz_df.sum(axis=1)

    # convert series back to array
    modz = modz_df.to_numpy()

    return modz


def aggregate_chunks(
    adata,
    by=("BatchNumber", "PlateNumber", "Metadata_Well"),
    chunk_size=25,
    with_replacement=False,
    n_chunks=500,
    method="median",
    keep_obs=None,
    count=False,
    aggregate_reps=False,
    seed=0,
    **kwargs,
):
    """Aggregate data into random chunks within the same condition defined with 'by'.

    Args:
        adata (anndata.AnnData): Annotated data object.
        by (list of str): Variables to use for aggregation.
        chunk_size (int): Size of chunks.
        with_replacement (bool): Draw random chunks with replacement.
        n_chunks (int): If with_replacement is True, this many chunks are aggregated per condition.
        method (str): Method of aggregation.
            Should be one of: Mean, median, modz.
        keep_obs (list of str): Identifiers for observations to keep.
            Keep all if None.
        count (bool): Add population count to observations if True.
        aggregate_reps (bool): Aggregate representations similar to adata.X
        seed (int): Seed random data selection.
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

    np.random.seed(seed)

    adata_agg = []

    for groups, sub_df in tqdm(
        adata.obs.groupby(list(by)), desc="Aggregating chunks.."
    ):

        group_ix = sub_df.index
        avail_ix = list(group_ix)
        stop = True
        counter = 0

        while (len(avail_ix) >= chunk_size) and stop:
            choice = np.random.choice(avail_ix, chunk_size, replace=False)
            if not with_replacement:
                avail_ix = [ix for ix in avail_ix if ix not in choice]
            choice_adata = adata[choice, :].copy()
            choice_adata = aggregate(
                choice_adata,
                by=by,
                method=method,
                keep_obs=keep_obs,
                count=count,
                aggregate_reps=aggregate_reps,
                qc=False,
                verbose=False,
                **kwargs,
            )
            adata_agg.append(choice_adata)
            if (counter >= n_chunks) and with_replacement:
                stop = False
            counter += 1

    adata_agg = adata_agg[0].concatenate(*adata_agg[1:])

    return adata_agg
