# import internal libraries
from collections import defaultdict
import logging
from typing import Union, Optional, List

# import external libraries
import numpy as np
from scipy import stats
import pandas as pd
import anndata as ad

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def aggregate(
    adata: ad.AnnData,
    by: Union[tuple, list, str] = ("BatchNumber", "PlateNumber", "Metadata_Well"),
    method: str = "median",
    keep_obs: Optional[Union[List[str], str]] = None,
    count: bool = False,
    aggregate_reps: bool = False,
    qc: bool = False,
    drop_qc: bool = False,
    min_cells: int = 300,
    verbose: bool = False,
    **kwargs,
):
    """Aggregate multidimensional morphological data by populations.

    This function aggregated groups of cells (e.g. per well) with a given method.
    Quality control can be used to filter aggregated groups based on the number of cells per group.
    Thereby, wells with low cell count can be dropped from the data.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object
    by : list, tuple, str
        Variables to aggregate by
    method : str
        Method of aggregation.
        Should be one of: `mean`, `median`, `modz`.
    keep_obs : list of str or str, optional
        Identifiers for observations to keep, keep all if None
    count : bool
        Add population count to observations if True
    aggregate_reps : bool
        Aggregate representations in `.obsm` similar to `adata.X`
    qc : bool
        Quality control based on cell counts in the aggregated populations.
        If True, returns aggregated anndata object and dropped populations.
    drop_qc : bool
        Drop groups after quality control
    min_cells : int
        Minimum number of cells per population.
        Population is deleted from data if below threshold.
    verbose : bool
    **kwargs
        Keyword arguments passed to methods.

    Returns
    -------
    anndata.AnnData
        Aggregated AnnData object

    Raises
    ______
    AssertionError
        If variables in `by` are not in .var
    TypeError
        If `keep_obs` is neither of type list nor string
    AssertionError
        If `method` is neither `mean`, `median` nor `modz`

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np
    >>> import pandas as pd

    >>> data = np.random.rand(5, 5)
    >>> obs = pd.DataFrame({'group': [0, 0, 1, 1, 1]})
    >>> adata = ad.AnnData(data, obs=obs)

    >>> mp.pp.aggregate(adata, by='group')
    AnnData object with n_obs × n_vars = 2 × 5
        obs: 'group', 'Metadata_Cellnumber'
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
                f"`keep_obs` is expected to be string or list, instead got {type(keep_obs)}"
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
        agg = None

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
            agg = _modz(adata[group_ix, :].X.copy(), **kwargs).reshape(1, -1)
            if aggregate_reps:
                for rep in adata.obsm.keys():
                    agg_rep = _modz(
                        adata[group_ix, :].obsm[rep].copy(), **kwargs
                    ).reshape(1, -1)
                    X_reps[rep].append(agg_rep)

        # concatenate aggregated groups
        if agg is not None:
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


def _modz(
    X: np.ndarray, method: str = "spearman", min_weight: Union[float, int] = 0.01
) -> np.ndarray:
    """Modified z-score transformation.

    Modified z-scores (normally calculated with median and median absolute deviation) are aggregated by
    weighting each sample by their mean correlation to all other samples. This procedure is modified from cmapPy.

    Parameters
    ----------
    X : np.array
        Data representation
    method : str
        Correlation method. Either `pearson` or `spearman`.
    min_weight : float or int
        Minimum correlation to clip all non-negative values lower to

    Returns
    -------
    numpy.array
        Modz transformed aggregated data

    Raises
    ______
    AssertionError
        If array is empty
    AssertionError
        If method is neither `pearson` nor `spearman`

    References
    __________
    .. [1] https://github.com/cmap/cmapPy/blob/master/cmapPy/math/agg_wt_avg.py
    """
    # check variables
    assert X.shape[0] > 0, "array object must include at least one sample"
    # No aggregation if length of array is 1
    if X.shape[0] == 1:
        return X

    method_dict = {
        "pearson": lambda x: np.corrcoef(x),
        "spearman": lambda x: stats.spearmanr(x.T)[0],
    }
    method = method.lower()
    assert method in list(
        method_dict.keys()
    ), f"method must be one of {list(method_dict.keys())}, instead got {method}"
    corr_func = method_dict[method]

    # extract pairwise correlations of samples
    R = corr_func(X)

    # for constant arrays the mean is calculated
    if isinstance(R, float):
        R = np.ones((X.shape[0], X.shape[0]))

    # fill diagonal of correlation matrix with np.nan
    np.fill_diagonal(R, np.nan)

    # remove negative values
    R = R.clip(min=0)

    # get average correlation for each profile (will ignore NaN)
    raw_weights = np.nanmean(R, axis=1)

    # threshold weights (any value < min_weight will become min_weight)
    raw_weights = raw_weights.clip(min=min_weight)

    # normalize raw_weights so that they add to 1
    weights = raw_weights / sum(raw_weights)

    # apply weights to values
    X = np.sum(X * weights.reshape(-1, 1), axis=0).reshape(1, -1)

    return X


def aggregate_chunks(
    adata: ad.AnnData,
    by: Union[tuple, list, str],
    chunk_size: int = 25,
    n_chunks: Optional[Union[int, str]] = None,
    method: str = "median",
    seed: int = 0,
    **kwargs,
):
    """Aggregate chunked data.

    This function aggregates data into random chunks
    within the same group defined with 'by'.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object
    by : list or str or tuple of str or str
        Variables to aggregate by
    chunk_size : int
        Size of chunks
    n_chunks : int, optional
        Number of chunks. If None, as many chunks as possible are aggregated.
        If 'equal' all groups have equal size.
    method : str
        Method of aggregation.
        Should be one of: `Mean`, `median` or `modz`.
    seed : int
        Seed random data selection
    **kwargs
        Keyword arguments passed to aggregate

    Returns
    -------
    anndata.AnnData
        Aggregated AnnData object

    Raises
    ------
    KeyError
        If variables in `by` are not in .var_names

    Examples
    --------
    In this example 10 chunks with 3 samples each are repeatedly drawn and aggregated
    with replacement.

    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np
    >>> import pandas as pd

    >>> data = np.random.rand(5, 5)
    >>> obs = pd.DataFrame({'group': [0, 0, 1, 1, 1]})
    >>> adata = ad.AnnData(data, obs=obs)

    >>> mp.pp.aggregate_chunks(adata, by='group', chunk_size=3, n_chunks=10, with_replacement=True)
    AnnData object with n_obs × n_vars = 10 × 5
        obs: 'group', 'batch'
    """
    adata.obs_names_make_unique()
    # check that variables in by are in anndata
    if isinstance(by, str):
        by = [by]
    else:
        by = list(by)
    if not all(var in adata.obs.columns for var in by):
        raise KeyError(f"Variables defined in 'by' are not in annotations: {by}")

    avail_methods = ["equal"]
    if isinstance(n_chunks, str):
        assert (
            n_chunks in avail_methods
        ), f"`n_chunks` must be one of {avail_methods}, instead got {n_chunks}"
        if n_chunks == "equal":
            n_chunks = np.min(adata.obs[by].value_counts() // chunk_size)

    rng = np.random.default_rng(seed)

    # Get group indices as list of lists
    group_ixs = adata.obs.groupby(by).apply(lambda x: list(x.index)).to_list()

    # Get sample of every group
    if n_chunks is None:
        sample_ixs = np.concatenate(
            [
                rng.choice(
                    ixs, size=(len(ixs) // chunk_size) * chunk_size, replace=False
                )
                for ixs in group_ixs
            ]
        ).flatten()
    else:
        sample_ixs = np.concatenate(
            [
                rng.choice(ixs, size=n_chunks * chunk_size, replace=False)
                for ixs in group_ixs
            ]
        ).flatten()

    # Chunk names
    n_all_chunks = int(len(sample_ixs) / chunk_size)
    chunks = np.repeat(np.arange(n_all_chunks), chunk_size)

    # Add chunk names to adata
    adata.obs.loc[:, "Chunk"] = -1
    adata.obs.loc[sample_ixs, "Chunk"] = chunks
    adata = adata[adata.obs["Chunk"] != -1, :]

    agg_adata = aggregate(
        adata,
        by="Chunk",
        method=method,
        verbose=False,
        **kwargs,
    )

    return agg_adata
