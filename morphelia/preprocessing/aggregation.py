# import internal libraries
from collections import defaultdict
import logging
from typing import Union, Optional, List

# import external libraries
import numpy as np
from scipy import stats
import pandas as pd
import anndata as ad
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def aggregate(
    adata: ad.AnnData,
    by: Union[tuple, list, str] = ("BatchNumber", "PlateNumber", "Metadata_Well"),
    method: str = "median",
    keep_obs: Optional[Union[List[str], str]] = None,
    count: bool = True,
    aggregate_reps: bool = True,
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
    by: Union[tuple, list, str] = ("BatchNumber", "PlateNumber", "Metadata_Well"),
    chunk_size: int = 25,
    with_replacement: bool = False,
    n_chunks: int = 500,
    method: str = "median",
    keep_obs: Optional[Union[List[str], str]] = None,
    count: bool = False,
    aggregate_reps: bool = False,
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
    with_replacement : bool
        Draw random chunks from data with replacement
    n_chunks : int
        If with_replacement is True, this many chunks are aggregated per group
    method : str
        Method of aggregation.
        Should be one of: `Mean`, `median` or `modz`.
    keep_obs : list of str or str, optional
        Identifiers for observations to keep.
        Keep all if None.
    count : bool
        Add population count to observations if True
    aggregate_reps : bool
        Aggregate representations similar to adata.X
    seed : int
        Seed random data selection
    **kwargs
        Keyword arguments passed to methods

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
            if (counter >= n_chunks - 1) and with_replacement:
                stop = False
            counter += 1

    adata_agg = adata_agg[0].concatenate(*adata_agg[1:])

    return adata_agg
