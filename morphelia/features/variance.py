import numpy as np
from tqdm import tqdm
from scipy.stats import median_abs_deviation as mad
import logging

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def drop_near_zero_variance(
    adata, freq_thresh=0.05, unique_thresh=0.01, drop=True, verbose=False
):
    """Drop features that have low variance and therefore low expected information content.
    Low variance is assumed if single values appear more than one time in a feature vector.
    The rules are as following:
    1. A feature vector is of low variance if second_max_count / max_count < freq_thresh
    2. A feature vector is of low variance if num_unique_values / n_samples < unique_thresh
    with:
        second_max_count: counts of second frequent value
        max_count: counts of most frequent value
        freq_thresh: threshold for rule 1.
        num_unique_values: number of unique values in a feature vector
        n_samples: number of total samples
        unique_thresh: threshold for rule 2.

    This idea is modified from caret::nearZeroVar():
    https://www.rdocumentation.org/packages/caret/versions/6.0-88/topics/nearZeroVar

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        freq_thresh (float): Threshold for frequency.
        unique_thresh (float): Threshold for uniqueness.
        drop (bool): Drop features with low variance directly.
        verbose (bool)

    Returns:
        anndata.AnnData
        .uns['near_zero_variance_feats']: Dropped features with near zero variance.
        .var['near_zero_variance_feats']: True if feature has near zero variance.
            Only if drop is False.
    """
    # check variables
    assert 0 <= freq_thresh <= 1, (
        f"freq_thresh must be between 0 and 1, " f"instead got {freq_thresh}"
    )
    assert 0 <= unique_thresh <= 1, (
        f"unique_thresh must be between 0 and 1, " f"instead got {unique_thresh}"
    )

    # store dropped features
    drop_feats = []
    # get number of samples
    n_samples = adata.shape[0]

    # test first and second rule
    # iterate over features
    for feat in tqdm(adata.var_names, desc="Iterating over features"):
        # get unique features and their counts
        unique, counts = np.unique(adata[:, feat].X, return_counts=True)
        counts = np.sort(counts)

        if len(counts) > 1:
            # fist rule
            max_count = counts[-1]
            second_max_count = counts[-2]

            freq_ratio = second_max_count / max_count

            # apply freq_thresh
            if freq_ratio < freq_thresh:
                drop_feats.append(feat)

            # second rule
            unique_ratio = len(unique) / n_samples

            if unique_ratio < unique_thresh:
                drop_feats.append(feat)

    drop_feats = list(set(drop_feats))
    mask = [False if var in drop_feats else True for var in adata.var_names]

    if verbose:
        logger.info(f"Drop {len(drop_feats)} features with low variance: {drop_feats}")

    if drop:
        adata = adata[:, mask].copy()
        adata.uns["near_zero_variance_feats"] = drop_feats
    else:
        mask = [True if var in drop_feats else False for var in adata.var_names]
        adata.var["near_zero_variance_feats"] = mask

    return adata


def drop_low_cv(
    adata,
    by=("BatchNumber", "PlateNumber"),
    method="std",
    cutoff=0.5,
    drop=True,
    verbose=False,
):
    """Find features with low coefficients of variance which is interpreted as a low content of biological
    information.
    Depending on the method the normalized standard deviation or mean absolute deviation for every feature
    is calculated. If 'by' is given, the mean deviation of groups (batches or plates) is calculated.
    Features below a given threshold are dropped from the data.
    By default a coefficient of variance below 1 is considered to indicate low variance.

    The coefficients of variance are:
        std_norm = std / abs(mean)
        mad_norm = mad / abs(median)

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        by (iterable, str or None): Groups to apply function to.
                If None, apply to whole anndata.AnnData object.
        method (str): Standard deviation ('std') or mean absolute deviation ('mad').
        cutoff (float): Drop features with deviation below cutoff.
        drop (bool): Drop features with low variance directly.
        verbose (bool)

    Returns:
        anndata.AnnData
        .uns['low_cv_feats']: Dropped features with low coefficients of variance.
        .var['low_cv_feats']: True for features with low coefficients of variance.
            Only if drop is False.
    """
    # check variables
    if by is not None:
        if isinstance(by, str):
            by = [by]
        elif isinstance(by, tuple):
            by = list(by)

        if not all(var in adata.obs.columns for var in by):
            raise KeyError(f"Variables defined in 'by' are not in annotations: {by}")

    # check method
    method = method.lower()
    avail_methods = ["std", "mad"]
    assert method in avail_methods, (
        f"Method not in {avail_methods}, " f"instead got {method}"
    )

    assert isinstance(cutoff, (int, float)), (
        f"cutoff is expected to be type(float), " f"instead got {type(cutoff)}"
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        if by is not None:
            # store deviations
            norm_devs = []
            for groups, sub_df in adata.obs.groupby(by):
                # cache indices of group
                group_ix = sub_df.index

                if method == "std":
                    deviation = np.nanstd(adata[group_ix, :].X, axis=0)
                    norm_dev = deviation / np.abs(
                        np.nanmean(adata[group_ix, :].X, axis=0)
                    )
                elif method == "mad":
                    deviation = mad(
                        adata[group_ix, :].X, scale="normal", nan_policy="omit"
                    )
                    norm_dev = deviation / np.abs(
                        np.nanmedian(adata[group_ix, :].X, axis=0)
                    )

                norm_devs.append(norm_dev)

            norm_devs = np.stack(norm_devs)
            norm_dev = np.nanmean(norm_devs, axis=0)

        else:
            if method == "std":
                deviation = np.nanstd(adata.X, axis=0)
                norm_dev = deviation / np.abs(np.nanmean(adata.X, axis=0))
            elif method == "mad":
                deviation = mad(adata.X, scale="normal", nan_policy="omit")
                norm_dev = deviation / np.abs(np.nanmedian(adata.X, axis=0))

    # mask by cutoff
    mask = np.logical_and((norm_dev > cutoff), (norm_dev != np.nan))
    drop_feats = adata.var_names[~mask]
    if verbose:
        logger.info(
            f"Drop {len(drop_feats)} features with low coefficient of variance: {drop_feats}"
        )

    # drop
    if drop:
        adata = adata[:, mask].copy()
        adata.uns["low_cv_feats"] = drop_feats
    else:
        adata.var["low_cv_feats"] = ~mask

    return adata


def drop_low_variance(
    adata,
    by=("BatchNumber", "PlateNumber"),
    cutoff=0.5,
    drop=True,
    verbose=False,
):
    """Find features with low variance. This approach tries to account for the mean-variance
    relationship by applying a variance-stabilizing transformation before ranking variance of features.
    The implementation was described by Stuart et al., 2019:
    Stuart et al. (2019), Comprehensive integration of single-cell data. Cell.

    Different to Stuart et al. a polynomial fit with one degree (linear regression) is calculated to
    predict the variance of each feature as a function of its mean.
    Since negative mean values are possible in morphological data, the absolute values for means are taken.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        by (iterable, str or None): Groups to apply function to.
                If None, apply to whole anndata.AnnData object.
        cutoff (float): Drop features with deviation below cutoff.
        drop (bool): Drop features with low variance directly.
        verbose (bool)

    Returns:
        anndata.AnnData
        .uns['low_variance_feats']: Dropped features with low variance.
        .var['low_varinace_feats']: True for features with low varinace.
            Only if drop is False.
    """
    # check variables
    if by is not None:
        if isinstance(by, str):
            by = [by]
        elif isinstance(by, tuple):
            by = list(by)

        if not all(var in adata.obs.columns for var in by):
            raise KeyError(f"Variables defined in 'by' are not in annotations: {by}")

    assert isinstance(cutoff, (int, float)), (
        f"cutoff is expected to be type(float), " f"instead got {type(cutoff)}"
    )

    if by is not None:
        # store standardized variances
        stand_vars = []
        for groups, sub_df in adata.obs.groupby(by):
            # cache indices of group
            group_ix = sub_df.index
            X = adata[group_ix, :].X.copy()

            # get standardized variances
            stand_var = _stand_variance(X)
            stand_vars.append(stand_var)

        stand_vars = np.stack(stand_vars)
        stand_var = np.nanmean(stand_vars, axis=0)

    else:
        stand_var = _stand_variance(adata.X.copy())

    # apply cutoff
    mask = np.logical_and((stand_var > cutoff), (stand_var != np.nan))
    drop_feats = adata.var_names[~mask]

    if verbose:
        logger.info(
            f"Drop {len(drop_feats)} features with low coefficient of variance: {drop_feats}"
        )

    # drop
    if drop:
        adata = adata[:, mask].copy()
        adata.uns["low_variance_feats"] = drop_feats
    else:
        adata.var["low_variance_feats"] = ~mask

    return adata


def _stand_variance(X):
    """Calculate the standardized variances as described in
    Stuart et al. 2019.

    Args:
        adata (numpy.array): Multidimensional morphological data.
    """
    # calculate variance and mean
    variance = np.nanvar(X, axis=0)
    mean = np.nanmean(X, axis=0)

    # log10 transform variance and mean
    variance = np.log10(variance)
    mean = np.log10(np.abs(mean))

    # fit linear regression
    lr = np.polyfit(mean, variance, 1)
    p = np.poly1d(lr)

    # transform data
    X_trans = (X - mean) / np.sqrt(10 ** p(mean))
    # clip values above sqrt(N) with N number of cells
    X_trans[X_trans > np.sqrt(X_trans.shape[0])] = np.sqrt(X_trans.shape[0])

    # calculate variance of standardized values
    # stand_var = np.nanvar(X_trans, axis=0)
    stand_var = mad(X, nan_policy="omit", scale="normal")

    return stand_var
