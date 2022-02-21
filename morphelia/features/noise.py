import numpy as np
import warnings
import logging

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def drop_noise(
    adata,
    by="Metadata_Treatment",
    mean_std_thresh=0.8,
    drop=True,
    verbose=False,
):
    """Removal of features with high mean of standard deviations within treatment groups.
    Features with mean standard deviation above mean_std_thresh will be removed.

    Normally distributed data is expected.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        by (str, tuple or list): Variable in observations that contains perturbations.
            Should group data into groups with low expected standard deviation.
            I.g. same treatment and concentration.
        mean_std_thresh (float): Threshold for high mean standard deviations.
        drop (bool): True to drop features directly.
        verbose (bool)

    Returns:
        adata.AnnData
        adata.uns['noisy_feats'] = Dropped features with noise.
        adata.var['noisy_feats'] = True if feature contains noise.
             Only if drop is False.
    """
    # check variables
    if by is not None:
        if isinstance(by, str):
            by = [by]
        elif isinstance(by, tuple):
            by = list(by)

        if not all(var in adata.obs.columns for var in by):
            raise KeyError(
                f"Variables defined in 'group_vars' are not in annotations: {by}"
            )

    # brief check for normal distribution
    means = np.nanmean(adata.X, axis=0)
    if not all(np.logical_and(means > -2, means < 2)):
        warnings.warn(
            "Data does not seem to be normally distributed, "
            "use normalize() with 'standard', 'mad_robust' or 'robust' beforehand."
        )

    # iterate over group_vars groups
    if by is not None:
        # store standard deviations
        stds = []

        for _, sub_df in adata.obs.groupby(by):
            # cache indices of group
            group_ix = sub_df.index

            # calculate group std for every group
            X_group = adata[group_ix, :].X
            group_std = np.std(X_group, axis=0)
            stds.append(group_std)

        # calculate mean of group stds
        stds = np.stack(stds, axis=0)
        mean_stds = np.mean(stds, axis=0)

    else:
        # calculate std over whole X
        mean_stds = np.std(adata.X, axis=0)

    # get features to drop
    mask = mean_stds > mean_std_thresh
    drop_feats = adata.var_names[mask]

    if verbose:
        logger.info(f"Drop {len(drop_feats)} noisy features: {drop_feats}")

    if drop:
        adata = adata[:, ~mask].copy()

        adata.uns["noisy_feats"] = drop_feats
    else:
        adata.var["noisy_feats"] = mask

    return adata
