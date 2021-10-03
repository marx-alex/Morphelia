import numpy as np


def remove_noise(adata,
                 by='Metadata_Treatment',
                 mean_std_thresh=0.8,
                 drop=True,
                 verbose=False):
    """Removal of features with high mean of standard deviations within treatment groups.
    Features with mean standard deviation above mean_std_thresh will be removed.
    Normally distributed data is expected.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        by (str, tuple or list): Variable in observations that contains perturbations.
        mean_std_thresh (float): Threshold for high mean standard deviations.
        drop (bool): True to drop features directly.
        verbose (bool)
    """
    # check variables
    if by is not None:
        if isinstance(by, str):
            by = [by]
        elif isinstance(by, tuple):
            by = list(by)

        if not all(var in adata.obs.columns for var in by):
            raise KeyError(f"Variables defined in 'group_vars' are not in annotations: {by}")

    # iterate over group_vars groups
    if by is not None:
        # store standard deviations
        stds = []

        for group, sub_df in adata.obs.groupby(by):
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
        print(f"Drop {len(drop_feats)} noisy features: {drop_feats}")

    if drop:
        adata = adata[:, ~mask]
    else:
        adata.var['noisy'] = mask

    return adata
