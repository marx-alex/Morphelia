# import external libraries
import numpy as np


def subsample(
    adata,
    perc=0.1,
    by=("BatchNumber", "PlateNumber", "Metadata_Well"),
    grouped=None,
    with_replacement=False,
    seed=0,
):
    """Gives a subsample of the data by selecting objects from given groups.

    Args:
        adata (anndata.AnnData): Annotated data object.
        perc (float): Percentage of objects to store in subsample.
        by (list, None): Group by those variables before sampling.
        grouped (str): Sample groups, not single instances.
        with_replacement (bool): Sample with replacement.
        seed (int): Seed for initialization.

    Returns:
        anndata.AnnData
    """
    assert 0 <= perc <= 1, f"Use a float between 0 and 1 for perc: {perc}"
    if grouped is not None:
        assert (
            grouped in adata.obs.columns
        ), f"Variable for grouped sampling not in .obs: {grouped}"

    # seed
    np.random.seed(seed)

    if by is not None:
        # check that variables in by are in anndata
        assert all(
            var in adata.obs.columns for var in by
        ), f"Variables defined in 'by' are not in annotations: {by}"

        # store subsample indices
        subsample_ixs = []

        # iterate over md with grouping variables
        for groups, sub_df in adata.obs.groupby(list(by)):

            group_ix = _get_subsample_ixs(
                sub_df, perc=perc, replace=with_replacement, grouped=grouped
            )

            subsample_ixs.append(group_ix)

        # make anndata object from subsample
        subsample_ix = np.concatenate(subsample_ixs)

    else:
        subsample_ix = _get_subsample_ixs(
            adata.obs, perc=perc, replace=with_replacement, grouped=grouped
        )

    # if with replacement, observation names are not unique
    adata = adata[subsample_ix, :]
    adata.obs_names_make_unique()
    return adata.copy()


def _get_subsample_ixs(df, perc=0.1, replace=False, grouped=None):
    # cache indices subsample
    if grouped is None:
        n_samples = round(perc * len(df))
        group_ix = np.random.choice(df.index, size=n_samples, replace=replace)
    else:
        # assert groups are complete
        unique_groups = df[grouped].unique()
        n_unique_groups = len(unique_groups)
        n_samples = round(perc * n_unique_groups)
        groups_choice = np.random.choice(unique_groups, n_samples, replace=replace)
        group_ixs = [df.loc[df[grouped] == choice, :].index for choice in groups_choice]
        group_ix = np.concatenate(group_ixs)

    return group_ix
