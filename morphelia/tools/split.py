import numpy as np
from sklearn.model_selection import train_test_split as tts


def train_test_split(adata, stratify=None, test_size=0.2, random_state=0):
    """
    Split anndata object into a training and a testing set.

    Arguments:
        adata (anndata.AnnData): Multidimensional morphological data.
        stratify (str): Variable in .obs to use for stratification.
        test_size (float): Size of test set.
        random_state (int)

    Returns:
        anndata.AnnData, anndata.AnnData
    """

    if stratify is not None:
        assert (
            stratify in adata.obs.columns
        ), f"stratify is not a label in .obs: {stratify}"
        stratify = adata.obs[stratify].to_numpy().flatten()

    ix_data = np.arange(adata.shape[0])
    ix_train, ix_test = tts(
        ix_data,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )

    adata_train = adata[ix_train, :].copy()
    adata_test = adata[ix_test, :].copy()

    return adata_train, adata_test


def group_shuffle_split(
    adata,
    group,
    stratify=None,
    test_size=0.2,
    random_state=0,
):
    """
    Train-test-split for grouped data.
    Avoids splitting of groups.

    Arguments:
        adata (anndata.AnnData): Multidimensional morphological data.
        group (str): Variable in .obs with groups.
        stratify (str): Variable in .obs to use for stratification.
        test_size (float): Size of test set.
        random_state (int)

    Returns:
        anndata.AnnData, anndata.AnnData
    """
    assert group in adata.obs.columns, f"group is not a label in .obs: {group}"
    groups = adata.obs[group].to_numpy()
    unique_groups = np.unique(groups)

    if stratify is not None:
        assert (
            stratify in adata.obs.columns
        ), f"stratify is not a label in .obs: {stratify}"
        stratify = adata.obs[stratify].to_numpy()

    trees_train, trees_test = tts(
        unique_groups,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )

    train_mask = np.isin(groups, trees_train)
    test_mask = np.isin(groups, trees_test)

    adata_train = adata[train_mask, :].copy()
    adata_test = adata[test_mask, :].copy()

    return adata_train, adata_test
