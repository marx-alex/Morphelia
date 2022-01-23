import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GroupShuffleSplit


def train_test_split(adata,
                     stratify=None,
                     test_size=0.2,
                     random_state=0):
    """
    Split anndata object into a training and a testing set.

    :param adata:
    :param stratify:
    :param test_size:
    :param random_state:
    :return:
    """

    if stratify is not None:
        assert stratify in adata.obs.columns, f"stratify is not a label in .obs: {stratify}"
        stratify = adata.obs[stratify].to_numpy().flatten()

    ix_data = np.arange(adata.shape[0])
    ix_train, ix_test = tts(ix_data,
                            test_size=test_size,
                            stratify=stratify,
                            random_state=random_state)

    adata_train = adata[ix_train, :].copy()
    adata_test = adata[ix_test, :].copy()

    return adata_train, adata_test


def tree_split(adata,
               tree_var='Metadata_Trace_Tree',
               stratify=None,
               test_size=0.2,
               random_state=0):
    """
    Train-test-split for tracked time-lapse data.
    Tree var should be a unique label for tracked trees.

    :param adata:
    :param tree_var:
    :param stratify:
    :param test_size:
    :param random_state:
    :return:
    """
    assert tree_var in adata.obs.columns, f"tree_var is not a label in .obs: {tree_var}"
    all_trees = adata.obs[tree_var].to_numpy()
    trees, tree_ixs = np.unique(all_trees, return_index=True)

    if stratify is not None:
        assert stratify in adata.obs.columns, f"stratify is not a label in .obs: {stratify}"
        stratify = adata.obs[stratify].to_numpy()[tree_ixs]

    trees_train, trees_test = tts(trees,
                                  test_size=test_size,
                                  stratify=stratify,
                                  random_state=random_state)

    train_mask = np.isin(all_trees, trees_train)
    test_mask = np.isin(all_trees, trees_test)

    adata_train = adata[train_mask, :].copy()
    adata_test = adata[test_mask, :].copy()

    return adata_train, adata_test


def group_split(adata,
                y_label='Metadata_Treatment_Enc',
                groups=None,
                test_size=0.2,
                random_state=0):
    X = adata.X.copy()

    assert y_label in adata.obs.columns, f"y_label not in .obs: {y_label}"
    y = adata.obs[y_label].to_numpy().flatten()

    assert groups in adata.obs.columns, f"groups is not a label in .obs: {groups}"
    groups = adata.obs[groups].to_numpy().flatten()

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    ix_train, ix_test = [], []
    for ix_tv, ix_t in gss_test.split(X, y, groups):
        ix_train.append(ix_tv)
        ix_test.append(ix_t)
    ix_train = ix_train[0]
    ix_test = ix_test[0]

    adata_train = adata[ix_train, :].copy()
    adata_test = adata[ix_test, :].copy()

    return adata_train, adata_test
