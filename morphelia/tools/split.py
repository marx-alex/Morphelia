from typing import Optional, Tuple

import numpy as np
import anndata as ad
from sklearn.model_selection import train_test_split as tts


def train_test_split(
    adata: ad.AnnData,
    stratify: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 0,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Split an AnnData object into a training and a testing set.

    If needed, the split can be stratified by a group-variable.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    stratify : str, optional
        Variable in .obs to use for stratification
    test_size : float
        Size of test set.
    random_state : int
        Initialization for reproduction

    Returns
    -------
    anndata.AnnData, anndata.AnnData
        Train and test data

    Raises
    ------
    AssertionError
        If `stratify` is not in `.obs`

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np

    >>> data = np.random.rand(5, 5)
    >>> adata = ad.AnnData(data)

    >>> adata_train, adata_test = mp.tl.train_test_split(adata, test_size=0.2)  # split the data
    >>> adata_train.shape, adata_test.shape
    ((4, 5), (1, 5))
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
    adata: ad.AnnData,
    group: str,
    stratify: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 0,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Train-test-split for grouped data.

    This function avoids splitting groups.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    group : str
        Variable in `.obs` with groups
    stratify : str, optional
        Variable in `.obs` to use for stratification
    test_size : float
        Size of test set
    random_state : int

    Returns
    -------
    anndata.AnnData, anndata.AnnData
        Train and test data

    Raises
    ------
    AssertionError
        If `stratify` or `group` is not in `.obs`

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np
    >>> import pandas as pd

    >>> data = np.random.rand(10, 5)
    >>> obs = pd.DataFrame({"group": [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]})
    >>> adata = ad.AnnData(data, obs=obs)

    >>> adata_train, adata_test = mp.tl.group_shuffle_split(adata, group='group', test_size=0.2)
    >>> adata_train.obs
        group
    0	0
    1	0
    2	0
    3	1
    4	1
    5	1
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
