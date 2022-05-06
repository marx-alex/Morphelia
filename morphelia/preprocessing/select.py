import anndata as ad
import numpy as np


def select_by_group(
    adata: ad.AnnData,
    var: str = "Metadata_Concentration",
    by: str = "Metadata_Treatment",
    method: str = "median",
) -> ad.AnnData:
    """Select specified subgroups.

    Iteratively select experimental subgroups.
    For every group in 'by' a group in 'var' is selected based on a specified method.
    Thereby, for example only the treatments with highest concentrations can be selected.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    var : str
        Variable in .obs to use for selection
    by : str
        Variable in .obs to use group grouping
    method: str
        `median`: Selects the median group
        `max`: Selects the maximum group
        `min`: Selects the minimum group

    Returns
    -------
    anndata.AnnData
        AnnData object with selected subgroups

    Raises
    -------
    AssertionError
        If `var` or `by` is not in `.obs`
    AssertionError
        If `method` is unknown

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np
    >>> import pandas as pd

    >>> data = np.random.rand(9, 5)
    >>> obs = pd.DataFrame({
    >>>     'treatment': [
    >>>         0, 0, 0, 1, 1, 1, 2, 2, 2
    >>>     ],
    >>>     'concentration': [
    >>>         0, 1, 2, 0, 1, 2, 0, 1, 2
    >>>     ]
    >>> })
    >>> adata = ad.AnnData(data, obs=obs)

    >>> adata = mp.pp.select_by_group(
    >>>     adata,
    >>>     by='treatment',
    >>>     var='concentration',
    >>>     method='max'
    >>> )  # select the highest concentration per treatment

    >>> adata.obs['concentration']
                concentration	treatment
    2-0	0	    2	            0
    5-1	1	    2	            1
    8-2	2	    2	            2
    """
    assert var in adata.obs.columns, f"var is not in .obs: {var}"
    assert by in adata.obs.columns, f"by is not in .obs: {by}"

    avail_methods = ["median", "max", "min"]
    method = method.lower()
    assert (
        method in avail_methods
    ), f"method must be one of {avail_methods}, instead got {method}"

    selected_adatas = []

    for group, df in adata.obs.groupby(by):
        group_index = df.index
        avail_groups = df[var].unique()

        if method == "median":
            selected_subgroup = np.median(avail_groups)
        elif method == "max":
            selected_subgroup = np.max(avail_groups)
        elif method == "min":
            selected_subgroup = np.min(avail_groups)
        else:
            raise NotImplementedError

        if selected_subgroup not in avail_groups:
            selected_subgroup = min(
                avail_groups, key=lambda x: abs(x - selected_subgroup)
            )

        selected_index = group_index[df[var] == selected_subgroup]
        selected_adatas.append(adata[selected_index])

    if len(selected_adatas) > 1:
        adata = selected_adatas[0].concatenate(*selected_adatas[1:])
    else:
        adata = selected_adatas[0]

    return adata
