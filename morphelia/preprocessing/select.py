import anndata as ad
import numpy as np


def select_by_group(
    adata: ad.AnnData,
    var: str = "Metadata_Concentration",
    by: str = "Metadata_Treatment",
    method: str = "median",
) -> ad.AnnData:
    """
    Iteratively select experimental subgroups.
    For every group in 'by' a group in 'var' is selected based on a specified method.

    Args:
        adata: Multidimensional morphological data.
        var: Variable in .obs to use for selection.
        by: Variable in .obs to use group grouping.
        method:
            median: Selects the median group.
            max: Selects the maximum group.
            min: Selects the minimum group.

    Returns:
        Selected AnnData object.
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
