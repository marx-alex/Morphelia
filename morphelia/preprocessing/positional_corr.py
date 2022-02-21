import numpy as np
from tqdm import tqdm
from scipy.stats import median_absolute_deviation as mad
from morphelia.tools import MedianPolish


def correct_plate_eff(
    adata,
    row_var="Metadata_Row",
    col_var="Metadata_Col",
    by=("BatchNumber", "PlateNumber"),
    max_iterations=10,
    method="median",
):
    """
    Implements Tukey's two-way median polish algorithm for additive models to
    correct positional effects in morphological data analysis.
    This procedure involves iterative median smoothing of rows and columns and division of each
    well value by the plate median absolute deviation. This generates B scores.
    This method should only be used for random plate layouts.

    Args:
        adata (anndata.AnnData): Per well aggregated multidimensional morphological data.
        row_var (str): Variable name for plate rows.
        col_var (str): Variable name for plate columns.
        by (list or tuple): Identifier variables for single plates.
        max_iterations (int): Maximum iterations.
        method (str): Median or mean (results would equal ANOVA).

    Returns:
        adata (anndata.AnnData)
    """

    # check variables
    assert (
        row_var in adata.obs.columns
    ), f"Variable for plate rows not in annotations: {row_var}."
    assert (
        col_var in adata.obs.columns
    ), f"Variable for plate rows not in annotations: {col_var}."

    if isinstance(by, tuple):
        by = list(by)
    else:
        assert isinstance(by, list), (
            f"Variables that identify single plates should be in a list or tuple, "
            f"instead got {type(by)}"
        )
    assert all(var in adata.obs.columns for var in by), (
        f"Variables that identify single " f"plates not in annotations: {by}"
    )

    method = method.lower()
    avail_methods = ["median", "mean"]
    assert method in avail_methods, (
        f"Method not in {avail_methods}, " f"instead got {method}"
    )

    # iterate over single plates
    for groups, sub_df in tqdm(
        adata.obs.groupby(by), desc="Iterate over all single plates"
    ):
        # check that adata object is already aggregated by well
        well_lst = list(zip(sub_df[row_var], sub_df[col_var]))
        assert len(well_lst) == len(
            set(well_lst)
        ), "AnnData object does not seem to be aggregated by well."

        # cache indices of group
        group_ix = sub_df.index

        # get unique well infos
        rows, row_pos = np.unique(sub_df["Metadata_Row"], return_inverse=True)
        cols, col_pos = np.unique(sub_df["Metadata_Col"], return_inverse=True)

        # iterate over features
        for feat in tqdm(adata.var_names, desc="Iterate over all features"):
            pivot_table = np.zeros((len(rows), len(cols)), dtype=float)
            pivot_table[row_pos, col_pos] = adata[group_ix, feat].X.flatten()

            # do median polish
            mp = MedianPolish(max_iterations=max_iterations, method=method)
            _, _, _, pivot_table, _ = mp.median_polish(pivot_table)

            # divide by median absolute deviation to get B score
            plate_mad = mad(pivot_table, axis=None, nan_policy="omit")
            pivot_table = pivot_table / plate_mad

            # transform pivot table back to 1d array
            adata[group_ix, feat].X = pivot_table.flatten()

    return adata
