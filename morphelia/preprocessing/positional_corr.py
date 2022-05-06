from typing import Union, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
import anndata as ad
from scipy.stats import median_absolute_deviation as mad
from morphelia.tools import MedianPolish


def correct_plate_eff(
    adata: ad.AnnData,
    row_var: str = "Metadata_Row",
    col_var: str = "Metadata_Col",
    by: Optional[Union[str, List[str], Tuple[str]]] = None,
    max_iterations: int = 10,
    method: str = "median",
):
    """Plate-effect correction.

    Implements Tukey's two-way median polish algorithm for additive models to
    correct positional effects in morphological data analysis.
    This procedure involves iterative median smoothing of rows and columns and division of each
    well value by the plate median absolute deviation. This generates B scores.
    This method should only be used for random plate layouts.

    Parameters
    ----------
    adata : anndata.AnnData
        Per-well aggregated multidimensional morphological data.
    row_var: str
        Variable name for plate rows.
    col_var : str
        Variable name for plate columns.
    by : str or list of str or tuple of str
        Identifier variables for single plates.
    max_iterations : int
        Maximum iterations.
    method : str
        `median` or `mean` (results would equal ANOVA).

    Returns
    -------
    anndata.AnnData
        Corrected `AnnData` object

    Raises
    -------
    AssertionError
        If `row_var` or `col_var` is not in `.obs`
    AssertionError
        If any label in `by` is not in `.obs`
    AssertionError
        If `method` is unknown
    AssertionError
        If data is not aggregated

    Examples
    ________
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np
    >>> import pandas as pd

    >>> data = np.random.rand(9, 5)
    >>> obs = pd.DataFrame({
    >>>     'row': [
    >>>         0, 0, 0, 1, 1, 1, 2, 2, 2
    >>>     ],
    >>>     'col': [
    >>>         0, 1, 2, 0, 1, 2, 0, 1, 2
    >>>     ]
    >>> })
    >>> adata = ad.AnnData(data, obs=obs)
    >>> adata[adata.obs['row'] == 0, :].X = adata[adata.obs['row'] == 0, :].X + 1  # add row effect

    >>> adata[adata.obs['row'] == 0, :].X
    ArrayView([[1.0934595, 1.4263059, 1.4732207, 1.5801971, 1.7162442],
               [1.027069 , 1.7313974, 1.7669635, 1.0097665, 1.3082862],
               [1.2328655, 1.5034275, 1.9537214, 1.5578113, 1.0974687]],
              dtype=float32)

    >>> adata = mp.pp.correct_plate_eff(
    >>>     adata,
    >>>     row_var='row',
    >>>     col_var='col',
    >>>     method='mean'
    >>> )
    >>> adata[adata.obs['row'] == 0, :].X
    ArrayView([[-0.35755962, -0.69392955, -0.64658576, -0.09151882,
                 1.2689846 ],
               [ 0.01890996,  1.4426355 , -0.01395249, -0.76600957,
                 0.23418863],
               [ 0.33864966, -0.748706  ,  0.66053826,  0.8575284 ,
                -1.5031731 ]], dtype=float32)
    """

    # check variables
    assert (
        row_var in adata.obs.columns
    ), f"Variable for plate rows not in annotations: {row_var}."
    assert (
        col_var in adata.obs.columns
    ), f"Variable for plate rows not in annotations: {col_var}."

    if by is not None:
        if isinstance(by, str):
            by = [by]
        else:
            assert isinstance(by, (list, tuple)), (
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
    if by is not None:
        for groups, sub_df in tqdm(
            adata.obs.groupby(by), desc="Iterate over all single plates"
        ):
            # check that adata object is already aggregated by well
            well_lst = list(zip(sub_df[row_var], sub_df[col_var]))
            assert len(well_lst) == len(
                set(well_lst)
            ), "AnnData object does not seem to be aggregated by well"

            # cache indices of group
            group_ix = sub_df.index

            # get unique well infos
            rows, row_pos = np.unique(sub_df["Metadata_Row"], return_inverse=True)
            cols, col_pos = np.unique(sub_df["Metadata_Col"], return_inverse=True)

            # iterate over features
            for feat in tqdm(adata.var_names, desc="Median polish over all features"):
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

    else:
        # check that adata object is already aggregated by well
        well_lst = list(zip(adata.obs[row_var], adata.obs[col_var]))
        assert len(well_lst) == len(
            set(well_lst)
        ), "AnnData object does not seem to be aggregated by well"

        # get unique well infos
        rows, row_pos = np.unique(adata.obs[row_var], return_inverse=True)
        cols, col_pos = np.unique(adata.obs[col_var], return_inverse=True)

        # iterate over features
        for feat in tqdm(adata.var_names, desc="Median polish over all features"):
            pivot_table = np.zeros((len(rows), len(cols)), dtype=float)
            pivot_table[row_pos, col_pos] = adata[:, feat].X.flatten()

            # do median polish
            mp = MedianPolish(max_iterations=max_iterations, method=method)
            _, _, _, pivot_table, _ = mp.median_polish(pivot_table)

            # divide by median absolute deviation to get B score
            plate_mad = mad(pivot_table, axis=None, nan_policy="omit")
            pivot_table = pivot_table / plate_mad

            # transform pivot table back to 1d array
            adata[:, feat].X = pivot_table.flatten()

    return adata
