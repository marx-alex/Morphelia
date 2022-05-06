# external libraries
import numpy as np
import pandas as pd
import anndata as ad

# internal libraries
import logging
from typing import Union, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def pseudostitch(
    adata: ad.AnnData,
    update_loc: bool = False,
    tile_grid: tuple = (5, 5),
    img_size: Tuple[int] = (2048, 2048),
    img_overlap: float = 0.2,
    trow_var: str = "Metadata_TileRow",
    tcol_var: str = "Metadata_TileCol",
    tile_var: str = "Metadata_Field",
    tile_reading: str = "horizontal",
    group_vars: Union[tuple, list] = ("BatchNumber", "PlateNumber", "Metadata_Well"),
    loc_id: str = "Location",
    y_id: str = "Y",
    x_id: str = "X",
    verbose: bool = False,
):
    """Pseudostitching.

    Updates location of objects if images were captured as grids with overlap.
    Duplicated objects in the overlap regions are removed.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data.
    update_loc : bool
        Whether to update all location variables defined
        by loc_id, y_id and x_id
    tile_grid : tuple of int
        Rows and columns of tile grid in well.
    img_size : tuple of int
        Size of images (pixel in y- and x-direction).
    img_overlap : float
        Overlap between tiles in percentage.
    tcol_var : str
        Column name for tile column.
    trow_var : str
        Column name for tile row.
    tile_var : str
        Columns with tile identifiers
    tile_reading : str
        Reading method of microscope: horizontal,
        horizontal_serp, vertical, vertical_serp.
    group_vars : tuple of str or list of str
    Variables to use for grouping wells.
    loc_id : str
        Unique identifier for object location variables (x and y)
    y_id : str
        Unique identifier for x and y location variables.
    x_id : str
        Unique identifier for x and y location variables.
    verbose : bool

    Returns
    -------
    anndata.AnnData
        Psuedostitched AnnData object
    """
    # first create a dictionary with tiles and their respective positions
    tile_grid_dict = _get_tile_grid_dict(tile_grid, tile_reading)

    # get dictionary with translation for every tile
    tile_trans = _get_tile_translation(tile_grid_dict, img_size, img_overlap)

    # check that tile_var exits
    if tile_var not in adata.obs.columns:
        raise KeyError(f"Variable for tiles does not exist: {tile_var}")

    # check that columns for tile row and column exist
    if tcol_var not in adata.obs.columns or trow_var not in adata.obs.columns:
        logger.warning(
            "Column variables for tile positions are not in the data frame, trying to add them."
        )
        # add new columns to morphome
        adata.obs[trow_var] = adata.obs[tile_var].map(lambda x: tile_grid_dict[x][0])
        adata.obs[tcol_var] = adata.obs[tile_var].map(lambda x: tile_grid_dict[x][1])

    # get location variables for x and y locations
    locvars_x = [col for col in adata.obs.columns if loc_id in col and x_id in col]
    locvars_y = [col for col in adata.obs.columns if loc_id in col and y_id in col]
    assert (
        len(locvars_x) != 0 and len(locvars_y) != 0
    ), "No columns found for locations."

    # check that variables for grouping exist
    assert len(
        [
            col
            for col in adata.obs.columns
            if any(matcher.lower() in col.lower() for matcher in group_vars)
        ]
    ) == len(group_vars), f"Grouping variables not in observations: {group_vars}"
    # check variables for grouping
    assert len(group_vars) == 3, (
        f"3 grouping variables expected: Batch number, "
        f"Plate Number and Well Number. ({group_vars})"
    )

    # indicate duplicates in overlap regions
    duplicate_var = "Metadata_Duplicate"
    adata.obs[duplicate_var] = 0

    # iterate over wells, update location vars and indicate duplicate objects in overlapping regions
    adata.obs = adata.obs.groupby(list(group_vars)).apply(
        _update_pos,
        update_loc=update_loc,
        tile_grid_dict=tile_grid_dict,
        tile_trans=tile_trans,
        locvars_x=locvars_x,
        locvars_y=locvars_y,
        trow_var=trow_var,
        tcol_var=tcol_var,
        duplicate_var=duplicate_var,
        img_size=img_size,
        tile_grid=tile_grid,
    )
    if verbose:
        cell_count_raw = adata.shape[0]
    # delete duplicates
    adata = adata[adata.obs[duplicate_var] == 0, :]
    # drop duplicate observation
    adata.obs = adata.obs.drop(duplicate_var, axis=1)

    if verbose:
        cell_count_stitch = adata.shape[0]
        logger.info(f"{cell_count_raw - cell_count_stitch} duplicates removed.")

    return adata


def _update_pos(
    well_df: pd.DataFrame,
    update_loc: bool,
    tile_grid_dict: dict,
    tile_trans: dict,
    locvars_x: list,
    locvars_y: list,
    trow_var: str,
    tcol_var: str,
    duplicate_var: str,
    img_size: tuple,
    tile_grid: tuple,
):
    """
    Takes observations from a single wells from pandas.groupby
    and updates locations.

    Parameters
    ----------
    well_df : pandas.DataFrame
        Observations from a single well.
    update_loc : bool
        Whether to update all location variables defined
        by loc_id, y_id and x_id
    tile_grid_dict : dict
        Tile numbers and their respective positions in the grid.
    tile_trans : dict
        Tiles and their translation (x and y direction).
    locvars_x : list
        List of observations with x locations.
    locvars_y : list
        List of observations with y locations.
    tcol_var : str
        Column name for tile column.
    trow_var : str
        Column name for tile row.
    duplicate_var : str
        Variable name where duplicate indicator is stored.
    img_size : tuple
        Size of images (pixel in y- and x-direction).
    tile_grid : tuple
        Rows and columns of tile grid in well.

    Returns
    -------
    pandas.DataFrame
        DataFrame with updated locations
    """
    # cache shape of stitched tiles
    cached_tiles = np.zeros((tile_grid[0] * img_size[0], tile_grid[1] * img_size[1]))

    # iterate over tile positions and update with tile_trans dictionary
    for node, (row, col) in tile_grid_dict.items():
        if update_loc:
            # update x locations
            well_df.loc[
                (well_df[trow_var] == row) & (well_df[tcol_var] == col),
                locvars_x,
            ] = well_df.loc[
                (well_df[trow_var] == row) & (well_df[tcol_var] == col),
                locvars_x,
            ].add(
                tile_trans[node][0], axis=0
            )
            # update y locations
            well_df.loc[
                (well_df[trow_var] == row) & (well_df[tcol_var] == col),
                locvars_y,
            ] = well_df.loc[
                (well_df[trow_var] == row) & (well_df[tcol_var] == col),
                locvars_y,
            ].add(
                tile_trans[node][1], axis=0
            )

            # indicate duplicates
            for index, r in well_df[
                (well_df[trow_var] == row) & (well_df[tcol_var] == col)
            ].iterrows():
                if cached_tiles[int(r[locvars_y[0]]), int(r[locvars_x[0]])] == 1:
                    well_df.loc[index, duplicate_var] = 1

        else:
            # update x locations
            upd_loc_x = well_df.loc[
                (well_df[trow_var] == row) & (well_df[tcol_var] == col),
                locvars_x[0],
            ].add(tile_trans[node][0], axis=0)
            # update y locations
            upd_loc_y = well_df.loc[
                (well_df[trow_var] == row) & (well_df[tcol_var] == col),
                locvars_y[0],
            ].add(tile_trans[node][1], axis=0)

            # indicate duplicates
            for index, r in well_df[
                (well_df[trow_var] == row) & (well_df[tcol_var] == col)
            ].iterrows():
                if cached_tiles[int(upd_loc_y[index]), int(upd_loc_x[index])] == 1:
                    well_df.loc[index, duplicate_var] = 1

        # update cached tiles
        x_min, x_max, y_min, y_max = (
            int(tile_trans[node][0]),
            int(tile_trans[node][0] + img_size[1]),
            int(tile_trans[node][1]),
            int(tile_trans[node][1] + img_size[0]),
        )
        cached_tiles[y_min:y_max, x_min:x_max] = 1

    return well_df


def _get_tile_grid_dict(tile_grid: tuple, tile_reading: str):
    """Create Dictionary with row and column for each tile in a grid.

    Parameters
    ----------
    tile_grid : tuple
        Rows and columns of tile grid in well
    tile_reading : str
        Reading method of microscope: horizontal,
        horizontal_serp, vertical, vertical_serp

    Returns
    -------
    dict
        Tile numbers and their respective positions in the grid
    """
    # extract rows and columns
    assert (
        len(tile_grid) == 2
    ), "Grid should be a tuple with two integers for rows and columns of tiles."
    tile_rows, tile_cols = tile_grid

    # create a dictionary with ImageNumbers as keys and TileRow and TileCol as items
    if tile_reading == "horizontal":
        col_ls = list(range(1, tile_cols + 1)) * tile_rows
        row_ls = [row for row in range(1, tile_rows + 1) for _ in range(tile_cols)]
    elif tile_reading == "vertical":
        row_ls = list(range(1, tile_rows + 1)) * tile_cols
        col_ls = [col for col in range(1, tile_cols + 1) for _ in range(tile_rows)]
    elif tile_reading == "horizontal_serp":
        row_ls = [row for row in range(1, tile_rows + 1) for _ in range(tile_cols)]
        col_ls = (
            list(range(1, tile_cols + 1)) + list(range(1, tile_cols + 1))[::-1]
        ) * (tile_rows // 2)
        if len(col_ls) == 0:
            col_ls = list(range(1, tile_cols + 1))
        elif (tile_rows % 2) != 0:
            col_ls = col_ls + list(range(1, tile_cols + 1))
    elif tile_reading == "vertical_serp":
        col_ls = [col for col in range(1, tile_cols + 1) for _ in range(tile_rows)]
        row_ls = (
            list(range(1, tile_rows + 1)) + list(range(1, tile_rows + 1))[::-1]
        ) * (tile_cols // 2)
        if len(row_ls) == 0:
            row_ls = list(range(1, tile_rows + 1))
        elif (tile_rows % 2) != 0:
            row_ls = row_ls + list(range(1, tile_rows + 1))
    else:
        reading_methods = [
            "horizontal",
            "horizontal_serp",
            "vertical",
            "vertical_serp",
        ]
        raise ValueError(f"{tile_reading} not in reading methods: {reading_methods}")

    tiles = list(range(1, (tile_rows * tile_cols) + 1))
    tile_grid_dict = dict(zip(tiles, list(zip(row_ls, col_ls))))

    return tile_grid_dict


def _get_tile_translation(tile_grid_dict: dict, img_size: tuple, img_overlap: float):
    """Calculates translation for each tile depending on their position
    in a grid, their size and overlap.
    Translations are returned as values in a dictionary with tile positions as keys.

    Parameters
    ----------
    tile_grid_dict : dict
        Tile numbers and their respective positions in the grid.
    img_size : tuple
        Size of images (pixel in y- and x-direction).
    img_overlap : float
        Overlap between tiles in percentage.

    Returns
    -------
    dict
        Tiles and their translation (x and y direction).
    """
    assert (
        len(img_size) == 2
    ), "img_size should be a tuple with two integers for y- and x-dimensions of a tile image."
    assert (img_overlap < 1) and (img_overlap > 0), (
        "img_overlap should be a float thats represents tile overlap"
        f"in percentage, instead got: {img_overlap}"
    )
    y_size, x_size = img_size

    tile_trans = {}
    # iterate over tiles and find translations in x and y direction
    for tile, (row, col) in tile_grid_dict.items():
        x_trans = ((col - 1) * x_size) - ((col - 1) * (x_size * img_overlap))
        y_trans = ((row - 1) * y_size) - ((row - 1) * (y_size * img_overlap))
        tile_trans[tile] = (x_trans, y_trans)

    return tile_trans
