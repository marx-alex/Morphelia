# external libraries
import numpy as np

# internal libraries
import warnings


def pseudostitch(md, update_loc=False, tile_grid=(5, 5),
                 img_size=(2048, 2048), img_overlap=0.2,
                 trow_var="Metadata_TileRow",
                 tcol_var="Metadata_TileCol",
                 tile_var="Metadata_Field",
                 tile_reading="horizontal",
                 group_vars=("BatchNumber", "PlateNumber", "Metadata_Well"),
                 loc_id="Location", y_id="Y", x_id="X",
                 verbose=False):
    """Updates location of objects if images were captured as grids with overlap.
    Duplicated objects in the overlap regions are removed.

    Args:
        md (anndata.AnnData): Multidimensional morphological data.
        update_loc (bool): Whether to update all location variables defined
            by loc_id, y_id and x_id
        tile_grid (tuple): Rows and columns of tile grid in well.
        img_size (tuple): Size of images (pixel in y- and x-direction).
        img_overlap (float): Overlap between tiles in percentage.
        tcol_var (str): Column name for tile column.
        trow_var (str): Column name for tile row.
        tile_var (str): Columns with tile identifiers
        tile_reading (str): Reading method of microscope: horizontal,
                horizontal_serp, vertical, vertical_serp.
        group_vars (iterable): Variables to use for grouping wells.
        loc_id (str): Unique identifier for object location variables (x and y)
        y_id (str): Unique identifier for x and y location variables.
        x_id (str): Unique identifier for x and y location variables.
        verbose (bool)

    Returns:
        (anndata.AnnData)
    """
    # first create a dictionary with tiles and their respective positions
    tile_grid_dict = _get_tile_grid_dict(tile_grid, tile_reading)

    # get dictionary with translation for every tile
    tile_trans = _get_tile_translation(tile_grid_dict, img_size, img_overlap)

    # check that tile_var exits
    if tile_var not in md.obs.columns:
        raise KeyError(f"Variable for tiles does not exist: {tile_var}")

    # check that columns for tile row and column exist
    if tcol_var not in md.obs.columns or trow_var not in md.obs.columns:
        warnings.warn(f"Column variables for tile positions are not in the data frame, trying to add them.",
                      UserWarning)
        # add new columns to morphome
        md.obs[trow_var] = md.obs[tile_var].map(lambda x: tile_grid_dict[x][0])
        md.obs[tcol_var] = md.obs[tile_var].map(lambda x: tile_grid_dict[x][1])

    # get location variables for x and y locations
    locvars_x = [col for col in md.obs.columns if loc_id in col and x_id in col]
    locvars_y = [col for col in md.obs.columns if loc_id in col and y_id in col]
    assert len(locvars_x) != 0 and len(locvars_y) != 0, f"No columns found for locations."

    # check that variables for grouping exist
    assert len([col for col in md.obs.columns if any(
        matcher.lower() in col.lower() for matcher in group_vars)]) == len(
        group_vars), f"Grouping variables not in observations: {group_vars}"
    # check variables for grouping
    assert len(group_vars) == 3, f"Grouping variables should be three: Batch number," \
                                 f"Plate Number and Well Number. ({group_vars})"

    # indicate duplicates in overlap regions
    duplicate_var = "Metadata_Duplicate"
    md.obs[duplicate_var] = 0

    # iterate over wells, update location vars and indicate duplicate objects in overlapping regions
    md.obs = md.obs.groupby(list(group_vars)).apply(_update_pos,
                                                    update_loc=update_loc,
                                                    tile_grid_dict=tile_grid_dict,
                                                    tile_trans=tile_trans,
                                                    locvars_x=locvars_x,
                                                    locvars_y=locvars_y,
                                                    trow_var=trow_var, tcol_var=tcol_var,
                                                    duplicate_var=duplicate_var, img_size=img_size,
                                                    tile_grid=tile_grid)
    if verbose:
        cell_count_raw = md.shape[0]
    # delete duplicates
    md = md[md.obs[duplicate_var] == 0, :]
    # drop duplicate observation
    md.obs = md.obs.drop(duplicate_var, axis=1)

    if verbose:
        cell_count_stitch = md.shape[0]
        print(f"{cell_count_raw - cell_count_stitch} duplicates removed.")

    return md


def _update_pos(well_df, update_loc, tile_grid_dict, tile_trans, locvars_x, locvars_y,
                trow_var, tcol_var, duplicate_var, img_size, tile_grid):
    """Takes observations from a single wells from pandas.groupby
    and updates location variables.

    Args:
        well_df (pd.DataFrame): Observations from a single well.
        update_loc (bool): Whether to update all location variables defined
            by loc_id, y_id and x_id
        tile_grid_dict (dict): Tile numbers and their respective positions in the grid.
        tile_trans (dict): Tiles and their translation (x and y direction).
        locvars_x (list): List of observations with x locations.
        locvars_y (list): List of observations with y locations.
        tcol_var (str): Column name for tile column.
        trow_var (str): Column name for tile row.
        duplicate_var (str): Variable name where duplicate indicator is stored.
        img_size (tuple): Size of images (pixel in y- and x-direction).
        tile_grid (tuple): Rows and columns of tile grid in well.

    Returns:
        (pandas.DataFrame): Updated well data frame.
    """
    # cache shape of stitched tiles
    cached_tiles = np.zeros((tile_grid[0] * img_size[0], tile_grid[1] * img_size[1]))

    # iterate over tile positions and update with tile_trans dictionary
    for node, (row, col) in tile_grid_dict.items():
        if update_loc:
            # update x locations
            well_df.loc[(well_df[trow_var] == row) & (well_df[tcol_var] == col), locvars_x] = well_df.loc[
                (well_df[trow_var] == row) & (well_df[tcol_var] == col), locvars_x].add(
                tile_trans[node][0], axis=0)
            # update y locations
            well_df.loc[(well_df[trow_var] == row) & (well_df[tcol_var] == col), locvars_y] = well_df.loc[
                (well_df[trow_var] == row) & (well_df[tcol_var] == col), locvars_y].add(
                tile_trans[node][1], axis=0)

            # indicate duplicates
            for index, r in well_df[(well_df[trow_var] == row) & (well_df[tcol_var] == col)].iterrows():
                if cached_tiles[int(r[locvars_y[0]]), int(r[locvars_x[0]])] == 1:
                    well_df.loc[index, duplicate_var] = 1

        else:
            # update x locations
            upd_loc_x = well_df.loc[(well_df[trow_var] == row) & (
                    well_df[tcol_var] == col), locvars_x[0]].add(
                tile_trans[node][0], axis=0)
            # update y locations
            upd_loc_y = well_df.loc[(well_df[trow_var] == row) & (
                    well_df[tcol_var] == col), locvars_y[0]].add(
                tile_trans[node][1], axis=0)

            # indicate duplicates
            for index, r in well_df[(well_df[trow_var] == row) & (well_df[tcol_var] == col)].iterrows():
                if cached_tiles[int(upd_loc_y[index]), int(upd_loc_x[index])] == 1:
                    well_df.loc[index, duplicate_var] = 1

        # update cached tiles
        x_min, x_max, y_min, y_max = (int(tile_trans[node][0]), int(tile_trans[node][0] + img_size[1]),
                                      int(tile_trans[node][1]), int(tile_trans[node][1] + img_size[0]))
        cached_tiles[y_min:y_max, x_min:x_max] = 1

    return well_df


def _get_tile_grid_dict(tile_grid, tile_reading):
    """Create Dictionary with row and column for each tile in a grid.

    Args:
        tile_grid (tuple): Rows and columns of tile grid in well.
        tile_reading (str): Reading method of microscope: horizontal,
                horizontal_serp, vertical, vertical_serp.

    Returns:
        (dict): Tile numbers and their respective positions in the grid.
    """
    # extract rows and columns
    assert len(tile_grid) == 2, f"Grid should be a tuple with two integers for rows and columns of tiles."
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
        col_ls = (list(range(1, tile_cols + 1)) + list(range(1, tile_cols + 1))[::-1]) * (tile_rows // 2)
        if len(col_ls) == 0:
            col_ls = list(range(1, tile_cols + 1))
        elif (tile_rows % 2) != 0:
            col_ls = col_ls + list(range(1, tile_cols + 1))
    elif tile_reading == "vertical_serp":
        col_ls = [col for col in range(1, tile_cols + 1) for _ in range(tile_rows)]
        row_ls = (list(range(1, tile_rows + 1)) + list(range(1, tile_rows + 1))[::-1]) * (tile_cols // 2)
        if len(row_ls) == 0:
            row_ls = list(range(1, tile_rows + 1))
        elif (tile_rows % 2) != 0:
            row_ls = row_ls + list(range(1, tile_rows + 1))
    else:
        reading_methods = ['horizontal', 'horizontal_serp', 'vertical', 'vertical_serp']
        raise ValueError(f"{tile_reading} not in reading methods: {reading_methods}")

    tiles = list(range(1, (tile_rows * tile_cols) + 1))
    tile_grid_dict = dict(zip(tiles, list(zip(row_ls, col_ls))))

    return tile_grid_dict


def _get_tile_translation(tile_grid_dict, img_size, img_overlap):
    """Calculates translation for each tile depending on their position
    in a grid, their size and overlap.
    Translations are returned as values in a dictionary with tile positions as keys.

    Args:
        tile_grid_dict (dict): Tile numbers and their respective positions in the grid.
        img_size (tuple): Size of images (pixel in y- and x-direction).
        img_overlap (float): Overlap between tiles in percentage.

    Returns:
        (dict): Tiles and their translation (x and y direction).
    """
    assert len(img_size) == 2, f"img_size should be a tuple with two integers for y- and x-dimensions of a tile image."
    assert (img_overlap < 1) and (img_overlap > 0), ("img_overlap should be a float thats represents tile overlap"
                                                      f"in percentage, instead got: {img_overlap}")
    y_size, x_size = img_size

    tile_trans = {}
    # iterate over tiles and find translations in x and y direction
    for tile, (row, col) in tile_grid_dict.items():
        x_trans = ((col - 1) * x_size) - ((col - 1) * (x_size * img_overlap))
        y_trans = ((row - 1) * y_size) - ((row - 1) * (y_size * img_overlap))
        tile_trans[tile] = (x_trans, y_trans)

    return tile_trans