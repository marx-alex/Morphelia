import numpy as np
from scipy.spatial import cKDTree


def trace(adata,
          time_var="Metadata_Time",
          group_vars=("Metadata_Well", "Metadata_Field"),
          x_loc="Cells_Location_Center_X",
          y_loc="Cells_Location_Center_Y",
          trace_var="Metadata_Trace_Parent",
          tree_id="Metadata_Trace_Tree",
          start_tp=0):
    """Traces objects of a dataframe over time.

    Iterates over every well or field that can be traced over time.
    Fore every object and time point calculate the nearest neighbor from
    the previous time point.

    Args:
        adata (anndata.AnnData): Morphological data from cells with different time stamps.
        time_var (str): Variable name for time point in annotations.
        group_vars (iterable): Variables in annotations.
            Should point to specific wells/ or fields on plates that can be compared over time.
        x_loc (str): Identifier for x location in annotations.
        y_loc (str): Identifier for y location in annotations.
        trace_var (str): Variable name used to store index of parent cell.
        tree_id (str): Variable name used to store unique branch number for a certain field.
        start_tp (int): Start time point.

    Returns:
        adata (anndata.AnnData)
    """

    # check variables
    if isinstance(group_vars, str):
        group_vars = [group_vars]
    elif isinstance(group_vars, tuple):
        group_vars = list(group_vars)

    if isinstance(group_vars, list):
        assert all(gv in adata.obs.columns for gv in group_vars), \
            f"One or all group_vars not in .obs.columns: {group_vars}"
    else:
        raise KeyError(f"Expected type(list) or type(str) for group_vars, "
                       f"instead got {type(group_vars)}")

    assert time_var in adata.obs.columns, f"time_var not in .obs.columns: {time_var}"
    assert x_loc in adata.obs.columns, f"x_loc not in .obs.columns: {x_loc}"
    assert x_loc in adata.obs.columns, f"x_loc not in .obs.columns: {x_loc}"
    assert y_loc in adata.obs.columns, f"y_loc not in .obs.columns: {y_loc}"

    # create new column to store index of parent object
    adata.obs[trace_var] = np.nan
    # create new column and store id for every trace tree
    adata.obs[tree_id] = np.nan
    n_start_tp = len(adata[adata.obs[time_var] == start_tp])
    if n_start_tp > 1:
        adata.obs.loc[adata.obs[time_var] == start_tp, tree_id] = range(n_start_tp)
    else:
        raise ValueError(f"No observation with time_var {time_var} and start_tp {start_tp}")

    # iterate over every field and get field at different times
    for ix, (groups, field_df) in enumerate(adata.obs.groupby(list(group_vars))):

        # cache lagged values
        lagged = None

        # iterate over timepoints
        for t, t_df in field_df.groupby(time_var):

            # find closest object in lagged objects
            if lagged is not None:
                # get locations of objects and lagged objects
                t_loc = t_df[[x_loc, y_loc]].to_numpy()
                t_loc_lagged = lagged[[x_loc, y_loc]].to_numpy()

                # get nearest object in lagged objects for every object
                tree = cKDTree(t_loc_lagged)
                _, parent_ix = tree.query(t_loc, k=1)

                # assign lagged trace ids to objects
                adata.obs.loc[t_df.index, tree_id] = lagged.iloc[parent_ix][tree_id].tolist()
                # assign trace parents to objects
                adata.obs.loc[t_df.index, trace_var] = lagged.iloc[parent_ix].index

            # cache field_df
            lagged = adata.obs.loc[t_df.index, [x_loc, y_loc, tree_id]]

    return adata
