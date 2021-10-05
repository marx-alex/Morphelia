import numpy as np
from scipy.spatial import cKDTree


def trace(adata,
          time_var="Metadata_Time",
          group_vars=("Metadata_Well", "Metadata_Field"),
          x_loc="Cells_Location_Center_X",
          y_loc="Cells_Location_Center_Y",
          trace_var="Metadata_Trace_Parent",
          tree_id="Metadata_Trace_Tree"):
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
        trace_var (str): Variable name used to store index of trace parent.
        tree_id (str): Variable name used to store unique branch number for a certain field.

    Returns:
        adata (anndata.AnnData)
    """

    # check that time_var and group_vars are in morphome variables
    if not all(var in adata.obs.columns for var in [time_var] + list(group_vars)):
        raise KeyError(f"Check that variables for time and grouping are also in morphome annotations:"
                       f"Time variable: {time_var}, grouping variables: {group_vars}")

    assert x_loc in adata.obs.columns, f"x_loc not in annotations: {x_loc}"
    assert y_loc in adata.obs.columns, f"y_loc not in annotations: {y_loc}"

    # create new column to store index of parent object
    adata.obs[trace_var] = np.nan
    # create new column to store branch
    adata.obs[tree_id] = 0

    # iterate over every field and get field at different times
    for ix, (groups, field_df) in enumerate(adata.obs.groupby(list(group_vars))):

        # ad trace index to objects with first time stamp
        adata.obs.loc[(field_df[time_var] == adata.obs[time_var].min()).index, tree_id] = range(
            len(adata.obs.loc[(field_df[time_var] == adata.obs[time_var].min()).index, tree_id]))

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
