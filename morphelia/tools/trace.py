# import internal libraries
import math
import os

# import external libraries
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA

__all__ = ["trace", "show_trace"]


def trace(obs, time_var="Metadata_Time", group_vars=("Metadata_Well", "Metadata_Field"),
          x_loc_id="Location_Center_X", y_loc_id="Location_Center_Y", loc_obj="Cells",
          trace_var="Metadata_Trace_Parent", tree_id="Metadata_Trace_Tree"):
    """Traces objects of a dataframe over time.

    It iterates over every well or field that can be traced over time.
    Fore every object and time point it calculates the nearest neighbor from
    the previous time point.

    Args:
        obs (pandas.DataFrame): Morphological data from cells with different time stamps.
            Annotations for objects are sufficient.
        time_var (str): Column names in morphome indicating time point
        group_vars (iterable): Column names in morphome to use for grouping.
            Should point to specific wells/ or fields on plates that can be compared over time.
        x_loc_id, y_loc_id (str): Identifiers for locations in column variables.
        loc_obj (str): Identifier for object for location identification in column variables.
        trace_var (str): Variable name used to store a trace identifier.
        tree_id (str): Variable name used to store branch number.

    Returns:
        pandas.DataFrame: Annotations with traces for all single objects.
    """
    # check that time_var and group_vars are in morphome variables
    if not all(var in obs.columns for var in [time_var] + list(group_vars)):
        raise KeyError(f"Check that variables for time and grouping are also in morphome annotations:"
                       f"Time variable: {time_var}, grouping variables: {group_vars}")

    # get location variables for x and y locations
    loc_x = [var for var in obs.columns if (x_loc_id in var) and (loc_obj in var)]
    loc_y = [var for var in obs.columns if (y_loc_id in var) and (loc_obj in var)]
    if len(loc_x) != 1 and len(loc_y) != 1:
        raise KeyError(f"No or more than one location variable found for object {loc_obj}."
                       f"Check the object or the identifiers for x and y locations: {x_loc_id}, {y_loc_id}.")
    loc_x = loc_x[0]
    loc_y = loc_y[0]

    # create new column to store index of parent object
    obs[trace_var] = np.nan
    # create new column to store branch
    obs[tree_id] = 0

    # iterate over every field and get field at different times
    for ix, (groups, field_df) in enumerate(obs.groupby(list(group_vars))):

        # ad trace index to objects with first time stamp
        obs.loc[(field_df[time_var] == obs[time_var].min()).index, tree_id] = range(
            len(obs.loc[(field_df[time_var] == obs[time_var].min()).index, tree_id]))

        # cache lagged values
        lagged = None

        # iterate over timepoints
        for t, t_df in field_df.groupby(time_var):

            # find closest object in lagged objects
            if lagged is not None:
                # get locations of objects and lagged objects
                t_loc = t_df[[loc_x, loc_y]].to_numpy()
                t_loc_lagged = lagged[[loc_x, loc_y]].to_numpy()

                # get nearest object in lagged objects for every object
                tree = cKDTree(t_loc_lagged)
                _, parent_ix = tree.query(t_loc, k=1)

                # assign lagged trace ids to objects
                obs.loc[t_df.index, tree_id] = lagged.iloc[parent_ix][tree_id].tolist()
                # assign trace parents to objects
                obs.loc[t_df.index, trace_var] = lagged.iloc[parent_ix].index

            # cache field_df
            lagged = obs.loc[t_df.index, [loc_x, loc_y, tree_id]]

    return obs


def show_trace(obs, fields,
               time_var="Metadata_Time", time_unit="h", trace_var="Metadata_Trace_Parent",
               tree_id="Metadata_Trace_Tree",
               x_loc_id="Location_Center_X", y_loc_id="Location_Center_Y", loc_obj="Cells",
               node_size=None, node_color=None, color_label=None, save=None):
    """Visualizes the trace of all objects in a specified field of view.

    Args:
        obs (pd.DataFrame): Morphological data from cells with different time stamps.
            Annotations for objects are sufficient.
        fields (defaultdict(list)): Variables and annotations that define field to visualize.
        time_var (str): Column names in morphome indicating time point.
        time_unit (str): Unit of time.
        trace_var (str): Variable name used to store a trace identifier.
        tree_id (str): Variable name used to store branch number.
        x_loc_id, y_loc_id (str): Identifiers for locations in column variables.
        loc_obj (str): Identifier for object for location identification in column variables.
        node_size (array): Array to use for node size. Indices must match indices of morphome.
        node_color (array): Array to use for node color. Indices must match indices of morphome.
        color_label (str): Label to use for color legend.
        save (str): If path is given, store figures.
    """
    # check that variables of fields are in morphome
    if not all(var in obs.columns for var in fields.keys()):
        raise KeyError(f"Variables defined in show are not in morphome annotations: {fields.keys()}")
    # check that values of fields are lists
    if not all(type(val) == list for val in fields.values()):
        raise TypeError(f"Values of show are not lists: {type(fields.values()[0])}")
    # check that values of fields have same length:
    if any(len(val) != len(list(fields.values())[0]) for val in fields.values()):
        raise ValueError("Values in show have different length.")
    # check that time_var, trace_var and branch_var are in morphome variables
    if not all(var in obs.columns for var in [time_var, trace_var, tree_id]):
        raise KeyError(f"Assert that variables for time, trace ids and trace branches are also in"
                       f" morphome annotation: {time_var}, {trace_var}, {tree_id}")

    # get location variables for x and y locations
    loc_x = [var for var in obs.columns if (x_loc_id in var) and (loc_obj in var)]
    loc_y = [var for var in obs.columns if (y_loc_id in var) and (loc_obj in var)]
    if len(loc_x) != 1 and len(loc_y) != 1:
        raise KeyError(f"No or more than one location variable found for object {loc_obj}."
                       f"Check the object or the identifiers for x and y locations: {x_loc_id}, {y_loc_id}.")
    loc_x = loc_x[0]
    loc_y = loc_y[0]

    # iterate over fields to show
    field_ids_lst = list(zip(*fields.values()))
    field_vars_lst = [tuple(fields.keys())] * len(field_ids_lst)

    for field_vars, field_ids in list(zip(field_vars_lst, field_ids_lst)):
        # create filder to select requested field
        morphome_filter = dict(zip(field_vars, field_ids))
        # create query term
        query_term = [f"({key} == '{item}')" if (
                type(item) == str) else f"({key} == {item})" for key, item in morphome_filter.items()]
        query_term = " and ".join(query_term)

        field_df = obs.query(query_term)

        # create graph from annotations
        G, pos = _create_graph(field_df, trace_var, loc_x, loc_y, time_var)
        # take input array as size
        if node_size is not None:
            ns = node_size[pd.to_numeric(field_df.index, errors='coerce')]
            ns_st = (ns - ns.min()) / (ns.max() - ns.min())
            ns = ns_st * 100
        else:
            ns = 100

        if node_color is not None:
            nc = node_color[pd.to_numeric(field_df.index, errors='coerce')]
            cmap = plt.get_cmap('plasma')
            vmin = nc.min()
            vmax = nc.max()
        else:
            nc = 'firebrick'
            cmap = None
            vmin = None
            vmax = None

        # options for plotting
        options = {
            'node_color': nc,
            'cmap': cmap,
            'vmin': vmin,
            'vmax': vmax,
            'node_size': ns,
            'width': 1,
            'alpha': 0.6,
            # 'connectionstyle': "angle,rad=2",
            'with_labels': False
        }

        # xs = [x for x, y in pos.values()]
        # height = int((len(field_df) / len(set(xs))) / 10)

        fig, ax = plt.subplots(figsize=(10, 15))
        nx.draw_networkx(G, pos=pos, **options)
        plt.axis("on")
        ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)
        ax.set_xlabel(f"Time ({time_unit})")
        # plot colorbar
        if node_color is not None:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm._A = []
            cbar = plt.colorbar(sm)
            if color_label is not None:
                cbar.ax.set_ylabel(f"{color_label}", rotation=270)
        fig.suptitle(f"Traces for: {', '.join([f'{key}: {val}' for key, val in morphome_filter.items()])}")
        plt.tight_layout()

        if save is not None:
            if not os.path.exists(save):
                raise OSError(f"Path does not exist: {save}")
            fig.savefig(save, dpi=fig.dpi)

    return None


def _create_graph(objects, trace_var, loc_x, loc_y, time_var):
    """Creates directed graph for all given objects.

    Args:
        objects (pandas.DataFrame): Stores objects with variables for location, and parent objects.
        trace_var (str): Variable name used to store a trace identifier.
        loc_x, loc_y (str): Variable name used to store object locations.
        time_var (str): Column names in morphome indicating time point.

    Returns:
        tuple: networkx directed Graph and positions
    """
    # create edges from indices and trace
    indices = list(pd.to_numeric(objects.index, errors='coerce'))
    parents = pd.to_numeric(objects[trace_var], errors='coerce').tolist()
    edges = list(zip(parents, indices))

    # delete tuples with np.nan
    edges = [edge for edge in edges if not any(math.isnan(n) for n in edge)]

    # create graph
    G = nx.DiGraph()
    G.add_nodes_from(indices)
    G.add_edges_from(edges)

    # get node positions
    locs = objects[[loc_x, loc_y]].to_numpy()
    y = PCA(n_components=1).fit_transform(locs)
    x = objects[time_var].to_list()
    xy = list(zip(x, y.flatten()))
    pos = dict(zip(pd.to_numeric(objects.index, errors='coerce'), xy))

    return G, pos
