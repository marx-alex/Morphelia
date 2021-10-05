# import internal libraries
import math
import os

# import external libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA

def show_trace(adata,
               fields,
               dim='2d',
               time_var="Metadata_Time",
               time_unit="h",
               trace_var="Metadata_Trace_Parent",
               tree_id="Metadata_Trace_Tree",
               x_loc_id="Location_Center_X",
               y_loc_id="Location_Center_Y",
               loc_obj="Cells",
               size=None, color=None, save=None,
               **kwargs):
    """Visualizes the trace of all objects in a specified field of view.

    Args:
        adata (anndata.AnnData): Morphological data from cells with different time stamps.
        fields (defaultdict(list)): Variables and annotations that define field to visualize.
        dim (str): Plot traces in 2 or 3 dimensions.
        time_var (str): Variable name for time point in annotations.
        time_unit (str): Unit to use for time.
        trace_var (str): Variable name used to store a trace identifier.
        tree_id (str): Variable name used to store branch number.
        x_loc_id (str): Identifier for x location in annotations.
        y_loc_id (str): Identifier for y location in annotations.
        loc_obj (str): Object name to use for location identification.
        size (str): Variable to use for size of edges in 2d representation of traces.
        color (str): Variable to use for color of edges in 2d representation of traces.
        save (str): If path is given, store figures.
        **kwargs (dict): Keyword arguments passed to matplotlib.pyplot.plot
    """
    # check that variables of fields are in morphome
    if not all(var in adata.obs.columns for var in fields.keys()):
        raise KeyError(f"Variables defined in show are not in anndata annotations: {fields.keys()}")
    # check that values of fields are lists
    if not all(type(val) == list for val in fields.values()):
        raise TypeError(f"Values of show are not lists: {type(fields.values()[0])}")
    # check that values of fields have same length:
    if any(len(val) != len(list(fields.values())[0]) for val in fields.values()):
        raise ValueError("Values in show have different length.")
    # check that time_var, trace_var and branch_var are in morphome variables
    if not all(var in adata.obs.columns for var in [time_var, trace_var, tree_id]):
        raise KeyError(f"Assert that variables for time, trace ids and trace branches are also in"
                       f" anndata annotation: {time_var}, {trace_var}, {tree_id}")

    # get location variables for x and y locations
    loc_x = [var for var in adata.obs.columns if (x_loc_id in var) and (loc_obj in var)]
    loc_y = [var for var in adata.obs.columns if (y_loc_id in var) and (loc_obj in var)]
    if len(loc_x) != 1 and len(loc_y) != 1:
        raise KeyError(f"No or more than one location variable found for object {loc_obj}."
                       f"Check the object or the identifiers for x and y locations: {x_loc_id}, {y_loc_id}.")
    loc_x = loc_x[0]
    loc_y = loc_y[0]

    # check dimensions
    dim = dim.lower()
    dims_avail = ['2d', '3d']
    assert dim in dims_avail, f'dim should be one of {dims_avail}, ' \
                              f'instead got {dim}'

    # iterate over fields to show
    field_ids_lst = list(zip(*fields.values()))
    field_vars_lst = [tuple(fields.keys())] * len(field_ids_lst)

    for field_vars, field_ids in list(zip(field_vars_lst, field_ids_lst)):
        # create filter to select requested field
        morphome_filter = dict(zip(field_vars, field_ids))
        # create query term
        query_term = [f"({key} == '{item}')" if (
                type(item) == str) else f"({key} == {item})" for key, item in morphome_filter.items()]
        query_term = " and ".join(query_term)

        adata_field = adata[adata.obs.query(query_term).index, :].copy()

        # create graph from annotations
        G, pos = _create_graph(adata_field.obs, dim, trace_var, loc_x, loc_y, time_var)

        # Extract node and edge positions from the layout
        node_xyz = np.array([pos[v] for v in sorted(G)])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

        # get parameters
        if size is not None:
            if size in adata.var_names.to_list():
                s = adata_field[:, size].X.copy().flatten()
            elif size in adata.obs.columns:
                s = adata_field.obs[size].to_numpy.flatten()
            else:
                raise ValueError(f"Size value not found: {size}.")

            s = (s - np.min(s)) / (np.max(s) - np.min(s))
            s = s * 20
        else:
            s = 5

        if color is not None:
            if color in adata.var_names.to_list():
                c = adata_field[:, color].X.copy().flatten()
            elif color in adata.obs.columns:
                c = adata_field.obs[color].to_numpy.flatten()
            else:
                raise ValueError(f"Size value not found: {color}.")

            vmin = np.min(c)
            vmax = np.max(c)
        else:
            c = 'firebrick'
            vmin = None
            vmax = None

        kwargs.setdefault('cmap', 'plasma')
        kwargs.setdefault('vmin', vmin)
        kwargs.setdefault('vmax', vmax)
        kwargs.setdefault('alpha', 0.6)

        # create figure
        fig = plt.figure(figsize=(7, 7))
        if dim == '3d':
            ax = fig.add_subplot(111, projection="3d")

            kwargs.setdefault('c', c)
            kwargs.setdefault('s', s)

            sc = ax.scatter(xs=node_xyz[:, 0], ys=node_xyz[:, 1], zs=node_xyz[:, 2], **kwargs)
            # plot edges
            for edge in edge_xyz:
                plt.plot(*edge.T, color="tab:gray")

            # ax.grid(False)
            # suppress tick labels
            for dim in (ax.xaxis, ax.yaxis):
                dim.set_ticks([])
            # Set axes labels
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel(f"time ({time_unit})")

        else:
            ax = fig.add_subplot(111)

            kwargs.setdefault('node_color', c)
            kwargs.setdefault('node_size', s)
            kwargs.setdefault('arrowstyle', "-")
            kwargs.setdefault('width', 1)
            kwargs.setdefault('with_labels', False)
            kwargs.setdefault('edge_color', 'tab:gray')

            nx.draw_networkx(G, pos=pos, ax=ax, **kwargs)

            plt.axis("on")
            ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)
            ax.set_xlabel(f"time ({time_unit})")
            # plot colorbar

        if color is not None:
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(kwargs['cmap']), norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm._A = []
            cbar = plt.colorbar(sm, shrink=0.3, pad=0.06, ax=ax)
            if color is not None:
                cbar.ax.set_ylabel(f"{color}", rotation=270, labelpad=15.0)

        fig.suptitle(f"Traces for: {', '.join([f'{key}: {val}' for key, val in morphome_filter.items()])}")
        plt.tight_layout()

        if save is not None:
            if not os.path.exists(save):
                raise OSError(f"Path does not exist: {save}")
            fig.savefig(save, dpi=fig.dpi)

    return None


def _create_graph(objects, dim, trace_var, loc_x, loc_y, time_var):
    """Creates directed graph for all given objects.

    Args:
        objects (pandas.DataFrame): Stores objects with variables for location, and parent objects.
        dim (str): Plot traces in 2 or 3 dimensions.
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
    if dim == '2d':
        locs = objects[[loc_x, loc_y]].to_numpy()
        y = PCA(n_components=1).fit_transform(locs)
        x = objects[time_var].to_list()
        xy = list(zip(x, y.flatten()))
        pos = dict(zip(pd.to_numeric(objects.index, errors='coerce'), xy))
    elif dim == '3d':
        x = objects[loc_x].to_list()
        y = objects[loc_y].to_list()
        z = objects[time_var].to_list()
        xyz = list(zip(x, y, z))
        pos = dict(zip(pd.to_numeric(objects.index, errors='coerce'), xyz))
    else:
        raise ValueError(f"dim neither '2d' nor '3d', instead got {dim}")

    return G, pos