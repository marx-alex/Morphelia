import os
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib import cm
from morphelia.tools import get_cmap


def _recursive_tree_pos(G, root, pos=None, t=0, vert_loc=0.5, width=1):

    if pos is None:
        pos = {root: (t, vert_loc)}
    else:
        pos[root] = (t, vert_loc)

    neighbors = list(G.neighbors(root))

    n_neighbors = len(neighbors)

    if n_neighbors > 0:
        if n_neighbors == 1:
            dy = width
            neighbors_y = [vert_loc]
        else:
            dy = width / 2
            neighbors_y = np.linspace(vert_loc - dy, vert_loc + dy, n_neighbors)

        for ix, neighbor in enumerate(neighbors):
            t = G.nodes[neighbor]["t"]
            vert_loc = neighbors_y[ix]
            pos = _recursive_tree_pos(
                G, neighbor, pos=pos, t=t, vert_loc=vert_loc, width=dy
            )

    return pos


def plot_tree(
    adata: ad.AnnData,
    root: int,
    time_var: str = "Metadata_Time",
    track_var: str = "Metadata_Track",
    parent_var: str = "Metadata_Track_Parent",
    root_var: str = "Metadata_Track_Root",
    gen_var: str = "Metadata_Gen",
    cmap: str = "wandb16",
    edges_width: int = 3,
    show: bool = False,
    save: Optional[str] = None,
) -> Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]:
    """Plot a lineage tree after tracking.

    Parameters
    ----------
        adata : anndata.AnnData
            Multidimensional morphological data
        root : int
            Root number as given by `root_var`
        time_var : str
            Variable with time information
        track_var : str
            Variable with track information
        parent_var : str
            Variable with parent information
        root_var : str
            Variable with root information
        gen_var : str
            Variable with generation information
        cmap : str
            Matplotlib or morphelia colormap
        edges_width : int
            Width of edges
        show : bool
            Show and return axis object
        save : str, optional
            Path where to save as `lineage_tree.png`

    Returns
    -------
    matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure if show is False and Axes

    Raises
    ------
    AssertionError
        If lineage is not a direct acyclic graph
    OSError
        If figure can not be saved at specified path
    """
    # subset adata
    tree = adata.obs.loc[
        adata.obs[root_var] == root, [track_var, parent_var, time_var, gen_var]
    ]
    tree["root"] = False

    def get_start_end_tp(df):
        df[[track_var, parent_var]] = (
            df[[track_var, parent_var]].astype(int).astype(str)
        )
        start = df.iloc[[df[time_var].argmin()], :]

        start.loc[start[parent_var] == start[track_var], "root"] = True

        end = df.iloc[[df[time_var].argmax()], :]

        end[parent_var] = end[track_var]

        start[track_var] = start[track_var] + "_start"
        start[parent_var] = start[parent_var] + "_end"
        end[track_var] = end[track_var] + "_end"
        end[parent_var] = end[parent_var] + "_start"

        start[gen_var] = np.nan

        df = pd.concat([start, end])
        return df

    # build tree structure
    tree = tree.groupby(track_var, as_index=False).apply(get_start_end_tp)
    tree = tree.set_index(track_var, drop=False)

    # create graph
    G_tree = nx.from_pandas_edgelist(
        tree.loc[~tree["root"], :],
        source=parent_var,
        target=track_var,
        edge_attr=True,
        create_using=nx.DiGraph(),
    )

    # remove self directed edges
    G_tree.remove_edges_from(nx.selfloop_edges(G_tree))

    assert nx.is_directed_acyclic_graph(
        G_tree
    ), f"Graph is not a DAG, check tree: {tree}"

    # add time points to nodes
    for node in G_tree.nodes:
        t = int(tree.loc[node, time_var])
        G_tree.nodes[node]["t"] = t

    # get positions
    root_str = str(root) + "_start"
    pos = _recursive_tree_pos(G_tree, root=root_str)

    # get color
    if cmap in plt.colormaps():
        cmap = cm.get_cmap(cmap).copy()
    else:
        cmap = get_cmap(cmap)
    edges = G_tree.edges()
    colors = [
        cmap(int(G_tree[u][v][gen_var]))
        if not np.isnan(G_tree[u][v][gen_var])
        else "lightgray"
        for u, v in edges
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    nx.draw_networkx_edges(
        G_tree,
        pos=pos,
        ax=ax,
        width=edges_width,
        arrows=False,
        edge_color=colors,
    )
    plt.axis("on")
    ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)
    ax.set_xlabel("Time (h)")
    plt.title(f"Lineage tree for root {root}")

    if save is not None:
        if not os.path.exists(save):
            raise OSError(f"Path does not exist: {save}")
        fig.savefig(os.path.join(save, "lineage_tree.png"), dpi=fig.dpi)

    if show:
        plt.show()
        return ax

    return fig, ax
