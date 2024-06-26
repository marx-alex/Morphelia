import networkx as nx
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from typing import Union, Optional, Iterable


def plot_hmm(
    adata: ad.AnnData,
    node_size_factor: Union[float, int] = 1,
    edge_width_factor: Union[float, int] = 1,
    state_key: str = "HMMState",
    node_palette: Optional[Iterable] = None,
    pie_palette: Optional[Iterable] = None,
    draw_stationary_dist: Optional[str] = None,
    draw_labels: bool = True,
    show_self_loops: bool = False,
    pos: Optional[dict] = None,
    seed: int = 0,
    show: bool = True,
    save: Optional[str] = None,
    node_kwargs: Optional[dict] = None,
    edge_kwargs: Optional[dict] = None,
    label_kwargs: Optional[dict] = None,
):
    f"""Plot a transitions and stationary distributions of a
    Hidden Markov Model.

    Use `morphelia.ts.fit_hmm` beforehand.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated dataset with multidimensional morphological data
    node_size_factor : float, int
        Regulated the size of the nodes
    edge_width_factor : float, int
        Regulates the width of the edges
    state_key : str
        Key in `.obs` with states.
    node_palette : iterable, optional
        Colors for each state (node)
        Looks for predefined colors in `adata.uns[f'{state_key}_colors']`.
    pie_palette : iterable, optional
        Colors for each entity in `draw_stationary_dist`.
        Looks for predefined colors in `adata.uns[f'{draw_stationary_dist}_colors']`.
    draw_stationary_dist : str
        Show stationary distributions as pie charts on nodes
    draw_labels : bool
        Draw labels for hidden states on nodes
    show_self_loops : bool
        Show self-loops
    pos : dict, optional
        Use given node positions
    seed : int
    show : bool
        Show and return axes
    save : str, optional
        Path where to save figure
    node_kwargs : dict, optional
        Passed to networkx.draw_networkx_nodes
    edge_kwargs : dict, optional
        Passed to networkx.draw_networkx_edges
    label_kwargs : dict, optional
        Passed to networkx.draw_labels

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Only if show is False
    ax : matplotlib.pyplot.axes
    """
    assert "hmm" in adata.uns, "`hmm` not in `.uns`, use `fit_hmm` beforehand"

    # hmm to directed graph
    adjacency = adata.uns["hmm"]["params"]["transmat"].copy()
    stationary = adata.uns["hmm"]["params"]["stationary"].copy()
    G = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)

    # calculate state positions
    if pos is None:
        pos = nx.spring_layout(G, seed=seed)
    adata.uns["hmm"]["positions"] = pos

    # self assigning edges to zero
    if not show_self_loops:
        np.fill_diagonal(adjacency, 0)
        adjacency = adjacency / adjacency.sum(axis=1).reshape(-1, 1)
    # Make adjacency lines bigger
    adjacency = edge_width_factor * adjacency
    # create G with new weights
    G = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    edge_widths = list(nx.get_edge_attributes(G, "weight").values())

    node_sizes = node_size_factor * stationary

    # plot
    if edge_kwargs is None:
        edge_kwargs = dict()
    if node_kwargs is None:
        node_kwargs = dict()
    if label_kwargs is None:
        label_kwargs = dict()
    edge_kwargs.setdefault("arrowstyle", "->")
    edge_kwargs.setdefault("arrowsize", 10)
    edge_kwargs.setdefault("connectionstyle", "arc3,rad=0.1")

    if node_palette is None:
        if f"{state_key}_colors" in adata.uns:
            node_palette = adata.uns[f"{state_key}_colors"][: len(node_sizes)]

    fig, ax = plt.subplots()

    nx.draw_networkx_edges(G, pos, width=edge_widths, ax=ax, **edge_kwargs)

    if draw_stationary_dist is None:
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_size=5000 * node_sizes,
            node_color=node_palette,
            ax=ax,
            **node_kwargs,
        )
        nodes.set_zorder(0)

    else:
        assert (
            "hmm_by_key" in adata.uns
        ), "'hmm_by_key' not in '.uns', use 'fit_hmm_by_key' beforehand"
        stationary_dist = {
            k: adata.uns["hmm_by_key"][k]["stationary"]
            for k in adata.uns["hmm_by_key"].keys()
        }
        stationary_dist = pd.DataFrame(stationary_dist)
        labels = list(adata.uns["hmm_by_key"].keys())

        if pie_palette is None:
            if f"{draw_stationary_dist}_colors" in adata.uns:
                pie_palette = adata.uns[f"{draw_stationary_dist}_colors"]

        for ix, node in enumerate(G.nodes()):
            loc = ax.transLimits.transform(pos[node])
            radius = node_sizes[ix]
            ins = ax.inset_axes(
                [loc[0] - radius, loc[1] - radius, radius * 2, radius * 2], zorder=0
            )
            patches, texts = ins.pie(stationary_dist.iloc[ix, :], colors=pie_palette)

        ax.legend(patches, labels, bbox_to_anchor=(1.1, 1.05), frameon=False)

    if draw_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, **label_kwargs)

    ax.set_axis_off()

    # save
    if save is not None:
        try:
            plt.savefig(os.path.join(save, "hmm_plot.png"))
        except OSError:
            print(f"Can not save figure to {save}.")

    if show:
        plt.show()
        return ax

    return fig, ax
