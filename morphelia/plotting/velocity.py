from typing import Optional
import os

import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc

from morphelia.external import _waypoint_sampling as waypoint_sampling
from morphelia.tools import choose_representation
from sklearn.metrics import pairwise_distances


def plot_velocity(
    adata: ad.AnnData,
    rep: str,
    vect_rep: str = "X_vect",
    kind: str = "quiver",
    by: Optional[str] = None,
    grid_dim: int = 50,
    n_waypoints: int = 50,
    min_cells: int = 20,
    n_cols: int = 5,
    size: int = 2,
    cmap: Optional[str] = None,
    save: Optional[str] = None,
    show: bool = False,
    velo_kwargs: dict = None,
    **kwargs,
):
    """Plot velocity as quiver on embedding.

    Parameters
    ----------
    adata: anndata.AnnData
        Multidimensional morphological data
    rep : str
        Representation to plot
    vect_rep : str
        Representation with vectors
    kind : str
        Plot quiver (`quiver`) or streamplot (`stream`)
    by : str, optional
        Key to plot different categories from
    grid_dim : int
        Grid dimension for streamplot
    n_waypoints : int
        Number of waypoints to use for quiver
    min_cells : int
        Minimum number of cells per waypoint
    n_cols : int
        Number of column in categorical plot
    size : int
        Size of foreground scatter in categorical plot
    cmap : str, optional
        Name of colormap to use in categorical plot
    save : str, optional
        Path where to save plot
    show : bool
        Show and only return axes if True
    velo_kwargs : dict
        Keyword arguments passed to either `pyplot.quiver` or `pyplot.streamplot`
    **kwargs
        Keyword arguments passed to `scanpy.pl.embedding`

    Returns
    -------
    matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure if show is False and Axes

    Raises
    ------
    AssertionError
        If `kind` is not `quiver` or `stream`
    OSError
        If figure can not be saved at specified path
    """

    if cmap is None:
        cmap = "tab10"
    cmap = plt.get_cmap(cmap)
    fg_size, bg_size = size, size / 2
    if velo_kwargs is None:
        velo_kwargs = {}

    avail_kinds = ["quiver", "stream"]
    assert (
        kind in avail_kinds
    ), f"'kind' must be one of {avail_kinds}, instead got {kind}"

    if by is not None:
        cats = list(np.sort(adata.obs[by].unique()))
        n_rows = int(np.ceil(len(cats) / n_cols))

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))

        for i, cat in enumerate(cats):
            row = int(np.floor(i / n_cols))
            col = np.remainder(i, n_cols)

            cat_data = adata[adata.obs[by] == cat, :]
            na_data = adata[adata.obs[by] != cat, :]
            # plot background
            axs[row, col].scatter(
                na_data.obsm[rep][:, 0],
                na_data.obsm[rep][:, 1],
                color="lightgrey",
                s=bg_size,
            )
            # plot foreground
            axs[row, col].scatter(
                cat_data.obsm[rep][:, 0],
                cat_data.obsm[rep][:, 1],
                color=cmap(i),
                s=fg_size,
            )
            if kind == "quiver":
                X, Y, U, V = _get_quiver(
                    cat_data,
                    X=rep,
                    V=vect_rep,
                    n_waypoints=n_waypoints,
                    min_cells=min_cells,
                )
                axs[row, col].quiver(X, Y, U, V, color="black", **velo_kwargs)
            elif kind == "stream":
                X, Y, U, V = _get_streamline(
                    cat_data, X=rep, V=vect_rep, grid_dim=grid_dim, min_cells=min_cells
                )
                axs[row, col].streamplot(X, Y, U, V, color="black", **velo_kwargs)
            axs[row, col].set_title(cat)
            axs[row, col].set_yticks([])
            axs[row, col].set_xticks([])

    else:
        fig, axs = plt.subplots()
        axs = sc.pl.embedding(adata, basis=rep, ax=axs, show=False, **kwargs)

        if kind == "quiver":
            X, Y, U, V = _get_quiver(
                adata, X=rep, V=vect_rep, n_waypoints=n_waypoints, min_cells=min_cells
            )
            axs.quiver(X, Y, U, V, color="black", **velo_kwargs)
        elif kind == "stream":
            X, Y, U, V = _get_streamline(
                adata, X=rep, V=vect_rep, grid_dim=grid_dim, min_cells=min_cells
            )
            axs.streamplot(X, Y, U, V, color="black", **velo_kwargs)

        axs.set_yticks([])
        axs.set_xticks([])

    # save
    if save:
        try:
            plt.savefig(
                os.path.join(save, "velocity.png"),
            )
        except OSError:
            print(f"Can not save figure to {save}.")

    if show:
        plt.show()
        return axs

    return fig, axs


def _get_quiver(
    adata: ad.AnnData,
    X: str = "X_nne",
    V: str = "X_vect",
    n_waypoints: int = 50,
    min_cells: int = 20,
):
    """
    Get grid parameters to draw a quiver plot.
    Use morphelia.tools.vectorize_emb beforehand.
    """
    assert X in adata.obsm.keys(), f"X not in .obsm: {X}"
    assert V in adata.obsm.keys(), f"V not in .obsm: {V}"

    X = choose_representation(adata, rep=X)
    assert len(X.shape) == 2, f"X must be 2-dimensional, instead got shape: {X.shape}"
    V = choose_representation(adata, rep=V)
    assert len(V.shape) == 2, f"V must be 2-dimensional, instead got shape: {V.shape}"

    # get waypoints
    wps = waypoint_sampling(X, n_waypoints=n_waypoints)
    waypoints = X[wps, :]

    # compute distance waypoints
    wp_dists = pairwise_distances(X, X[wps, :])
    # closest waypoint for every cell
    wp_ixs = np.argmin(wp_dists, axis=1)

    # find average vector for every waypoint
    wp_vect = np.zeros((len(wps), 2))
    for ix in range(len(wps)):
        if np.sum(wp_ixs == ix) > min_cells:
            wp_vect[ix, :] = np.nanmean(V[wp_ixs == ix, :], axis=0)

    vect_sum = np.sum(wp_vect, axis=1)
    zero_vect = vect_sum != 0

    return (
        waypoints[zero_vect, 0],
        waypoints[zero_vect, 1],
        wp_vect[zero_vect, 0],
        wp_vect[zero_vect, 1],
    )


def _get_streamline(
    adata: ad.AnnData,
    X: str = "X_nne",
    V: str = "X_vect",
    grid_dim: int = 50,
    min_cells: int = 5,
):
    """
    Get grid parameters to draw a streamplot.
    Use morphelia.tools.vectorize_emb beforehand.
    """
    assert X in adata.obsm.keys(), f"X not in .obsm: {X}"
    assert V in adata.obsm.keys(), f"V not in .obsm: {V}"

    X = choose_representation(adata, rep=X)
    assert len(X.shape) == 2, f"X must be 2-dimensional, instead got shape: {X.shape}"
    V = choose_representation(adata, rep=V)
    assert len(V.shape) == 2, f"V must be 2-dimensional, instead got shape: {V.shape}"

    # get boundary vals
    max_val = np.max(X, axis=0)
    max_x, max_y = max_val[0], max_val[1]
    min_val = np.min(X, axis=0)
    min_x, min_y = min_val[0], min_val[1]

    # get meshgrid of waypoints and their indices
    wp_x = np.linspace(min_x, max_x, grid_dim)
    wp_y = np.linspace(min_y, max_y, grid_dim)
    wp_ix = np.arange(grid_dim)

    wp_X, wp_Y = np.meshgrid(wp_x, wp_y)
    wps = np.column_stack((wp_X.flatten(), wp_Y.flatten()))
    wp_X_ix, wp_Y_ix = np.meshgrid(wp_ix, wp_ix)
    wp_ix = np.column_stack((wp_X_ix.flatten(), wp_Y_ix.flatten()))

    # compute distance waypoints
    wp_dists = pairwise_distances(X, wps)
    # closest waypoint for every cell
    wp_nrbs = np.argmin(wp_dists, axis=1)

    # find average vector for every waypoint
    Xv = np.zeros((grid_dim, grid_dim))
    Yv = np.zeros((grid_dim, grid_dim))
    for ix in range(len(wps)):
        if np.sum(wp_nrbs == ix) > min_cells:
            grid_x, grid_y = wp_ix[ix, 0], wp_ix[ix, 1]
            vect = np.nanmean(V[wp_nrbs == ix, :], axis=0)
            Xv[grid_y, grid_x] = vect[0]
            Yv[grid_y, grid_x] = vect[1]

    return wp_X, wp_Y, Xv, Yv
