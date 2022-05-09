from typing import Optional, Union, Tuple
import os

import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc


def plot_density(
    adata: ad.AnnData,
    by: str,
    rep: str,
    n_cols: int = 5,
    size: int = 2,
    cmap: Optional[str] = None,
    save: Optional[str] = None,
    show: bool = False,
    **kwargs,
) -> Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]:
    """Plot density.

    Convenient function for scanpy.pl.embedding_density.
    Gaussian kernel density estimates are calculated for a specified condition.
    Densities are then plotted as a grid with a plot for each unique condition.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    by : str
        Key in `.obs`
    rep : str
        Representation in `.obsm`
    n_cols : int
        Number of columns
    size : int
        Size of the scatter points
    cmap : str, optional
        Name of colormap to use in categorical plot
    save : str, optional
        Path where to save plot
    show : bool
        Show and only return axes if True

    Returns
    -------
    matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure if show is False and Axes

    Raises
    ------
    OSError
        If figure can not be saved at specified path
    """
    basis = rep.split("_")[-1]
    sc.tl.embedding_density(adata, basis=basis, groupby=by)
    dens_key = f"{basis}_density_{by}"

    if cmap is None:
        cmap = "Blues"
    cmap = plt.get_cmap(cmap)
    fg_size, bg_size = size, size / 2

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
            c=cat_data.obs[dens_key],
            s=fg_size,
            cmap=cmap,
            **kwargs,
        )

        axs[row, col].set_title(cat)
        axs[row, col].set_yticks([])
        axs[row, col].set_xticks([])

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
