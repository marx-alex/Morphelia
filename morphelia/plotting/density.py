from typing import Optional
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
):
    """
    Plot density.

    Args:
        adata: Multidimensional morphological data.
        by: Key in .obs.
        rep: Representation to plot.
        n_cols: Number of columns.
        size: Size of the scatter.
        cmap: Name of colormap to use in categorical plot.
        save: Path where to save plot.
        show: Show and only return axes if True.

    Returns:
        Figure and axes.
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
