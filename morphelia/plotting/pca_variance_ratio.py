# import internal libraries
import os
from typing import Optional

# import external libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import anndata as ad


def pca_variance_ratio(
    adata: ad.AnnData,
    n_pcs: Optional[int] = 100,
    show: bool = False,
    save: Optional[str] = None,
    **kwargs,
):
    """Plots cumulative percentage variance explained by each principal component.

    Use scanpy.pp.pca beforehand.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    n_pcs : int, optional
        Number of PCAs to plot
    show : bool
        Show and return axes
    save : str, optional
        Path where to save figure
    **kwargs
        Keyword passed to `matplotlib.pyplot.plot`

    Returns
    -------
    matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure if show is False and Axes

    Raises
    -------
    TypeError
        If `n_pcs` is not of type int or None
    OSError
        If figure can not be saved at specified path
    """
    if isinstance(n_pcs, int):
        if n_pcs > adata.shape[1]:
            n_pcs = adata.shape[1]
        var_ratio = getattr(adata, "uns")["pca"]["variance_ratio"][:n_pcs]
    elif n_pcs is None:
        var_ratio = getattr(adata, "uns")["pca"]["variance_ratio"]
    else:
        raise TypeError(f"n_pcas expected to be int or None, instead got {type(n_pcs)}")

    # get cumulative scores
    var_ratio = np.cumsum(var_ratio) * 100

    # get pcas for x_axis
    pca_ixs = range(1, len(var_ratio) + 1)

    # plot
    kwargs.setdefault("color", "#D66853")
    # kwargs.setdefault('marker', 'o')

    sns.set_theme()
    fig, ax = plt.subplots()
    ax.plot(pca_ixs, var_ratio, **kwargs)
    ax.set_ylabel("Percentage Explained Variance")
    ax.set_xlabel("Number of Principal Components")

    # save
    if save is not None:
        try:
            plt.savefig(os.path.join(save, "pca_variance_ratio.png"))
        except OSError:
            print(f"Can not save figure to {save}.")

    if show:
        plt.show()
        return ax

    return fig, ax
