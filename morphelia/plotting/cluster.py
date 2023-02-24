import os
from typing import Optional, Union, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import anndata as ad

from morphelia.tools.utils import get_subsample
from morphelia.tools.utils import choose_representation


def clustermap(
    adata: ad.AnnData,
    group_by: Optional[str] = None,
    use_rep: Optional[str] = None,
    n_pcs: int = 50,
    subsample: bool = False,
    sample_size: int = 1000,
    palette: str = "Set2",
    show: bool = False,
    save: Optional[str] = None,
    seed: int = 0,
    **kwargs,
) -> Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]:
    """Cluster map.

    This is a convenient function to plot AnnData groups
    as a seaborn clustermap.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    group_by : str, optional
        Group cell by variable in `.obs`
    use_rep : str, optional
        Calculate similarity/distance representation of X in `.obsm`
    n_pcs : int
        Number principal components to use if use_pcs is `X_pca`
    subsample : bool
        If True, fit models on subsample of data
    sample_size : int
        Size of subsample.
        Only if subsample is True.
    palette : str
        Seaborn color palette
    show : bool
        Show plot and return axis
    save : str, optional
        Path where to store plot as `clustermap.png`
    seed : int
        Seed for subsample calculation.
    **kwargs : dict
        Keyword arguments passed to `seaborn.clustermap`

    Returns
    -------
    matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure if show is False and Axes

    Raises
    ------
    AssertionError
        If `group_by` is not in `.obs`
    OSError
        If figure can not be saved at specified path

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np

    >>> data = np.random.rand(4, 4)
    >>> adata = ad.AnnData(data)
    >>> mp.pl.clustermap(adata)
    """
    # get subsample
    if subsample:
        adata = get_subsample(adata, sample_size=sample_size, seed=seed)

    # get representation of data
    if use_rep is None:
        use_rep = "X"
    X = choose_representation(adata, rep=use_rep, n_pcs=n_pcs)

    # get cell groups
    row_colors = None
    handles = None
    if group_by is not None:
        assert group_by in adata.obs.columns, f"group_by not in .obs: {group_by}"
        group_by = adata.obs[group_by].tolist()
        unique_groups = set(group_by)
        colors = sns.color_palette(palette, len(unique_groups))
        col_map = dict(zip(unique_groups, colors))
        row_colors = list(map(col_map.get, group_by))
        handles = [Patch(facecolor=col_map[name]) for name in col_map]

    kwargs.setdefault("metric", "euclidean")
    kwargs.setdefault("method", "ward")
    kwargs.setdefault("vmin", -3)
    kwargs.setdefault("vmax", 3)

    fig = plt.figure()
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    g = sns.clustermap(X, row_cluster=True, row_colors=row_colors, cmap=cmap, **kwargs)
    ax = g.ax_heatmap
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_yticks([])
    lgd = None
    if handles is not None:
        lgd = plt.legend(
            handles,
            col_map,
            bbox_to_anchor=(1, 1),
            bbox_transform=plt.gcf().transFigure,
            loc="upper left",
            frameon=False,
        )

    # save
    if save is not None:
        try:
            if handles is not None:
                plt.savefig(
                    os.path.join(save, "clustermap.png"),
                    bbox_extra_artists=[lgd],
                    bbox_inches="tight",
                )
            else:
                plt.savefig(os.path.join(save, "clustermap.png"))
        except OSError:
            print(f"Can not save figure to {save}.")

    if show:
        plt.show()
        return ax

    return fig, ax
