import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from morphelia.tools.utils import get_subsample


def clustermap(adata,
               group_by=None,
               subsample=False,
               sample_size=1000,
               palette='Set2',
               show=False,
               save=None,
               seed=0,
               **kwargs):
    """
    Cluster map.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        group_by (str): Group cell by variable in .obs.
        subsample (bool): If True, fit models on subsample of data.
        sample_size (int): Size of subsample.
        Only if subsample is True.
        palette (str): Seaborn color palette.
        show (bool): Show plot and return axis.
        save (str): Path where to store plot as 'clustermap.png'.
        seed (int): Seed for subsample calculation.
        **kwargs: Keyword arguments passed to seaborn.clustermap.

    Returns:
        fig, ax
    """
    # get subsample
    if subsample:
        adata = get_subsample(adata,
                              sample_size=sample_size,
                              seed=seed)

    # get cell groups
    row_colors = None
    handles = None
    if group_by is not None:
        assert group_by in adata.obs.columns, f'group_by not in .obs: {group_by}'
        group_by = adata.obs[group_by].tolist()
        unique_groups = set(group_by)
        colors = sns.color_palette(palette, len(unique_groups))
        col_map = dict(zip(unique_groups, colors))
        row_colors = list(map(col_map.get, group_by))
        handles = [Patch(facecolor=col_map[name]) for name in col_map]

    kwargs.setdefault('metric', 'euclidean')
    kwargs.setdefault('method', 'ward')
    kwargs.setdefault('vmin', -3)
    kwargs.setdefault('vmax', 3)

    fig = plt.figure()
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    g = sns.clustermap(adata.X, row_cluster=True,
                       row_colors=row_colors,
                       cmap=cmap, **kwargs)
    ax = g.ax_heatmap
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_yticks([])
    lgd = None
    if handles is not None:
        lgd = plt.legend(handles, col_map, bbox_to_anchor=(1, 1),
                         bbox_transform=plt.gcf().transFigure, loc='upper left',
                         frameon=False)

    # save
    if save is not None:
        try:
            if handles is not None:
                plt.savefig(os.path.join(save, "clustermap.png"),
                            bbox_extra_artists=[lgd], bbox_inches='tight')
            else:
                plt.savefig(os.path.join(save, "clustermap.png"))
        except OSError:
            print(f'Can not save figure to {save}.')

    if show:
        plt.show()
        return ax

    return fig, ax
