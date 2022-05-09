# import internal libraries
import os
import warnings
from typing import Optional, List, Union, Tuple

# import external libraries
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_corr_matrix(
    corr_matrix: np.ndarray,
    groups: Optional[List[str]] = None,
    save: Optional[str] = None,
    show: bool = False,
    **kwargs,
) -> Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]:
    """Plot correlation matrix

    Highly correlated features and
    invalid (nan) features are indicated.
    The figure can be saved at a specified location.

    Parameters
    ----------
    corr_matrix : numpy.ndarray)
        Correlation Matrix
    groups : list, optional
        Group labels
    show : bool
        Show and return axes
    save : str
        Path to save figure
    **kwargs
        Keyword arguments are passed to seaborn.clustermap

    Returns
    -------
    matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure if show is False and Axes

    Raises
    ------
    OSError
        If figure can not be saved at specified path

    Examples
    --------
    >>> import morphelia as mp
    >>> import numpy as np

    >>> data = np.random.rand(4, 4)
    >>> mp.pl.plot_corr_matrix(data)
    """
    # do not show features with only nan
    corr_matrix = np.nan_to_num(corr_matrix)

    kwargs.setdefault("cmap", "viridis")
    kwargs.setdefault("method", "ward")
    kwargs.setdefault("vmin", -2)
    kwargs.setdefault("vmax", 2)

    # get group labels
    row_colors = None
    handles = None
    if groups is not None:
        unique_groups = set(groups)
        colors = sns.color_palette("Set2", len(unique_groups))
        col_map = dict(zip(unique_groups, colors))
        row_colors = list(map(col_map.get, groups))
        handles = [Patch(facecolor=col_map[name]) for name in col_map]

    sns.set_theme()
    fig = plt.figure()
    cm = sns.clustermap(corr_matrix, row_colors=row_colors, **kwargs)
    plt.suptitle("Correlation between features", y=1.05, fontsize=16)
    cm.ax_row_dendrogram.set_visible(False)
    ax = cm.ax_heatmap
    ax.get_yaxis().set_visible(False)
    if corr_matrix.shape[1] > 50:
        warnings.warn("Labels are hidden with more than 50 features", UserWarning)
        ax.get_xaxis().set_visible(False)
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
            plt.savefig(
                os.path.join(save, "feature_correlation.png"),
                bbox_extra_artists=[lgd],
                bbox_inches="tight",
            )
        except OSError:
            print(f"Can not save figure to {save}.")

    if show:
        plt.show()
        return ax

    return fig, ax
