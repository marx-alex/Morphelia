from collections import defaultdict
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll


def plot_trajectory(
    adata: ad.AnnData,
    rep: str,
    time_var: str = "Metadata_Time",
    treat_var: str = "Metadata_Treatment",
    method: str = "mean",
    bar_label: str = "Time",
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
    show: bool = False,
) -> Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]:
    """Plot trajectory on embedding of cell states.

    This function calculates the `mean` or `median` trajectory
    along treatments and time.
    The trajectories can be plotted on a two dimensional embedding.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    rep : str
        Name of embedding in `.obsm`
    time_var : str
        Variable in `.obs` with time information
    treat_var : str
        Variable in `.obs` with treatments
    method : str
        Calculate trajectory as `mean` or `median`
    bar_label : str
        Label for trajectory in the plot
    ax : matplotlib.pyplot.Axes, optional
        Plot on existing axis
    fig : matplotlib.pyplot.Figure, optional
        Plot on existing figure
    show : bool
        Show plot

    Returns
    -------
    matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure if show is False and Axes
    """
    avail_methods = ["mean", "median"]
    method = method.lower()
    if method == "mean":
        agg_fun = np.nanmean
    elif method == "median":
        agg_fun = np.nanmedian
    else:
        raise ValueError(f"method must be one of {avail_methods}, instead got {method}")

    # representation must be 2d
    assert rep in adata.obsm, f"representaiton is not in .obsm: {rep}"
    assert (
        len(adata.obsm[rep].shape) == 2
    ), f"data representation must be 2-dimensional, instead got shape {adata.obsm[rep].shape}"
    assert (
        adata.obsm[rep].shape[1] == 2
    ), f"data must be of shape (,2), instead got shape {adata.obsm[rep].shape}"

    assert time_var in adata.obs.columns, f"time_var not in .obs: {time_var}"
    if treat_var is not None:
        assert treat_var in adata.obs.columns, f"treat_var not in .obs: {treat_var}"
        treats = adata.obs[treat_var].unique()
    tps = sorted(adata.obs[time_var].unique())

    traj = defaultdict(list)

    for tp in tps:
        adata_tp = adata[adata.obs[time_var] == tp, :].copy()

        if treat_var is not None:
            for treat in treats:
                adata_treat = adata_tp[adata_tp.obs[treat_var] == treat, :].copy()
                X = adata_treat.obsm[rep]
                centroid = agg_fun(X, axis=0)
                traj["tp"].append(tp)
                traj["treat"].append(treat)
                traj["centroid_x"].append(centroid[0])
                traj["centroid_y"].append(centroid[1])

        else:
            X = adata_tp.obsm[rep]
            centroid = agg_fun(X, axis=0)
            traj["tp"].append(tp)
            traj["centroid_x"].append(centroid[0])
            traj["centroid_y"].append(centroid[1])

    # convert to dataframe
    traj = pd.DataFrame(traj)

    cmap = plt.get_cmap("binary")
    cmap = _truncate_colormap(cmap, minval=0.2)

    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    if treat_var is not None:
        for treat in treats:
            x = traj.loc[traj["treat"] == treat, "centroid_x"]
            y = traj.loc[traj["treat"] == treat, "centroid_y"]
            _colorline(x, y, cmap=cmap, zorder=1, ax=ax)
    else:
        _colorline(traj["centroid_x"], traj["centroid_y"], cmap=cmap, zorder=1, ax=ax)

    norm = mpl.colors.Normalize(vmin=0, vmax=max(tps))
    cax = fig.add_axes([0.94, 0.2, 0.15, 0.05])
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
        label=bar_label,
    )

    if show:
        plt.show()
        return ax

    return fig, ax


def _colorline(
    x,
    y,
    z=None,
    cmap=plt.get_cmap("copper"),
    norm=plt.Normalize(0.0, 1.0),
    linewidth=2,
    alpha=1.0,
    zorder=1,
    ax=None,
):
    """
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = _make_segments(x, y)
    lc = mcoll.LineCollection(
        segments,
        array=z,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc, autolim=True)

    return lc


def _make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    https://stackoverflow.com/a/18926541
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap
