# import internal libraries
import math
import string
import os
from typing import Optional, Union, Tuple

# import external libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.lines import Line2D
import matplotlib
import numpy as np
import seaborn as sns
import anndata as ad

from morphelia.eval import dist_matrix

sns.set_theme()


def plot_plate(
    adata: ad.AnnData,
    well_var: str = "Metadata_Well",
    color: Optional[str] = None,
    size: Optional[str] = None,
    select: Optional[dict] = None,
    wells: int = 96,
    cmap: str = "plasma",
    show: bool = False,
    save: Optional[str] = None,
    fname: str = "qc_plot.png",
    **kwargs,
) -> Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]:
    """Plot data of a 96-well or 384-well plate into a well-shaped plot.

    Can be used for quality control after (microscopy) experiments with
    96-well or 384-well plates.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated dataset with multidimensional morphological data.
        Annotations should include information about batch, plate and well.
    well_var : str
        Name of variable that contains wells
    color : str, optional
        Variable to use for color representations
    size : str, optional
        Variable to use for size representations
    select : dict, optional
        Masks data by annotations
    wells : int
        Select type of plate: `96` or `384`
    cmap : str
        Matplotlib colormap
    show : bool
        Show and return axes
    save : str, optional
        Path where to save figure
    fname : str
        Name of plat when saved
    **kwargs
        Keyword arguments are passed to matplotlib.pyplot.scatter

    Returns
    -------
    matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure if show is False and Axes

    Raises
    ------
    ValueError
        If `select` is not properly configured
    KeyError
        If `well_var` is not in `.obs`
    KeyError
        If `size` is not in `.obs`
    ValueError
        If wells are not unique
    ValueError
        If wells is neither `96` nor `384`
    OSError
        If figure can not be saved at specified path

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np
    >>> import pandas as pd

    >>> data = np.random.rand(4, 4)
    >>> obs = pd.DataFrame(
    >>>     {'well': ['A1', 'A2', 'A3', 'A4']}
    >>> )
    >>> adata = ad.AnnData(data, obs=obs)
    >>> mp.pl.plot_plate(adata, well_var='well', color='0')
    """
    # mask md
    # index data if select is not None
    if select is not None:
        try:
            qry = " and ".join(["{} == {}".format(k, v) for k, v in select.items()])
            qry_ix = adata.obs.query(qry).index
            adata = adata[qry_ix, :]
        except ValueError:
            print("Something wrong with select. Can not be used to query AnnData!")

    # check that well variables are unique
    if well_var not in adata.obs.columns:
        raise KeyError(f"Variable for wells are not found: {well_var}")
    if not adata.obs[well_var].is_unique:
        raise ValueError(
            "Values for Wells are not unique. Maybe use select to pass a single plate."
        )

    # default
    top = 1000
    bot = 50

    def char_range(c1, c2):
        """Generates the characters from `c1` to `c2`, inclusive."""
        for c in range(ord(c1), ord(c2) + 1):
            yield chr(c)

    # shape of plate
    if wells == 96:
        kwargs["x"] = list(range(1, 13)) * 8
        kwargs["y"] = sorted(list(range(1, 9)) * 12)
        well_cols = list(range(1, 13)) * 8
        well_rows = sorted(list(char_range("A", "H")) * 12)
        well_lst = [r + str(c) for r, c in list(zip(well_rows, well_cols))]
    elif wells == 384:
        kwargs["x"] = list(range(1, 25)) * 16
        kwargs["y"] = sorted(list(range(1, 17)) * 24)
        well_cols = list(range(1, 25)) * 16
        well_rows = sorted(list(char_range("A", "P")) * 24)
        well_lst = [r + str(c) for r, c in list(zip(well_rows, well_cols))]
        top = 250
        bot = 15
    else:
        raise ValueError(f"Well should be int and either 96 or 384: {wells}")

    # modify kwargs input
    kwargs.setdefault("s", [top] * len(kwargs["y"]))
    kwargs.setdefault("c", "white")
    kwargs.setdefault("edgecolor", ["black"] * len(kwargs["y"]))
    kwargs.setdefault("linewidths", 1.5)
    kwargs.setdefault("plotnonfinite", True)

    if color is not None:
        if color in adata.var_names.to_list():
            color_dict = dict(
                zip(adata.obs[well_var], adata[:, color].X.copy().flatten())
            )
            kwargs["c"] = [
                color_dict[well] if well in color_dict.keys() else np.nan
                for well in well_lst
            ]
        elif color in adata.obs.columns:
            color_dict = dict(zip(adata.obs[well_var], adata.obs[color]))
            kwargs["c"] = [
                color_dict[well] if well in color_dict.keys() else np.nan
                for well in well_lst
            ]
        else:
            raise ValueError(f"Color value not found: {color}.")

    if size is not None:
        if size in adata.var.index.to_list():
            size_dict = dict(
                zip(adata.obs[well_var], adata[:, size].X.copy().flatten())
            )
            size_arr = np.array(
                [
                    size_dict[well] if well in size_dict.keys() else np.nan
                    for well in well_lst
                ]
            )
        elif size in adata.obs.columns:
            size_dict = dict(zip(adata.obs[well_var], adata.obs[size]))
            size_arr = np.array(
                [
                    size_dict[well] if well in size_dict.keys() else np.nan
                    for well in well_lst
                ]
            )
        else:
            raise KeyError(f"Size value not found: {size}")
        points = np.interp(
            size_arr, (np.nanmin(size_arr), np.nanmax(size_arr)), (bot, top)
        )
        points = np.nan_to_num(points, nan=top)
        kwargs["s"] = points

    # nan colors
    cmap = mpl.colormaps.get_cmap(cmap)
    cmap.set_bad("lightgrey")
    kwargs["cmap"] = cmap

    # create figure
    with sns.axes_style("white"):
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot2grid((1, 1), (0, 0), fig=fig)
        top = 1000
        bot = 50
        font_size = 15
        ticks_font = font_manager.FontProperties(
            style="normal", size=font_size, weight="normal"
        )

        # plot
        mesh = ax.scatter(**kwargs)

        # make color bar
        if color is not None:
            cbar = plt.colorbar(mesh, fraction=0.046, pad=0.04)
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.set_ylabel(f"{color}", rotation=270)

        # make size legend
        if size is not None:
            poslab = 1.2 if color is not None else 1.05
            medv = ((np.nanmax(size_arr) - np.nanmin(size_arr)) / 2) + np.nanmin(
                size_arr
            )
            topl = f"{np.nanmax(size_arr):.2f}"
            botl = f"{np.nanmax(size_arr):.2f}"
            medl = f"{medv:.2f}"
            medv = ((max(points) - min(points)) / 2) + max(points)

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="lightgrey",
                    label=topl,
                    markeredgecolor="black",
                    markersize=math.sqrt(top),
                    lw=0,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="lightgrey",
                    label=medl,
                    markeredgecolor="black",
                    markersize=math.sqrt(medv),
                    lw=0,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="lightgrey",
                    label=botl,
                    markeredgecolor="black",
                    markersize=math.sqrt(bot),
                    lw=0,
                ),
            ]

            lbl_space = top / 300
            ax.legend(
                handles=legend_elements,
                labelspacing=lbl_space,
                handletextpad=2,
                borderpad=2,
                frameon=True,
                bbox_to_anchor=(poslab, 1),
                loc="upper left",
                title=size,
            )

        # image aspects
        ax.set_xticks(sorted(list(set(well_cols))))
        ax.xaxis.tick_top()
        ax.set_yticks(sorted(list(range(1, len(set(well_rows)) + 1))))
        ax.set_yticklabels(string.ascii_uppercase[0 : len(set(well_rows))])
        ax.set_ylim(((len(set(well_rows)) + 0.5), 0.48))
        ax.set_aspect(1)
        ax.tick_params(axis="both", which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for label in ax.get_xticklabels():
            label.set_fontproperties(ticks_font)

        for label in ax.get_yticklabels():
            label.set_fontproperties(ticks_font)

    # save
    if save is not None:
        try:
            plt.savefig(os.path.join(save, fname))
        except OSError:
            print(f"Can not save figure to {save}.")

    if show:
        plt.show()
        return ax

    return fig, ax


def plot_batch_effect(
    adata: ad.AnnData,
    batch_var: str = "BatchNumber",
    plate_var: str = "PlateNumber",
    control_var: str = "Metadata_Treatment",
    control_id: str = "ctrl",
    method: str = "pearson",
    show: bool = False,
    save: Optional[str] = None,
    **kwargs,
) -> Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]:
    """Plot batch effect using the correlation of
    control wells along different plates and batches.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    batch_var : str
        Name of annotation that holds information about batch number
    plate_var : str
        Name of annotation that holds information about plate number.
        If None, show only effect between batches.
    control_var : str
        Name of annotation that holds information about conditions
    control_id : str
        Name of control wells in control_var
    method : str
        Method for similarity/ distance computation.
        Should be one of: `pearson`, `spearman`, `kendall`, `euclidean`, `mahalanobis`.
    show : bool
        Show and return axes
    save : str, optional
        Path where to save figure
    **kwargs
        Keyword arguments passed to seaborn.heatmap

    Returns
    -------
    matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure if show is False and Axes

    Raises
    ------
    AssertionError
        If `batch_var`, `plate_var` or `control_var` is not in `.obs`
    AssertionError
        If `control_id` is not in `control_var`
    AssertionError
        If `method` is unknown
    OSError
        If figure can not be saved at specified path
    """
    # check variables
    assert batch_var in adata.obs.columns, f"batch_var not in annotations: {batch_var}"
    if plate_var is not None:
        assert (
            plate_var in adata.obs.columns
        ), f"plate_var not in annotations: {plate_var}"
    assert (
        control_var in adata.obs.columns
    ), f"control_var not in annotations: {control_var}"

    assert (
        control_id in adata.obs[control_var].tolist()
    ), f"control_id not in {control_var}: {control_id}"

    # check method
    avail_methods = [
        "pearson",
        "spearman",
        "kendall",
        "euclidean",
        "mahalanobis",
    ]
    method = method.lower()
    assert method in avail_methods, (
        f"method should be in {avail_methods}, " f"instead got {method}"
    )

    # select control conditions
    adata = adata[adata.obs[control_var] == control_id, :]

    # compute similarity matrix
    sim_df = dist_matrix(
        adata,
        method=method,
        group_var=batch_var,
        other_group_vars=plate_var,
        show=False,
    )

    # plot
    cmap = matplotlib.cm.plasma
    kwargs["cmap"] = cmap

    fig = plt.figure(figsize=(9, 7))
    ax = sns.heatmap(sim_df, **kwargs)

    # save
    if save is not None:
        try:
            plt.savefig(os.path.join(save, "qc_plot.png"))
        except OSError:
            print(f"Can not save figure to {save}.")

    if show:
        plt.show()
        return ax

    return fig, ax
