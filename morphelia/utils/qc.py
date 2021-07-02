# import internal libraries
import math
import string

# import external libraries
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import matplotlib
import numpy as np


def plot_qc(md, well_var="Metadata_Well", color=None, size=None, select=None, wells=96, **kwargs):
    """Plot data of a 96-well or 384-well plate into a well-shaped plot.
    
    Can be used for quality control after (microscopy) experiments with
    96-well or 384-well plates.
    
    Args:
        md (anndata.AnnData): Annotated dataset with multidimensional morphological data.
            Annotations should include information about batch, plate and well.
        well_var (str): Name of variable that contains well.
        color (str): Variable to use for color representations.
        size (str): Variable to use for size representations.
        select (dict): Masks data by annotations. 
        wells (int): Select type of plate: 96 or 384.
    
    Returns:
        matplotlib.pyplot.figure
    """
    # mask md
    # index data if select is not None
    if select is not None:
        try:
            qry = ' and '.join(["{} == {}".format(k, v) for k, v in select.items()])
            qry_ix = md.obs.query(qry).index
            md = md[qry_ix, :]
        except ValueError:
            print("Something wrong with select. Can not be used to query AnnData!")

    # check that well variables are unique
    if well_var not in md.obs.columns:
        raise ValueError(f"Variable for wells are not found: {well_var}")
    if not md.obs[well_var].is_unique:
        raise ValueError(f"Values for Wells are not unique. Maybe use select to pass a single plate.")

    # create figure
    fig = plt.figure(figsize=(15, 7))
    ax = plt.subplot2grid((1, 1), (0, 0), fig=fig)
    top = 1000
    bot = 50
    font_size = 15
    ticks_font = font_manager.FontProperties(style='normal', size=font_size, weight='normal')

    def char_range(c1, c2):
        """Generates the characters from `c1` to `c2`, inclusive."""
        for c in range(ord(c1), ord(c2) + 1):
            yield chr(c)

    # shape of plate
    if wells == 96:
        kwargs['x'] = list(range(1, 13)) * 8
        kwargs['y'] = sorted(list(range(1, 9)) * 12)
        well_cols = list(range(1, 13)) * 8
        well_rows = sorted(list(char_range('A', 'H')) * 12)
        well_lst = [r + str(c) for r, c in list(zip(well_rows, well_cols))]
    elif wells == 384:
        kwargs['x'] = list(range(1, 25)) * 16
        kwargs['y'] = sorted(list(range(1, 17)) * 24)
        well_cols = list(range(1, 25)) * 16
        well_rows = sorted(list(char_range('A', 'P')) * 24)
        well_lst = [r + str(c) for r, c in list(zip(well_rows, well_cols))]
        top = 250
        bot = 15
    else:
        raise ValueError(f"Well should be int and either 96 or 384: {wells}")

    # modify kwargs input
    kwargs.setdefault('s', [top, ] * len(kwargs['y']))
    kwargs.setdefault('c', 'white')
    kwargs.setdefault('edgecolor', ['black', ] * len(kwargs['y']))
    kwargs.setdefault('linewidths', 1.5)
    # kwargs.setdefault('cmap', 'plasma')
    kwargs.setdefault('plotnonfinite', True)

    if color is not None:
        if color not in md.var.index.to_list():
            raise ValueError(f"Color value not found: {color}.")
        color_dict = dict(zip(md.obs[well_var], md[:, color].X.copy().flatten()))
        kwargs['c'] = [color_dict[well] if well in color_dict.keys() else np.nan for well in well_lst]

    if size is not None:
        if size not in md.var.index.to_list():
            raise ValueError(f"Size value not found: {size}")
        size_dict = dict(zip(md.obs[well_var], md[:, size].X.copy().flatten()))
        size_arr = np.array([size_dict[well] if well in size_dict.keys() else np.nan for well in well_lst])
        points = np.interp(size_arr, (np.nanmin(size_arr), np.nanmax(size_arr)), (bot, top))
        points = np.nan_to_num(points, nan=top)
        kwargs['s'] = points

    # nan colors
    cmap = matplotlib.cm.plasma
    cmap.set_bad("lightgrey")
    kwargs['cmap'] = cmap

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
        medv = ((np.nanmax(size_arr) - np.nanmin(size_arr)) / 2) + np.nanmin(size_arr)
        topl = f"{np.nanmax(size_arr):.2f}"
        botl = f"{np.nanmax(size_arr):.2f}"
        medl = f"{medv:.2f}"
        medv = ((max(points) - min(points)) / 2) + max(points)

        legend_elements = [
            Line2D([0], [0], marker='o', color='lightgrey', label=topl,
                   markeredgecolor='black', markersize=math.sqrt(top), lw=0),
            Line2D([0], [0], marker='o', color='lightgrey', label=medl,
                   markeredgecolor='black', markersize=math.sqrt(medv), lw=0),
            Line2D([0], [0], marker='o', color='lightgrey', label=botl,
                   markeredgecolor='black', markersize=math.sqrt(bot), lw=0),
        ]

        lbl_space = top / 300
        ax.legend(handles=legend_elements, labelspacing=lbl_space,
                  handletextpad=2, borderpad=2, frameon=True,
                  bbox_to_anchor=(poslab, 1), loc='upper left',
                  title=size)

    # image aspects
    ax.set_xticks(sorted(list(set(well_cols))))
    ax.xaxis.tick_top()
    ax.set_yticks(sorted(list(range(1, len(set(well_rows)) + 1))))
    ax.set_yticklabels(string.ascii_uppercase[0:len(set(well_rows))])
    ax.set_ylim(((len(set(well_rows)) + 0.5), 0.48))
    ax.set_aspect(1)
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for label in ax.get_xticklabels():
        label.set_fontproperties(ticks_font)

    for label in ax.get_yticklabels():
        label.set_fontproperties(ticks_font)

    return fig, ax
