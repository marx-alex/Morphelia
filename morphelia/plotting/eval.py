import os
from typing import Union, Optional, Tuple

import numpy as np
import seaborn as sns
import anndata as ad
import matplotlib.pyplot as plt


def plot_eval(
    adata: ad.AnnData,
    whisker_size: Union[float, int] = 0.005,
    linewidth: Union[float, int] = 4,
    vmin: Union[float, int] = 0.5,
    vmax: Union[float, int] = 1.0,
    c_repro: str = "#C84B31",
    c_effect: str = "#346751",
    show_outlier: bool = False,
    show: bool = False,
    save: Optional[str] = None,
) -> Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]:
    """Plot reproducibility and effect if calculated before.

    The lines show the interquartile range for both measures, the intersections show medians.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    whisker_size : float or int
        Size of whiskers
    linewidth : float or int
        Width of lines
    vmin : float or int
        Minimum value for x and y axis
    vmax : float or int
        Maximum value for x and y axis
    c_repro : str
        Color for reproducibility
    c_effect : str
        Color for effect
    show_outlier : bool
        Plot outlier values
    show : bool
        Show and return axes
    save : str, optional
        Path where to save figure

    Returns
    -------
    matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure if show is False and Axes

    Raises
    ------
    AssertionError
        If reproducibility and effect have not been calculated before
    OSError
        If figure can not be saved at specified path
    """
    out = (
        "Reproducibility and effect no yet calculated. Use morphelia.eval.repro_effect"
    )
    assert "eval" in adata.uns, out
    assert "repro" in adata.uns["eval"], out
    assert "effect" in adata.uns["eval"], out

    sns.set_theme(style="darkgrid")
    repro = adata.uns["eval"]["repro"]["percentiles"].to_numpy()
    repro_q1 = np.quantile(repro, 0.25)
    repro_q2 = np.quantile(repro, 0.5)
    repro_q3 = np.quantile(repro, 0.75)
    effect = adata.uns["eval"]["effect"]["percentiles"].to_numpy()
    effect_q1 = np.quantile(effect, 0.25)
    effect_q2 = np.quantile(effect, 0.5)
    effect_q3 = np.quantile(effect, 0.75)

    # points
    fig, axs = plt.subplots(figsize=(7, 7))
    plt.plot(
        [repro_q1, repro_q3],
        [effect_q2, effect_q2],
        color=c_repro,
        linewidth=linewidth,
    )
    plt.plot(
        [repro_q2, repro_q2],
        [effect_q1, effect_q3],
        color=c_effect,
        linewidth=linewidth,
    )
    # plot whisker
    plt.plot(
        [repro_q1, repro_q1],
        [effect_q2 - whisker_size, effect_q2 + whisker_size],
        color=c_repro,
        linewidth=linewidth,
    )
    plt.plot(
        [repro_q3, repro_q3],
        [effect_q2 - whisker_size, effect_q2 + whisker_size],
        color=c_repro,
        linewidth=linewidth,
    )
    plt.plot(
        [repro_q2 - whisker_size, repro_q2 + whisker_size],
        [effect_q1, effect_q1],
        color=c_effect,
        linewidth=linewidth,
    )
    plt.plot(
        [repro_q2 - whisker_size, repro_q2 + whisker_size],
        [effect_q3, effect_q3],
        color=c_effect,
        linewidth=linewidth,
    )

    # outlier
    if show_outlier:
        repro_out = repro[np.logical_or(repro < repro_q1, repro > repro_q3)]
        effect_out = effect[np.logical_or(effect < effect_q1, effect > effect_q3)]
        plt.scatter(
            repro_out,
            [effect_q2] * len(repro_out),
            color="darkgrey",
            alpha=0.6,
        )
        plt.scatter(
            [repro_q2] * len(effect_out),
            effect_out,
            color="darkgrey",
            alpha=0.6,
        )

    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)

    # labels
    plt.xlabel("Reproducibility", fontsize=14)
    plt.ylabel("Effect", fontsize=14)

    # save
    if save is not None:
        try:
            plt.savefig(os.path.join(save, "eval.png"))
        except OSError:
            print(f"Can not save figure to {save}.")

    if show:
        plt.show()
        return axs

    return fig, axs
