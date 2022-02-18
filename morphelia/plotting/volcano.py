import math
import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def volcano_plot(adata,
                 sign_line=True,
                 lfc_thr=None,
                 pv_thr=None,
                 color=None,
                 xlim=None,
                 ylim=None,
                 null_color=None,
                 show=False,
                 save=None):
    """
    Volcano plot.
    Use scanpy.tl.rank_genes_groups beforehand.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        sign_line (bool): Plot significant thresholds as lines.
        lfc_thr (tuple, list): Left and right threshold for log-fold-change.
        pv_thr (tuple, list): Left and right threshold for the p-values.
        color (tuple, list): Left and right color for significant values.
        xlim (tuple, list): Left and right limit of x axis.
        ylim (tuple, list): Left and right limit of y axis.
        null_color (str): Color for insignificant values.
        show (bool): Show plot and return None.
        save (str): Path where to save the plot as 'volcano.png'

    Returns:
        fig
    """
    sc_key = 'rank_genes_groups'
    assert (
        sc_key in adata.uns
    ), f'Key {sc_key} not in .uns. Use scanpy.tl.rank_genes_groups beforehand.'

    # get number of groups
    groups = list(adata.obs[adata.uns[sc_key]['params']['groupby']].unique())
    ref = adata.uns[sc_key]['params']['reference']
    if ref in groups:
        groups.remove(ref)
    print(groups)

    # parameters
    if lfc_thr is None:
        lfc_thr = (-1, 1)
    assert len(lfc_thr) == 2, f'two fold change thresholds expected, got {len(lfc_thr)}'
    if pv_thr is None:
        pv_thr = (0.05, 0.05)
    assert len(pv_thr) == 2, f'two p-value thresholds expected, got {len(pv_thr)}'
    if color is None:
        color = ('#D9534F', '#96CEB4')
    assert len(color) == 2, f'two color values expected, got {len(color)}'
    down_color, up_color = color[0], color[1]
    if null_color is None:
        null_color = '#D0CAB2'
    clset = {'significant up': up_color, 'significant down': down_color,
             'not significant': null_color}

    pvals = adata.uns[sc_key]['pvals_adj']
    logfoldchanges = adata.uns[sc_key]['logfoldchanges']
    # names = adata.uns[sc_key]['names']
    # scores = adata.uns[sc_key]['scores']

    # plot
    n_cols = 2
    n_rows = math.ceil(len(groups) / 2)
    scatter_size = 10000 / len(adata.var_names)
    plot_iter = list(itertools.product(range(n_rows), range(n_cols)))
    fig = plt.figure(constrained_layout=False, figsize=(4*n_cols, 4*n_rows))
    gs = plt.GridSpec(ncols=n_cols, nrows=n_rows, figure=fig)

    for ix, group in enumerate(groups):
        ax = plt.subplot(gs[plot_iter[ix]])
        lfc = logfoldchanges[group]
        pv = pvals[group]
        c = []
        for i in range(len(lfc)):
            if (lfc[i] < lfc_thr[0]) and (pv[i] < pv_thr[0]):
                c.append(down_color)
            elif (lfc[i] > lfc_thr[1]) and (pv[i] < pv_thr[1]):
                c.append(up_color)
            else:
                c.append(null_color)
        ax.scatter(lfc, -np.log10(pv), color=c, s=scatter_size)
        if sign_line:
            ax.axvline(lfc_thr[0], linestyle=':', color='k')
            ax.axvline(lfc_thr[1], linestyle=':', color='k')
            ax.axhline(-np.log10(pv_thr[0]), linestyle=':', color='k')
            ax.axhline(-np.log10(pv_thr[1]), linestyle=':', color='k')
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.set_title(group)
        ax.set_xlabel("$log_2$(FoldChange)")
        ax.set_ylabel("$-log_{10}$(p-value)")

    handles = [patches.Circle((0, 0), 1, fc=handle_color) for handle_color in clset.values()]
    labels = clset.keys()
    plt.legend(handles=handles, labels=labels,
               bbox_to_anchor=(1.05, 0.95), loc='upper left', frameon=False)

    if save is not None:
        if not os.path.exists(save):
            raise OSError(f"Path does not exist: {save}")
        fig.savefig(os.path.join(save, 'volcano.png'), dpi=fig.dpi)

    if show:
        plt.show()
        return None

    return fig
