# import internal libraries
import os

# import external libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def pca_variance_ratio(adata,
                       n_pcs=100,
                       show=False,
                       save=False,
                       **kwargs):
    """Plots cumulative percentage variance explained by each principal component.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        n_pcs (int): Number of PCAs to plot.
        show (bool): Show and return axes.
        save (str): Path where to save figure.
        kwargs (dict): Keyword passed to matplotlib.pyplot.plot
    """
    if isinstance(n_pcs, int):
        if n_pcs > adata.shape[1]:
            n_pcs = adata.shape[1]
        var_ratio = getattr(adata, 'uns')['pca']['variance_ratio'][:n_pcs]
    elif n_pcs is None:
        var_ratio = getattr(adata, 'uns')['pca']['variance_ratio']
    else:
        raise TypeError(f"n_pcas expected to be int or None, instead got {type(n_pcs)}")

    # get cumulative scores
    var_ratio = np.cumsum(var_ratio) * 100

    # get pcas for x_axis
    pca_ixs = range(1, len(var_ratio)+1)

    # plot
    kwargs.setdefault('color', '#D66853')
    # kwargs.setdefault('marker', 'o')

    sns.set_theme()
    fig, ax = plt.subplots()
    ax.plot(pca_ixs, var_ratio, **kwargs)
    ax.set_ylabel('percentage explained variance')
    ax.set_xlabel('number of principal components')

    # save
    if save:
        try:
            plt.savefig(os.path.join(save, "pca_variance_ratio.png"))
        except OSError:
            print(f'Can not save figure to {save}.')

    if show:
        plt.show()
        return ax

    return fig, ax
