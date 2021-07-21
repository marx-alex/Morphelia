# import external libraries
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ["boxplot", "violin"]


def boxplot(adata, x, y, hue=None, **kwargs):
    """Plot one or more variables of single or
    aggregated cells for different time points.

    Args:
        adata (anndata.AnnData): Annotated data matrix with multiple measurements for single objects over time.
        x (str): Variable name from md.obs.
        y (str of list of str): Name of variables to show.
        hue (str): Variable name from md.obs.

    Returns:
        matplotlib.pyplot.figure
    """
    # store y variables as vectors
    y_data = []
    sub_arr = adata[:, y].X
    for column in sub_arr.T:
        y_data.append(column.flatten())

    # create figure
    flierprops = dict(marker='o', markersize=1)
    sns.set_theme(style="whitegrid")
    fig_height = int(4 * len(y_data))
    fig, axs = plt.subplots(len(y_data), sharex=True, figsize=(10, fig_height))

    # plot
    if len(y_data) == 1:
        sns.boxplot(x=x, y=y_data[0], hue=hue, data=adata.obs, ax=axs, flierprops=flierprops, **kwargs)
        axs.set_ylabel(y)
        if hue is not None:
            axs.legend(loc='upper right')
    else:
        for ix in range(len(y_data)):
            sns.boxplot(x=x, y=y_data[ix], hue=hue, data=adata.obs, ax=axs[ix], flierprops=flierprops, **kwargs)
            axs[ix].set_ylabel(y[ix])
            if hue is not None:
                axs[ix].legend(loc='upper right')

    plt.tight_layout()

    return fig, axs


def violin(adata, x, y, hue=None, jitter=False, **kwargs):
    """Plot one or more variables of single or
    aggregated cells for different time points.

    Args:
        adata (anndata.AnnData): Annotated data matrix with multiple measurements for single objects over time.
        x (str): Variable name from md.obs.
        y (str of list of str): Name of variables to show.
        hue (str): Variable name from md.obs.
        jitter (bool): Draw strips of observations.

    Returns:
        matplotlib.pyplot.figure
    """
    # store y variables as vectors
    y_data = []
    sub_arr = adata[:, y].X.copy()
    for column in sub_arr.T:
        y_data.append(column.flatten())

    # get unique hue
    legend_len = 0
    if hue is not None:
        legend_len = len(adata.obs[hue].unique())

    # create figure
    inner = 'box'
    if jitter:
        inner = None
    sns.set_theme(style="whitegrid")
    # set height of figure
    fig_height = int(3 * len(y_data))
    if len(y_data) == 1:
        fig_height = 5
    fig, axs = plt.subplots(len(y_data), sharex=True, figsize=(10, fig_height))

    # plot
    if len(y_data) == 1:
        sns.violinplot(x=x, y=y_data[0], hue=hue, data=adata.obs, ax=axs, inner=inner, **kwargs)
        if jitter:
            sns.stripplot(x=x, y=y_data[0], hue=hue, data=adata.obs, ax=axs, jitter=jitter,
                          color='gray', edgecolor='gray', size=1)
        axs.set_ylabel(y)
        if hue is not None:
            handles, labels = axs.get_legend_handles_labels()
            axs.legend(loc='upper right', handles=handles[:legend_len], labels=labels[:legend_len])
    else:
        for ix in range(len(y_data)):
            sns.violinplot(x=x, y=y_data[ix], hue=hue, data=adata.obs, ax=axs[ix], inner=inner, **kwargs)
            if jitter:
                sns.stripplot(x=x, y=y_data[ix], hue=hue, data=adata.obs, ax=axs[ix], jitter=jitter,
                              color='gray', edgecolor='gray', size=1)
            axs[ix].set_ylabel(y[ix])
            if hue is not None:
                handles, labels = axs[ix].get_legend_handles_labels()
                axs[ix].legend(loc='upper right', handles=handles[:legend_len], labels=labels[:legend_len])

    plt.tight_layout()

    return fig, axs