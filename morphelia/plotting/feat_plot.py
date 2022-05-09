# import external libraries
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad

from typing import Optional, Union, List

__all__ = ["boxplot", "violin", "barplot"]


def boxplot(
    adata: ad.AnnData,
    x: str,
    y: Union[str, List[str]],
    hue: Optional[str] = None,
    y_label: Optional[Union[str, List[str]]] = None,
    x_label: Optional[str] = None,
    **kwargs,
):
    """Boxplot.

    Convenient function to plot a boxplot directory from an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with multiple measurements for single objects over time
    x : str
        Variable name from `.obs`
    y : str or list of str, optional
        Name of variables to show
    hue : str, optional
        Variable name from `.obs`
    y_label : str or list of str, optional
        Labels for y axis
    x_label : str, optional
        Label for x axis

    Returns
    -------
    matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure and Axes

    Raises
    ------
    KeyError
        If `y` is not in `.obs` or `.var_names`

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np
    >>> import pandas as pd

    >>> data = np.random.rand(4, 4)
    >>> obs = pd.DataFrame(
    >>>     {'cell': [0, 0, 1, 1],
    >>>      'protein': [0, 1, 2, 3]}
    >>> )
    >>> adata = ad.AnnData(data, obs=obs)
    >>> mp.pl.boxplot(adata, x='cell', y='protein')
    """
    y_data = []
    if not isinstance(y, list):
        ys = []
        ys.append(y)
    else:
        ys = y
    if all(y in adata.obs.columns for y in ys):
        sub_arr = adata.obs[ys].to_numpy().reshape(-1, 1)
    elif all(y in adata.var_names for y in ys):
        sub_arr = adata[:, ys].X
    else:
        raise KeyError(f"Variable for time not in AnnData object: {ys}")
    # store y variables as vectors
    for column in sub_arr.T:
        y_data.append(column.flatten())

    # create figure
    flierprops = dict(marker="o", markersize=1)
    sns.set_theme(style="whitegrid")
    fig_height = int(4 * len(y_data))
    fig, axs = plt.subplots(len(y_data), sharex=True, figsize=(10, fig_height))

    # plot
    if len(y_data) == 1:
        sns.boxplot(
            x=x,
            y=y_data[0],
            hue=hue,
            data=adata.obs,
            ax=axs,
            flierprops=flierprops,
            **kwargs,
        )

        # set y label
        if y_label is not None:
            axs.set_ylabel(y_label)
        else:
            axs.set_ylabel(ys[0])
        axs.set_xlabel(x_label)

        if hue is not None:
            axs.legend()
    else:
        for ix in range(len(y_data)):
            sns.boxplot(
                x=x,
                y=y_data[ix],
                hue=hue,
                data=adata.obs,
                ax=axs[ix],
                flierprops=flierprops,
                **kwargs,
            )

            # set y label
            if y_label is not None:
                axs.set_ylabel(y_label[ix])
            else:
                axs[ix].set_ylabel(ys[ix])

            if hue is not None:
                axs[ix].legend()
            if ix < (len(y_data) - 1):
                axs[ix].set_xlabel("")
            elif x_label is not None:
                axs[ix].set_xlabel(x_label)

    plt.tight_layout()

    return fig, axs


def barplot(
    adata: ad.AnnData,
    x: str,
    y: Union[str, List[str]],
    hue: Optional[str] = None,
    y_label: Optional[Union[str, List[str]]] = None,
    x_label: Optional[str] = None,
    **kwargs,
):
    """Barplot.

    Convenient function to plot a barplot directory from an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with multiple measurements for single objects over time
    x : str
        Variable name from `.obs`
    y : str or list of str, optional
        Name of variables to show
    hue : str, optional
        Variable name from `.obs`
    y_label : str or list of str, optional
        Labels for y axis
    x_label : str, optional
        Label for x axis

    Returns
    -------
    matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure and Axes

    Raises
    ------
    KeyError
        If `y` is not in `.obs` or `.var_names`

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np
    >>> import pandas as pd

    >>> data = np.random.rand(4, 4)
    >>> obs = pd.DataFrame(
    >>>     {'cell': [0, 0, 1, 1],
    >>>      'protein': [0, 1, 2, 3]}
    >>> )
    >>> adata = ad.AnnData(data, obs=obs)
    >>> mp.pl.barplot(adata, x='cell', y='protein')
    """
    y_data = []
    if not isinstance(y, list):
        ys = []
        ys.append(y)
    else:
        ys = y
    if all(y in adata.obs.columns for y in ys):
        sub_arr = adata.obs[ys].to_numpy().reshape(-1, 1)
    elif all(y in adata.var_names for y in ys):
        sub_arr = adata[:, ys].X
    else:
        raise KeyError(f"Variable for time not in AnnData object: {ys}")
    # store y variables as vectors
    for column in sub_arr.T:
        y_data.append(column.flatten())

    # create figure
    sns.set_theme(style="whitegrid")
    fig_height = int(4 * len(y_data))
    fig, axs = plt.subplots(len(y_data), sharex=True, figsize=(10, fig_height))

    # plot
    if len(y_data) == 1:
        sns.barplot(
            x=x,
            y=y_data[0],
            hue=hue,
            data=adata.obs,
            ax=axs,
            capsize=0.2,
            **kwargs,
        )

        # set y label
        if y_label is not None:
            axs.set_ylabel(y_label)
        else:
            axs.set_ylabel(ys[0])
        axs.set_xlabel(x_label)

        if hue is not None:
            axs.legend()
    else:
        for ix in range(len(y_data)):
            sns.barplot(
                x=x,
                y=y_data[ix],
                hue=hue,
                data=adata.obs,
                ax=axs[ix],
                capsize=0.2,
                **kwargs,
            )

            # set y label
            if y_label is not None:
                axs.set_ylabel(y_label[ix])
            else:
                axs[ix].set_ylabel(ys[ix])

            if hue is not None:
                axs[ix].legend()
            if ix < (len(y_data) - 1):
                axs[ix].set_xlabel("")
            elif x_label is not None:
                axs[ix].set_xlabel(x_label)

    plt.tight_layout()

    return fig, axs


def violin(
    adata: ad.AnnData,
    x: str,
    y: Union[str, List[str]],
    hue: Optional[str] = None,
    jitter: bool = False,
    y_label: Optional[Union[str, List[str]]] = None,
    x_label: Optional[str] = None,
    **kwargs,
):
    """Violin Plot.

    Convenient function to plot a violin plot directory from an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with multiple measurements for single objects over time
    x : str
        Variable name from `.obs`
    y : str or list of str, optional
        Name of variables to show
    hue : str, optional
        Variable name from `.obs`
    jitter : bool
        Draw strips of observations
    y_label : str or list of str, optional
        Labels for y axis
    x_label : str, optional
        Label for x axis

    Returns
    -------
    matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure and Axes

    Raises
    ------
    KeyError
        If `y` is not in `.obs` or `.var_names`

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np
    >>> import pandas as pd

    >>> data = np.random.rand(4, 4)
    >>> obs = pd.DataFrame(
    >>>     {'cell': [0, 0, 1, 1],
    >>>      'protein': [0, 1, 2, 3]}
    >>> )
    >>> adata = ad.AnnData(data, obs=obs)
    >>> mp.pl.violin(adata, x='cell', y='protein')
    """
    # store y variables as vectors
    y_data = []
    if not isinstance(y, list):
        ys = []
        ys.append(y)
    else:
        ys = y
    if all(y in adata.obs.columns for y in ys):
        sub_arr = adata.obs[ys].to_numpy().reshape(-1, 1)
    elif all(y in adata.var_names for y in ys):
        sub_arr = adata[:, ys].X
    else:
        raise KeyError(f"Variable for time not in AnnData object: f{ys}")
    # store y variables as vectors
    for column in sub_arr.T:
        y_data.append(column.flatten())

    # get unique hue
    legend_len = 0
    if hue is not None:
        legend_len = len(adata.obs[hue].unique())

    # create figure
    inner = "box"
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
        sns.violinplot(
            x=x,
            y=y_data[0],
            hue=hue,
            data=adata.obs,
            ax=axs,
            inner=inner,
            **kwargs,
        )
        if jitter:
            sns.stripplot(
                x=x,
                y=y_data[0],
                hue=hue,
                data=adata.obs,
                ax=axs,
                jitter=jitter,
                color="gray",
                edgecolor="gray",
                size=1,
            )

        # set y label
        if y_label is not None:
            axs.set_ylabel(y_label)
        else:
            axs.set_ylabel(ys[0])
        axs.set_xlabel(x_label)

        if hue is not None:
            handles, labels = axs.get_legend_handles_labels()
            axs.legend(handles=handles[:legend_len], labels=labels[:legend_len])
    else:
        for ix in range(len(y_data)):
            sns.violinplot(
                x=x,
                y=y_data[ix],
                hue=hue,
                data=adata.obs,
                ax=axs[ix],
                inner=inner,
                **kwargs,
            )
            if jitter:
                sns.stripplot(
                    x=x,
                    y=y_data[ix],
                    hue=hue,
                    data=adata.obs,
                    ax=axs[ix],
                    jitter=jitter,
                    color="gray",
                    edgecolor="gray",
                    size=1,
                )

            # set y label
            if y_label is not None:
                axs.set_ylabel(y_label[ix])
            else:
                axs[ix].set_ylabel(ys[ix])

            if hue is not None:
                handles, labels = axs[ix].get_legend_handles_labels()
                axs[ix].legend(handles=handles[:legend_len], labels=labels[:legend_len])
            if ix < (len(y_data) - 1):
                axs[ix].set_xlabel("")
            elif x_label is not None:
                axs[ix].set_xlabel(x_label)

    plt.tight_layout()

    return fig, axs
