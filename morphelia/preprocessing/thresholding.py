from collections import OrderedDict
from collections.abc import Iterable
import os
from typing import Union, List, Optional, Tuple

import numpy as np
import seaborn as sns
import anndata as ad
import matplotlib.pyplot as plt
from skimage.filters import (
    threshold_li,
    threshold_multiotsu,
    threshold_otsu,
    threshold_isodata,
    threshold_mean,
    threshold_minimum,
    threshold_triangle,
    threshold_yen,
)

# all skimage threshold algorithms
methods = OrderedDict(
    {
        "isodata": threshold_isodata,
        "li": threshold_li,
        "mean": threshold_mean,
        "minimum": threshold_minimum,
        "otsu": threshold_otsu,
        "triangle": threshold_triangle,
        "yen": threshold_yen,
        "multi_otsu": threshold_multiotsu,
    }
)


def assign_by_threshold(
    adata: ad.AnnData,
    dist: Union[np.ndarray, str],
    by: Union[List[str], str] = None,
    new_var: str = "Thresh_Assigned",
    method: str = "otsu",
    max_val: Optional[Union[int, float]] = None,
    min_val: Optional[Union[int, float]] = None,
    subsample: bool = False,
    sample_size: int = 10000,
    seed: int = 0,
    make_plot: bool = False,
    show: bool = True,
    return_fig: bool = False,
    save: Optional[str] = None,
    xlabel: Optional[str] = None,
    plt_xlim: Optional[Tuple[float, int]] = None,
    threshold_colors: Optional[Union[str, List[str]]] = None,
    threshold_labels: Optional[Union[str, List[str]]] = None,
    class_colors: Optional[Union[str, List[str]]] = None,
    class_labels: Optional[Union[str, List[str]]] = None,
    plt_kwargs: dict = None,
    **kwargs,
):
    """Assign new population labels by thresholding.

    Given a distribution (a single feature vector), this functions finds
    sub-distributions by applying thesholding methods.
    Subpopulations can be labeled and plotted.

    Parameters
    ----------
    adata : anndata.AnnData)
        Multidimensional morphological data
    dist : np.ndarray or str
        Distribution to find thresholds for. Can be array-like, a variable or
        an observation in the anndata object
    by : str or list of str
        Find thresholds by given groups. Should be observation variables in `.obs`
    new_var : str
        Name of new variable
    method : str
        Thresholding method. Can be one of: `isodata`, `li`, `mean`, `minimum`, `otsu`, `triabgle`, `yen`, `multi_otsu`.
        Scikit-image thesholding methods are used.
    max_val : int or float, optional
        Maximum value to consider to be a threshold
    min_val : int or float, optional
        Minimum value to consider to be a threshold
    subsample : bool
        Make a subsample before calculating threshold
    sample_size : int
        Size of subsample
    seed : int
        Seed for reproducibility
    make_plot : bool
        Make plots for each threshold calculation
    show : bool, optional
        Show plot. Return figure, if false
    return_fig : bool
        Return anndata object and figure
    save : str, optional
        Path, where to store figure
    xlabel : str, optional
        Label for x-axis in plot
    plt_xlim : tuple, optional
        Limits for x-axis.
    threshold_colors : list of str or str, optional
        Colors to use for threshold lines in plot
    threshold_labels : list or str or str, optional
        Labels for threshold lines in plot
    class_colors : list of str or str, optional
        Class-specific colors to use for histogram patches in plot
    class_labels : list of str or str, optional
        Class-specific labels. Integer labels are mapped to the given ones
    plt_kwargs : dict
        Keyword arguments that are passed to seaborn.histplot
    **kwargs
        Keyword arguments that are passed to the scikit-image threshold function

    Returns
    -------
    anndata.AnnData
        AnnData object with new population labels stores in `.obs`

    Raises
    -------
    AssertionError
        If any variable in `by` is not in `.obs`
    AssertionError
        If `class_labels` do not match new labels

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np

    >>> data = np.random.rand(10, 5)
    >>> data[:5, 0] = data[:5, 0] + 2  # define new population
    >>> adata = ad.AnnData(data)

    >>> adata = mp.pp.assign_by_threshold(
    >>>     adata,
    >>>     dist='0'
    >>> )

    >>> adata.obs  # first five instances are correctly assigned
        Thresh_Assigned
    0	1.0
    1	1.0
    2	1.0
    3	1.0
    4	1.0
    5	0.0
    6	0.0
    7	0.0
    8	0.0
    9	0.0
    """
    # dist can be a variable, observation or array
    if isinstance(dist, str):
        if dist in adata.obs.columns:
            dist = adata.obs[dist].to_numpy()
        elif dist in adata.var_names:
            dist = adata[:, dist].X
            dist = np.asarray(dist)

    if plt_kwargs is None:
        plt_kwargs = {}

    # check by
    if by is not None:
        if isinstance(by, str):
            by = [by]
        assert isinstance(by, Iterable), "by is not iterable"
        assert all(
            var in adata.obs.columns for var in by
        ), f"Variables defined in 'by' are not in annotations: {by}"

    method = method.lower()
    assert (
        method in methods
    ), f"Given method unknown: {method}. Use one of: {methods.keys()}."

    func = methods[method]

    adata.obs[new_var] = np.nan
    fig = None

    if by is not None:
        fig = []
        for groups, sub_df in adata.obs.groupby(by):
            group_mask = np.in1d(adata.obs.index, sub_df.index)
            group_dist = dist[group_mask]

            thresh, class_annotation = _find_thresh(
                dist=group_dist,
                func=func,
                min_val=min_val,
                max_val=max_val,
                sample_size=sample_size,
                subsample=subsample,
                seed=seed,
                **kwargs,
            )

            adata.obs.loc[sub_df.index, new_var] = class_annotation

            # plot
            if make_plot:
                if isinstance(groups, (list, tuple)):
                    group_str = ", ".join([f"{k}: {g}" for k, g in zip(by, groups)])
                else:
                    group_str = str(groups)
                f = _plot_thresh(
                    group_dist,
                    thresh,
                    xlabel,
                    threshold_colors,
                    threshold_labels,
                    class_colors,
                    class_labels,
                    plt_xlim,
                    group_str,
                    save,
                    show,
                    **plt_kwargs,
                )
                fig.append(f)

    else:
        thresh, class_annotation = _find_thresh(
            dist=dist,
            func=func,
            min_val=min_val,
            max_val=max_val,
            sample_size=sample_size,
            subsample=subsample,
            seed=seed,
            **kwargs,
        )

        # add annotation to adata
        adata.obs[new_var] = class_annotation

        # plot
        if make_plot:
            fig = _plot_thresh(
                dist,
                thresh,
                xlabel,
                threshold_colors,
                threshold_labels,
                class_colors,
                class_labels,
                plt_xlim,
                None,
                save,
                show,
                **plt_kwargs,
            )

    if class_labels is not None:
        if isinstance(class_labels, str):
            class_labels = [class_labels]
        unique_labels = adata.obs[new_var].unique()
        unique_labels = np.sort(unique_labels)
        assert len(class_labels) == len(
            unique_labels
        ), f"{len(class_labels)} class_labels given, but {len(unique_labels)} unique labels assigned."

        class_mapping = {ul: cl for ul, cl in zip(unique_labels, class_labels)}
        adata.obs[new_var] = adata.obs[new_var].map(class_mapping)

    if return_fig:
        return adata, fig

    return adata


def _find_thresh(
    dist,
    func,
    subsample=False,
    sample_size=10000,
    seed=0,
    min_val=None,
    max_val=None,
    **kwargs,
):
    """Finds threshold in a given distribution by a given function and
    with given constraints. Thresholds and class annotations are returned."""
    dist_window = dist.copy()
    if max_val is not None:
        dist_window = dist_window[dist_window < max_val]
    if min_val is not None:
        dist_window = dist_window[dist_window > min_val]

    if subsample and sample_size is not None:
        # get samples
        np.random.seed(seed)
        N = len(dist_window)
        if sample_size >= N:
            pass
        else:
            sample_ixs = np.random.choice(N, size=sample_size, replace=False)
            dist_window = dist_window[sample_ixs]

    thresh = func(dist_window, **kwargs)

    # make threshold iterable like multiclass thresholds
    thresh = np.asarray(thresh)
    if thresh.ndim == 0:
        thresh = thresh.reshape(-1)

    # annotate classes by threshold
    thresh = np.sort(thresh)
    class_annotation = np.zeros(dist.shape)
    for i, t in enumerate(thresh):
        class_annotation[dist > t] = i + 1

    return thresh, class_annotation


def _plot_thresh(
    dist,
    thresh,
    xlabel=None,
    threshold_colors=None,
    threshold_labels=None,
    class_colors=None,
    class_labels=None,
    plt_xlim=None,
    title=None,
    save=None,
    show=True,
    **kwargs,
):
    kwargs.setdefault("kde", True)
    kwargs.setdefault("stat", "density")
    kwargs.setdefault("linewidth", 0)

    fig, ax = plt.subplots()
    p = sns.histplot(dist, ax=ax, **kwargs)

    if class_colors is not None:
        if isinstance(class_colors, str):
            class_colors = [class_colors]
        assert (
            len(class_colors) == len(thresh) + 1
        ), f"{len(class_colors)} class_colors provided, but {len(thresh)} thresholds given."
        if class_labels is not None:
            if isinstance(class_labels, str):
                class_labels = [class_labels]

        nearest_rect = []
        for t in thresh:
            thresh_bins = [np.abs(patch.get_x() - t) for patch in p.patches]
            nearest_rect.append(np.argmin(thresh_bins))

        label_i = 0
        for rect_i, rectangle in enumerate(p.patches):
            if rect_i == 0:
                if class_labels is not None:
                    rectangle.set_label(class_labels[label_i])
            if label_i < len(thresh):
                if rect_i < nearest_rect[label_i]:
                    rectangle.set_facecolor(class_colors[label_i])
                else:
                    label_i += 1
                    rectangle.set_facecolor(class_colors[label_i])
                    if class_labels is not None:
                        rectangle.set_label(class_labels[label_i])
            else:
                rectangle.set_facecolor(class_colors[label_i])

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)
    if threshold_colors is None:
        threshold_colors = ["k" for _ in thresh]
    elif isinstance(threshold_colors, str):
        threshold_colors = [threshold_colors]
    if threshold_labels is None:
        threshold_labels = [None for _ in thresh]
    elif isinstance(threshold_labels, str):
        threshold_labels = [threshold_labels]

    for t, c, l in zip(thresh, threshold_colors, threshold_labels):
        ax.axvline(t, color=c, linestyle="dotted", label=l)

    if plt_xlim is None:
        plt_xlim = (None, None)
    ax.set_xlim(plt_xlim[0], plt_xlim[1])
    plt.legend()

    # save
    if save is not None:
        fname = "threshold"
        i = 0
        while os.path.isfile(os.path.join(save, f"{fname}.png")):
            fname = fname.split("_")
            fname = f"{fname[0]}_{i}"
            i += 1
        try:
            plt.savefig(os.path.join(save, f"{fname}.png"))
        except OSError:
            print(f"Can not save figure to {save}.")

    if show:
        plt.show()

    return fig
