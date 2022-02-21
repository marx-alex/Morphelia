# import internal libraries
import math
import os

# import external libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad


def temporal_reduction(
    md,
    trace_var="Metadata_Trace_Parent",
    time_var="Metadata_Time",
    show=None,
    vmin=0,
    vmax=50,
    save=None,
    meta_vars=(
        "BatchNumber",
        "PlateNumber",
        "Metadata_Well",
        "Metadata_Field",
    ),
):
    """Takes an AnnData object with data from different time points
    and a variable that indicates the links between objects over time.

    The function reduces measurements from all time points to single meaningful values.

    Args:
        md (anndata.AnnData): Annotated data matrix with multiple measurements for single objects over time.
        trace_var (str): Name of variable that stores the index of lagged objects.
        time_var(str): Name of variable with time points.
        show (list dict): Show chart of object variables.
        vmin, vmax (int): Minimum and maximum variables to show. Used for slicing.
        save (str): Path to save figure if show is not None.
        meta_vars (list): Variables that point to a single field of view.

    Returns:
        anndata.AnnData
    """
    # check that time_var and trace_var are in anndata object
    if not all(var in md.obs.columns for var in [time_var, trace_var]):
        raise KeyError(
            f"Assert that variables for time and trace ids are also in"
            f" AnnData annotations: {time_var}, {trace_var}"
        )

    # get list of time points
    tp = md.obs[time_var].unique().tolist()

    # get object indices from last time point
    last_trace = md.obs[md.obs[time_var] == md.obs[time_var].max()]
    trace_ix = last_trace.index.to_numpy()
    # cache information about object to show
    if show is not None:
        if not isinstance(show, int):
            raise TypeError(
                "Index of a single object is needed to show variables."
                f"Type must be integer: {type(show)}"
            )
        show_ix = trace_ix[show]
        show_info = md.obs.loc[show_ix, list(meta_vars)].to_dict()

    parent_ix = np.copy(trace_ix)
    trace_ix = trace_ix.reshape(-1, 1)
    # iterate over time points
    for _ in tp[:-1]:
        parent_ix = md.obs.loc[parent_ix, trace_var].to_numpy()
        trace_ix = np.hstack((parent_ix.reshape(-1, 1), trace_ix))

    # convert indices to int
    trace_ix = trace_ix.astype(int)
    # get a temporal restructured X
    T = np.take(
        md.X, trace_ix, axis=0
    )  # three-dimensional np.array (objects x time points x variables)
    # store time points as array
    tp = np.asarray(tp)

    #####################
    # This part is for visualization purposes only!
    #####################

    # show chart with variables of a single object
    if show is not None:
        fig, axes = _show_time_series(
            T[show, :, vmin:vmax],
            tp,
            show_info,
            var_names=list(md.var.index)[vmin:vmax],
        )

        # save if save is not None
        if save is not None:
            if not os.path.exists(save):
                raise OSError(f"Path does not exist: {save}")
            fig.savefig(save, dpi=fig.dpi)

    #####################
    # steps for dimensionality reduction
    #####################

    # calculate area under curve for time axis
    auc = np.trapz(T, tp, axis=1)

    # estimate the slope with least squares regression
    slope = (
        (tp[np.newaxis, :, np.newaxis] * T).mean(axis=1) - tp.mean() * T.mean(axis=1)
    ) / ((tp ** 2).mean() - (tp.mean()) ** 2)

    # create new anndata object
    new_ad = ad.AnnData(X=np.multiply(auc, slope), obs=last_trace, var=md.var)

    return new_ad


def _show_time_series(X, time_points, show_info, var_names):
    """Returns figures for time series variables of a object.

    Args:
        X (np.array): Two dimensional array (time x variables).
        time_points (np.array): One dimensional array with unique time points.
        show_info (dict): Information about object to plot.
        var_names (list): List of variable names

    Returns:
        matplotlib.pyplot.figure
    """
    # get shape of grid for plotting
    size = X.shape[-1]
    cols = 5
    rows = int(math.ceil(size / cols))

    sns.set()

    # plot multiple charts for every variable
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 2))
    fig.suptitle(f"{', '.join([f'{val}: {key}' for val, key in show_info.items()])}")

    # set options for plotting
    options = {"alpha": 0.9, "linewidth": 1.9, "color": "firebrick"}

    # iterate over variables
    for row in range(rows):
        for col in range(cols):
            v = (((row - 1) * cols) + col) - 1
            # if row == 0 and col == 0:
            #     new_plot(X, v, time_points, options, var_names)
            axes[row, col].plot(time_points, X[..., v], **options)
            axes[row, col].set_title(f"{var_names[v]}")
            axes[row, col].set_xlabel("Time")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, axes


# def new_plot(X, v, time_points, options, var_names):
#
#     fig1, axes1 = plt.subplots(1, 1, figsize=(15, 7))
#     axes1.fill_between(time_points, X[..., v], **options)
#
#     coef = np.polyfit(time_points, X[..., v], 1)
#     poly1d_fn = np.poly1d(coef)
#
#     axes1.plot(time_points, poly1d_fn(time_points), '--k')
#
#     axes1.annotate("Linear regression", xy=(21, 2900))
#     axes1.annotate("AUC", xy=(11, 900))
#
#     axes1.set_title(f"{var_names[v]}")
#     axes1.set_xlabel("Time")
#
#     return None
