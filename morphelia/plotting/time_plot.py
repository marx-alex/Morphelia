# import external libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# import internal libraries
from morphelia.preprocessing import aggregate
import os


def time_plot(adata, var,
              hue=None,
              units=None,
              time_var='Metadata_Time',
              time_unit='h',
              show=True,
              save=False,
              aggregate_data=True,
              **kwargs):
    """Plot temporal course of a variable in wells/ cells.

    Args:
        adata (anndata.AnnData): Multidimensional morpholigical data.
        var (str): Variable from adata.
        hue (str): Color time courses by other variable from adata.
        units (str): Plot units separately.
        time_var (str): Variable from adata for time.
        time_unit (str): Time unit.
        show (bool)
        save (str): Path to save.
        aggregate_data (bool): True to aggregate data over time.
        kwargs (dict): Keyword arguments passed to seaborn.lineplot
    """
    if aggregate_data:
        by = [time_var]
        if units is not None:
            by.append(units)
        if hue is not None:
            by.append(hue)
        adata = aggregate(adata, by=by, qc=False)

    # generate data frame from variables given
    if time_var in adata.obs.columns:
        time_vals = adata.obs[time_var]
    elif time_var in adata.var_names:
        time_vals = adata[:, time_var].X
    else:
        raise KeyError(f"Variable for time not in AnnData object: f{time_var}")

    data = pd.DataFrame({'time': time_vals})

    # variable
    if var in adata.obs.columns:
        var_vals = adata.obs[var]
    elif var in adata.var_names:
        var_vals = adata[:, var].X
    else:
        raise KeyError(f"Variable for plotting in AnnData object: {var}")

    data[var] = var_vals

    # color
    if hue is not None:
        if hue in adata.obs.columns:
            hue_vals = adata.obs[hue]
        elif hue in adata.var_names:
            hue_vals = adata[:, hue].X
        else:
            raise KeyError(f"Hue variable not in AnnData object: f{hue}")

        data['hue'] = hue_vals
        kwargs.setdefault('hue', 'hue')

    # units
    if units is not None:
        if units in adata.obs.columns:
            units_vals = adata.obs[units]
        elif units in adata.var_names:
            units_vals = adata[:, units].X
        else:
            raise KeyError(f"Units variable not in AnnData object: f{units}")

        data['units'] = units_vals
        kwargs.setdefault('units', 'units')
        kwargs.setdefault('estimator', None)

    # plot
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.lineplot(data=data, x='time', y=var, palette='Dark2', **kwargs)

    ax.legend()

    ax.set_xlabel(f"time ({time_unit})")
    ax.set_title(var)

    if show:
        plt.show()

    # save
    if save:
        try:
            plt.savefig(os.path.join(save, "time_plot.png"))
        except OSError:
            print(f'Can not save figure to {save}.')


def time_heatmap(adata,
                 time_var='Metadata_Time',
                 treat_var='Metadata_Treatment',
                 feats=None,
                 conc_var=None,
                 aggregate_data=True,
                 share_cbar=True,
                 show=True,
                 save=False,
                 **kwargs):
    """Heatmap representation of median fold change in given features over time.

    First aggregate features for treatments and concetrations if given.

    Args:
        adata (anndata.AnnData): Multidimensional morpholigical data.
        time_var (str): Time variable.
        treat_var (str): Treatment variable.
        feats (list): List of features to show.
        conc_var (str): Concentration variable.
        aggregate_data (bool): True if data has to be aggreagated over time.
        share_cbar (bool): True to share color bar among all subplots.
        show (bool)
        save (str): Path to save figure.
        kwargs (dict): Keyword arguments passed to seaborn.heatmap.
    """
    # check variables
    if time_var not in adata.obs.columns:
        raise ValueError(f"Varibale for time not in anndata object: {time_var}")
    if treat_var not in adata.obs.columns:
        raise ValueError(f"Varibale for treatment not in anndata object: {treat_var}")
    if conc_var is not None:
        if conc_var not in adata.obs.columns:
            raise ValueError(f"Varibale for concentration not in anndata object: {conc_var}")
    if feats is not None:
        if isinstance(feats, list):
            if not all(v in adata.var_names for v in feats):
                raise KeyError(f"Variables for features are not in annotations: {feats}")
        elif isinstance(feats, str):
            feat_lst = []
            feat_lst.append(feats)
            feats = feat_lst
        else:
            raise TypeError(f"feats is expected to be either list or string, instead got {type(feats)}")

    # get pandas dataframe from features
    adata = adata.copy()

    # just show the first feature if feats is None
    if feats is None:
        feats = adata.var_names[:10]
    if feats is not None:
        adata = adata[:, feats]

    # aggregate treatments over time
    if not aggregate_data:
        adata_df = pd.concat([adata.obs[[time_var, treat_var]], adata.to_df()], axis=1)
        df_ix = treat_var
    elif conc_var is not None:
        adata = aggregate(adata, by=[time_var, treat_var, conc_var], qc=False)
        adata_df = pd.concat([adata.obs[[time_var, treat_var, conc_var]], adata.to_df()], axis=1)
        df_ix = [treat_var, conc_var]
    else:
        adata = aggregate(adata, by=[time_var, treat_var], qc=False)
        adata_df = pd.concat([adata.obs[[time_var, treat_var]], adata.to_df()], axis=1)
        df_ix = treat_var

    # make figure
    if len(feats) <= 5:
        w = len(feats)
        h = 1
    else:
        w = 5
        h = len(feats) // 5

    h_size = (h / (h + w)) * 15
    w_size = (w / (h + w)) * 20

    fig, axs = plt.subplots(h, w, figsize=(w_size, h_size), squeeze=False,
                            sharex=True, sharey=True)
    cbar_ax = None
    if share_cbar:
        cbar_ax = fig.add_axes([.91, .3, .03, .4])

    i, j = 0, 0

    for ix, feat in enumerate(feats):
        feat_time = adata_df.pivot(df_ix, time_var, feat)

        sns.heatmap(feat_time, ax=axs[i][j],
                    cbar=ix == 0 if share_cbar else True,
                    cbar_ax=None if ix else cbar_ax,
                    **kwargs)
        axs[i][j].set_title(feat)

        if i < (h-1):
            axs[i][j].set_xlabel('')
        if j > 0:
            axs[i][j].set_ylabel('')

        if j < (w-1):
            j += 1
        else:
            j = 0
            i += 1
            
    plt.tight_layout(rect=[0, 0, .9, 1])

    if show:
        plt.show()

    # save
    if save:
        try:
            plt.savefig(os.path.join(save, "time_heatmap.png"))
        except OSError:
            print(f'Can not save figure to {save}.')
