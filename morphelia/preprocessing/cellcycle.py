from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def assign_cc(adata, new_feat="Metadata_CellCycle",
              feat='Primarieswithoutborder_Intensity_IntegratedIntensity_DNA',
              by=("BatchNumber", "PlateNumber"),
              treat_var="Metadata_Treatment",
              n=100, show=None, save=None, dead_thresh=None, seed=0, **kwargs):
    """Assigns each cell to a cell cycle phase based on the integrated intensity of nuclei.
    This function applies a gaussian mixture model to find two gaussian distributions -
    one for G1 and one for G2/S cells. The intersection of the two distributions is taken
    as threshold that distinguishes between cell cycle phases.
    If threshold for dead cells is given all cells of the G1 distribution
    below the confidence interval are assigned dead.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        new_feat (str): Name for new annotation.
        feat (str): Feature to apply threshold on.
        by (iterable): Get threshold for every batch, plat and treatment.
        treat_var (str): Treatment annotation.
        n (int): Sample size for gaussian model.
        show (dict): Define elements of by to show.
        save (str): Path where to save figure.
        dead_thresh (str): z-score to find a certain quartile of the G1 distribution.
            All cells below are assigned dead cells.
        seed (int): Passed to GMM for reproducibility.
        kwargs: Passed to sklearns GMM
    """
    if isinstance(by, str):
        by = [by]
    elif isinstance(by, tuple):
        by = list(by)

    by.append(treat_var)
    # check that variables in by are in anndata
    if not all(var in adata.obs.columns for var in by):
        raise KeyError(f"Variables defined in 'by' are not in annotations: {by}")

    # initiate new cell cycle annotation
    adata.obs[new_feat] = 0

    # number of distributions
    k = 2

    # iterate over md with grouping variables
    for groups, sub_df in adata.obs.groupby(by):
        # cache indices of group
        group_ix = sub_df.index

        # initiate and fit the KDE model
        X = adata[group_ix, feat].X.copy()
        X_min = np.min(X)
        X_max = np.max(X)
        X_plot = np.linspace(X_min, X_max, n)

        model = GaussianMixture(k, random_state=seed, **kwargs).fit(X.reshape(-1, 1))

        # get resulting fit
        dens = np.exp(model.score_samples(X_plot.reshape(-1, 1)))
        prob = model.predict_proba(X_plot.reshape(-1, 1))
        children = prob * dens[:, np.newaxis]

        # find statistical measures of G1 distribution (left)
        child_mean = model.means_.flatten()
        cov = model.covariances_
        child_std = [np.sqrt(np.trace(cov[i]) / k) for i in range(0, k)]
        child_g1, child_g2 = np.argsort(child_mean)
        g1_mean = child_mean[child_g1]
        g1_std = child_std[child_g1]

        # find intersection
        inter_ix = np.argwhere(np.diff(np.sign(children[:, child_g1] - children[:, child_g2]))).flatten()
        thresh = X_plot[inter_ix]

        # apply threshold to get new cell cycle annotations
        g1_ix = group_ix[(X < thresh).flatten()]
        adata.obs.loc[g1_ix, new_feat] = "G1"

        if dead_thresh is not None:
            # get threshold for dead cells
            thresh_dead = g1_mean - (dead_thresh * g1_std)

            # apply threshold for dead cells
            dead_ix = group_ix[(X < thresh_dead).flatten()]
            adata.obs.loc[dead_ix, new_feat] = "DEAD"

        group_dict = dict(zip(by, groups))

        # plot
        if group_dict == show:
            sns.set_theme()
            p = sns.histplot(X, kde=True, stat="density", linewidth=0)

            if dead_thresh is not None:
                for rectangle in p.patches:
                    if rectangle.get_x() < thresh_dead:
                        rectangle.set_facecolor('firebrick')

            # plt.plot(X_plot, dens, '--k', label='Mixture')
            plt.plot(X_plot, children[:, child_g1], linestyle='-', color='darkorange',
                     label='G1', alpha=0.8)
            plt.plot(X_plot, children[:, child_g2], linestyle='-', color='forestgreen',
                     label='G2/S', alpha=0.8)
            plt.axvline(thresh, color='k', linestyle='dotted', label='Threshold')
            if dead_thresh is not None:
                plt.axvline(thresh_dead, color='r', linestyle='dotted', label='Dead Threshold')
            plt.legend()
            plt.title(f"{', '.join([f'{key}: {item}' for key, item in group_dict.items()])}")
            plt.xlabel(f"{feat}")

            # save
            if save is not None:
                try:
                    plt.savefig(os.path.join(save, "cellcycle.png"))
                except OSError:
                    print(f'Can not save figure to {save}.')

    # assign remaining zeros to G2/S phase
    adata.obs.loc[adata.obs[new_feat] == 0, new_feat] = 'G2S'

    return adata
