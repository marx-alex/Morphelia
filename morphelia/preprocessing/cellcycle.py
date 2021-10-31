from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from morphelia.tools._utils import _get_subsample


def assign_cc(adata, new_feat="Metadata_CellCycle",
              feat='Primarieswithoutborder_Intensity_IntegratedIntensity_DNA',
              by=("BatchNumber", "PlateNumber", "Metadata_Treatment"),
              subsample=False,
              sample_size=1000,
              filter_outlier=True,
              show=None, save=None, dead_thresh=None, seed=0, **kwargs):
    """Assigns each cell to a cell cycle phase based on the integrated intensity of nuclei.
    This function applies otsu threshold to separate two intensity distributions -
    one for G1 and one for G2/S cells.
    If threshold for dead cells is given all cells of the G1 distribution
    below the confidence interval are assigned dead.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        new_feat (str): Name for new annotation.
        feat (str): Feature to apply threshold on.
        by (iterable): Get threshold for every batch, plat and treatment.
        subsample (bool): If True, fit models on subsample of data.
        sample_size (int): Size of supsample.
            Only if subsample is True.
        filter_outlier: Filter cells with intensity of three standard
            deviations or more above the median before thresholding.
        show (bool): Show plots.
        save (str): Path where to save figure.
        dead_thresh (int): z-score to find a certain quantile of the G1 distribution.
            All cells below are assigned dead cells.
        seed (int): Passed to GMM for reproducibility.
        kwargs: Passed to sklearns GMM
    """
    if isinstance(by, str):
        by = [by]
    elif isinstance(by, tuple):
        by = list(by)

    # check that variables in by are in anndata
    if not all(var in adata.obs.columns for var in by):
        raise KeyError(f"Variables defined in 'by' are not in annotations: {by}")

    # initiate new cell cycle annotation
    adata.obs[new_feat] = 'G2/S'

    # iterate over adata with grouping variables
    for groups, sub_df in adata.obs.groupby(by):

        # cache indices of group
        group_ix = sub_df.index

        # get subsample
        if subsample:
            adata_samp = _get_subsample(adata[group_ix, :].copy(),
                                        sample_size=sample_size,
                                        seed=seed)
        else:
            adata_samp = adata[group_ix, :].copy()

        X = adata_samp[:, feat].X.copy()

        # outlier detection
        X_median = np.median(X)
        X_std = np.std(X, ddof=1)

        if filter_outlier:
            upper_bound = int(X_median + (3 * X_std))
            X_bound = X[:upper_bound]
        else:
            X_bound = X

        # bimodal histogram
        otsu_thresh = threshold_otsu(X_bound.reshape(-1, 1), **kwargs)

        # apply threshold to get new cell cycle annotations
        g1_mask = (adata[group_ix, feat].X < otsu_thresh).flatten()
        g1_ix = group_ix[g1_mask]
        adata.obs.loc[g1_ix, new_feat] = "G1"

        if dead_thresh is not None:
            # get threshold for dead cells
            X_g1 = X[(X < otsu_thresh).flatten()]
            g1_mean = np.mean(X_g1)
            g1_std = np.std(X_g1, ddof=1)
            thresh_dead = g1_mean - (dead_thresh * g1_std)

            # apply threshold for dead cells
            dead_mask = (adata[group_ix, feat].X < thresh_dead).flatten()
            dead_ix = group_ix[dead_mask]
            adata.obs.loc[dead_ix, new_feat] = "DEAD"

        if not isinstance(groups, (list, tuple)):
            group_dict = {by[0]: groups}
        else:
            group_dict = dict(zip(by, groups))

        # plot
        if show:
            sns.set_theme()
            fig, ax = plt.subplots()
            p = sns.histplot(X_bound, kde=True, stat="density", linewidth=0, ax=ax)

            otsu_bins = [np.abs(patch.get_x() - otsu_thresh) for patch in p.patches]
            otsu_nearest = np.argmin(otsu_bins)

            if dead_thresh is not None:
                dead_bins = [np.abs(patch.get_x() - thresh_dead) for patch in p.patches]
                dead_nearest = np.argmin(dead_bins)

            g1_patch = False
            g2_patch = False
            dead_patch = False

            for rect_ix, rectangle in enumerate(p.patches):
                if rect_ix >= otsu_nearest:
                    rectangle.set_facecolor('#FFBF86')
                    if not g2_patch:
                        rectangle.set_label('G2/S')
                        g2_patch = True
                elif (rect_ix < dead_nearest) and (dead_thresh is not None):
                    rectangle.set_facecolor('#B91646')
                    if not dead_patch:
                        rectangle.set_label('Dead')
                        dead_patch = True
                else:
                    rectangle.set_facecolor('#CEE5D0')
                    if not g1_patch:
                        rectangle.set_label('G1')
                        g1_patch = True

            plt.axvline(otsu_thresh, color='k', linestyle='dotted', label='Otsu threshold')
            if dead_thresh is not None:
                plt.axvline(thresh_dead, color='r', linestyle='dotted', label='Dead threshold')
            plt.legend()
            plt.xlim([0, 800])
            plt.title(f"{', '.join([f'{key}: {item}' for key, item in group_dict.items()])}")
            plt.xlabel(f"{feat}")

            # save
            if save is not None:
                try:
                    title_str = f"{'_'.join([f'{key}_{item}' for key, item in group_dict.items()])}"
                    plt.savefig(os.path.join(save, f"{title_str}_cellcycle.png"))
                except OSError:
                    print(f'Can not save figure to {save}.')

    return adata
