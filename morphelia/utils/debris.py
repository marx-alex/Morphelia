import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.filters import threshold_minimum
import os


def filter_debris(md,
                  nucl_area='Primarieswithoutborder_AreaShape_Area',
                  cell_area='Cells_AreaShape_Area', max_quot=20,
                  n=5000, show=False, save=None, seed=0, thresh_bound=[1, 5],
                  verbose=False, **kwargs):
    """Calculates the quotient of cell area and nuclei area to detect vital and dead cells.
    In dead cells nuclei area of the segmentation almost equals cell area.
    A threshold based on skimage minimum method is calculated to distinguish between vital and dead cells.
    The histogram of the 'cell area/ nuclei area' quotients is smoothed until there are only two maxima.
    The minimum in between is the threshold value.

    Args:
        md (anndata.AnnData): Multidimensional morphological data.
        nucl_area (str): Variable with information about nuclear area
        cell_area (str): Varibale with information about cell area
        max_quot (int): Maximum value for quotient. (Not used for filtering)
        n (int): Sample size for gaussian model.
        show (bool): Define elements of by to show.
        save (str): Path where to save figure.
        seed (int): Passed to GMM for reproducibility.
        thresh_bound (list): Minimum and maximum threshold value.
        verbose (bool)
        kwargs: Passed to skimage minimum threshold.
    """
    # get samples
    np.random.seed(seed)
    unfiltered_len = md.shape[0]
    sample_ix = np.random.randint(unfiltered_len, size=n)
    try:
        na = md[:, nucl_area].X.copy()[sample_ix]
        ca = md[:, cell_area].X.copy()[sample_ix]
    except:
        na = md[:, nucl_area].X.copy()
        ca = md[:, cell_area].X.copy()

    # get sample distribution
    X = ca / na
    X = X[X < max_quot]

    # bimodal histogram
    thresh_min = threshold_minimum(X.reshape(-1, 1), **kwargs)

    # apply filter
    if len(thresh_bound) != 2:
        raise ValueError(f"Minimum and maximum for threshold value should be given: {thresh_bound}")
    if (thresh_min > thresh_bound[0]) and (thresh_min < thresh_bound[1]):
        md = md[(md[:, cell_area].X / md[:, nucl_area].X) > thresh_min, :]
    else:
        print("No threshold could be found that meets the criteria.")

    if verbose:
        filtered_len = md.shape[0]
        print(f"{unfiltered_len - filtered_len} cells filtered")

    # plot
    if show:
        sns.set_theme()
        p = sns.histplot(X, kde=True, stat="density", linewidth=0)
        bins = [patch.get_x() for patch in p.patches]
        nearest = (np.abs(bins - thresh_min)).argmin()
        for rect_ix, rectangle in enumerate(p.patches):
            if rect_ix < nearest:
                rectangle.set_facecolor('firebrick')
        p.patches[0].set_label('DEAD')
        plt.axvline(thresh_min, color='k', linestyle='dotted', label='Threshold')
        plt.legend()
        plt.xlabel(f"{cell_area}/{nucl_area}")

        # save
        if save is not None:
            try:
                plt.savefig(os.path.join(save, "debris_filter.png"))
            except OSError:
                print(f'Can not save figure to {save}.')

    return md
