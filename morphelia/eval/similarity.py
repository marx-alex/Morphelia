import pandas as pd
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt
import os


def similarity_matrix(adata,
                      method='pearson',
                      treat_var='Metadata_Treatment',
                      dose_var=None,
                      show=False,
                      save=None):
    """Computes a similarity or distance matrix between different treatments and doses if given.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        method (str): Method for similarity/ distance computation.
            Should be one of: pearson, spearman, kendall, euclidean, mahalanobis.
        treat_var (str): Variable for treatment conditions.
        dose_var (str): Variable for doses.
        show (bool): Show plot.
        save (str): Save plot to a specified location.

    Returns:
        numpy.array: similarity/ distance matrix
    """
    # check variables
    assert treat_var in adata.obs.columns, f"treat_var not in observations: {treat_var}"
    if dose_var is not None:
        assert dose_var in adata.obs.columns, f"dose_var not in observations: {dose_var}"

    # check method
    avail_methods = ['pearson', 'spearman', 'kendall', 'euclidean', 'mahalanobis']
    method = method.lower()
    assert method in avail_methods, f"method should be in {avail_methods}, " \
                                    f"instead got {method}"

    # load profiles to dataframe and transpose
    if dose_var is not None:
        profiles_df = pd.DataFrame(adata.X, columns=adata.var_names, index=[adata.obs[treat_var],
                                                                            adata.obs[dose_var]]).T
    else:
        profiles_df = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs[treat_var]).T

    # calculate similarity
    if method == 'pearson':
        profiles_df = profiles_df.corr(method='pearson')

    if method == 'spearman':
        profiles_df = profiles_df.corr(method='spearman')

    if method == 'kendall':
        profiles_df = profiles_df.corr(method='kendall')

    if method == 'euclidean':
        sim = squareform(pdist(profiles_df.T, metric='euclidean'))
        profiles_df = pd.DataFrame(sim, columns=profiles_df.columns, index=profiles_df.columns)

    if method == 'mahalanobis':
        sim = squareform(pdist(profiles_df.T, metric='manhattan'))
        profiles_df = pd.DataFrame(sim, columns=profiles_df.columns, index=profiles_df.columns)

    if show:
        sns.set_theme()
        plt.figure()
        cm = sns.heatmap(profiles_df)
        plt.suptitle(f'distance/ similarity: {method}', fontsize=16)

        if save:
            try:
                plt.savefig(os.path.join(save, "feature_correlation.png"))
            except OSError:
                print(f'Can not save figure to {save}.')

    return profiles_df
