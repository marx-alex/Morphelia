import pandas as pd
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os

from morphelia.tools._utils import _choose_representation


def similarity_matrix(adata,
                      method='pearson',
                      group_var='Metadata_Treatment',
                      other_group_vars=None,
                      use_rep=None,
                      n_pcs=50,
                      show=False,
                      save=None):
    """Computes a similarity or distance matrix between different treatments and doses if given.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        method (str): Method for similarity/ distance computation.
            Should be one of: pearson, spearman, kendall, euclidean, mahalanobis.
        group_var (str): Find similarity between groups. Could be treatment conditions for example.
        other_group_vars (list): Other variables that define groups that a similar.
        use_rep (str): Calculate similarity/distance representation of X in .obsm.
        n_pcs (int): Number principal components to use if use_pcs is 'X_pca'.
        show (bool): Show plot.
        save (str): Save plot to a specified location.

    Returns:
        numpy.array: similarity/ distance matrix
    """
    # check variables
    assert group_var in adata.obs.columns, f"treat_var not in observations: {group_var}"
    if other_group_vars is not None:
        if isinstance(other_group_vars, str):
            other_group_vars = [other_group_vars]
        assert isinstance(other_group_vars, list), "Expected type for other_group_vars is string or list, " \
                                                   f"instead got {type(other_group_vars)}"
        assert all(var in adata.obs.columns for var in other_group_vars), f"other_group_vars not in " \
                                                                          f"observations: {other_group_vars}"

    # check method
    avail_methods = ['pearson', 'spearman', 'kendall', 'euclidean', 'mahalanobis']
    method = method.lower()
    assert method in avail_methods, f"method should be in {avail_methods}, " \
                                    f"instead got {method}"

    # get representation of data
    if use_rep is None:
        use_rep = 'X'
    X = _choose_representation(adata,
                               rep=use_rep,
                               n_pcs=n_pcs)

    # load profiles to dataframe and transpose
    if other_group_vars is not None:
        group_vars = [group_var] + other_group_vars
        profiles_df = pd.DataFrame(X, index=[adata.obs[var] for var in group_vars]).T
    else:
        profiles_df = pd.DataFrame(X, index=adata.obs[group_var]).T

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
        
    # sort index
    profiles_df.sort_index(axis=0, inplace=True)
    profiles_df.sort_index(axis=1, inplace=True)

    if show:
        sns.set_theme()
        cmap = matplotlib.cm.plasma

        fig = plt.figure(figsize=(7, 5))
        ax = sns.heatmap(profiles_df, cmap=cmap)
        plt.suptitle(f'distance/ similarity: {method}', fontsize=16)

        if save:
            try:
                plt.savefig(os.path.join(save, "feature_correlation.png"))
            except OSError:
                print(f'Can not save figure to {save}.')

    return profiles_df
