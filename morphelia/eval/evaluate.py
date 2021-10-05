import numpy as np
import pandas as pd
from morphelia.eval import similarity_matrix
from scipy.stats import norm


def repro_effect(adata,
                 group_var="Metadata_Treatment",
                 other_group_vars=None,
                 method='pearson',
                 control_id='ctrl',
                 use_rep=None,
                 n_pcs=50,
                 return_scores=False):
    """Compute reproducibility and effect metric for every condition described with group_var
    and other_group_vars.
    Reproducibility is a measure for the within-similarity of certain conditions.
    Effect is a measure how well certain conditions differ from control conditions.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        group_var (str): Find similarity between groups. Could be treatment conditions for example.
        other_group_vars (list): Other variables that define groups that a similar.
        method (str): Method for similarity/ distance computation.
            Should be one of: pearson, spearman, kendall, euclidean, mahalanobis.
        control_id (str): Name of control wells in group_var.
        use_rep (str): Calculate similarity/distance representation of X in .obsm.
        n_pcs (int): Number principal components to use if use_pcs is 'X_pca'.
        return_scores (bool): If True, no anndata.AnnData object is returned, but the scores directly.

    Returns:
        anndata.AnnData
        .uns['eval']['repro']['percentiles']: pandas.DataFrame with reproducibility scores and conditions.
        .uns['eval']['repro']['method']: Method used to calculate similarity.
        .uns['eval']['effect']['percentiles']: pandas.DataFrame with effect scores and conditions.
        .uns['eval']['effect']['method']: Method used to calculate similarity.
        pandas.Series, pandas.Series: Reproducibility scores and effect scores.
            Only if return_scores is True.
    """
    # check method
    avail_methods = ['pearson', 'spearman', 'kendall', 'euclidean', 'mahalanobis']
    method = method.lower()
    assert method in avail_methods, f"method should be in {avail_methods}, " \
                                    f"instead got {method}"

    # calculate correlation matrix
    corr_df = similarity_matrix(adata,
                                method=method,
                                group_var=group_var,
                                other_group_vars=other_group_vars,
                                use_rep=use_rep,
                                n_pcs=n_pcs,
                                show=False)

    # calculate reproducibility and effect
    repro_df = reproducibility(adata,
                               group_var=group_var,
                               other_group_vars=other_group_vars,
                               method=method,
                               sim_matrix=corr_df)

    effect_df = effect(adata,
                       group_var=group_var,
                       other_group_vars=other_group_vars,
                       method=method,
                       control_id=control_id,
                       sim_matrix=corr_df)

    if return_scores:
        return repro_df, effect_df

    # update unstructured data
    if 'eval' not in adata.uns:
        adata.uns['eval'] = {}
    adata.uns['eval']['repro'] = {'percentiles': repro_df, 'method': method}
    adata.uns['eval']['effect'] = {'percentiles': effect_df, 'method': method}

    return adata


def reproducibility(adata,
                    group_var='Metadata_Treatment',
                    other_group_vars=None,
                    method='pearson',
                    use_rep=None,
                    n_pcs=50,
                    sim_matrix=None,
                    return_scores=False):
    """Computes reproducibility metric for wells with certain treatments and doses for
    different plates and batches.
    First, statistics for a null distribution with wells that should not have a high similarity
    are calculated. Then biological replicates that are expected to be very similar are compared
    to that distribution. The percentile of the mean similarity from wells with same treatments/ concentrations
    within the null distribution is returned.

        Args:
            adata (anndata.AnnData): Multidimensional morphological data.
            group_var (str): Find similarity between groups. Could be treatment conditions for example.
            other_group_vars (list): Other variables that define groups that a similar.
            method (str): Method for similarity/ distance computation.
                Should be one of: pearson, spearman, kendall, euclidean, mahalanobis.
            use_rep (str): Calculate similarity/distance representation of X in .obsm.
            n_pcs (int): Number principal components to use if use_pcs is 'X_pca'.
            sim_matrix (pandas.DataFrame): Use this matrix to compute reproducibility.
            return_scores (bool): If True, no anndata.AnnData object is returned, but the scores directly.

        Returns:
        anndata.AnnData
        .uns['eval']['repro']['percentiles']: pandas.DataFrame with reproducibility scores and conditions.
        .uns['eval']['repro']['method']: Method used to calculate similarity.
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

    # every treatment/dose pair should have at least one biological replicate
    if other_group_vars is None:
        assert len(adata.obs[group_var]) != len(adata.obs[group_var].unique), "Found no biological " \
                                                                              f"replicates for {group_var}"
    else:
        other_group_vars_lst = adata.obs[other_group_vars].values.tolist()
        all_group_vars = [(gv,) + tuple(ogv) for gv, ogv in zip(adata.obs[group_var], other_group_vars_lst)]
        assert len(all_group_vars) != len(set(all_group_vars)), "Found no biological " \
                                                                f"replicates for {group_var} and {other_group_vars}"

    # check method
    avail_methods = ['pearson', 'spearman', 'kendall', 'euclidean', 'mahalanobis']
    method = method.lower()
    assert method in avail_methods, f"method should be in {avail_methods}, " \
                                    f"instead got {method}"

    # calculate correlation matrix
    if sim_matrix is None:
        corr_df = similarity_matrix(adata,
                                    method=method,
                                    group_var=group_var,
                                    other_group_vars=other_group_vars,
                                    use_rep=use_rep,
                                    n_pcs=n_pcs,
                                    show=False)
    else:
        assert isinstance(sim_matrix, pd.DataFrame), f"sim_matrix expected to be type(pandas.DataFrame), " \
                                                     f"instead got {type(sim_matrix)}"
        corr_df = sim_matrix

    # get upper triangular matrix
    corr_tri_df = corr_df.where(~np.tril(np.ones(corr_df.shape)).astype(np.bool))

    groups = group_var
    if other_group_vars:
        groups = [group_var] + other_group_vars

    # get null distribution with value pairs that should have low similarity
    def _erase_vals(df, columns):
        df[columns] = np.nan
        return df

    null_dist_df = corr_tri_df.groupby(groups).apply(lambda x: _erase_vals(x, x.name))

    # get statistics from null distribution
    null_std = np.nanstd(null_dist_df, ddof=1)
    null_mean = np.nanmean(null_dist_df)

    # get mean similarity for every group
    repro_df = corr_tri_df.groupby(groups).apply(lambda x: np.nanmean(x[x.name]))
    repro_df.rename('reproducibility', inplace=True)

    # calculate z_scores
    repro_df = (repro_df - null_mean) / null_std
    # get percentile
    repro_df = repro_df.apply(lambda x: 1 - norm.sf(abs(x)))

    if return_scores:
        return repro_df

    # update unstructured data
    if 'eval' not in adata.uns:
        adata.uns['eval'] = {}
    adata.uns['eval']['repro'] = {'percentiles': repro_df, 'method': method}

    return adata


def effect(adata,
           group_var='Metadata_Treatment',
           other_group_vars=None,
           control_id='ctrl',
           method='pearson',
           use_rep=None,
           n_pcs=50,
           sim_matrix=None,
           return_scores=False):
    """Computes effect metric for wells with certain treatments and doses for
    different plates and batches.
    First, statistics for a null distribution with similarities/distances between control wells are calculated.
    Then the mean distance/similarity of Treatment/Concentration wells to control wells
    is compared to the null distribution.
    The percentile of the mean Treatment/Concentration similarity/distance on the null distribution is returned.
    The metric highly depends on reproducibility of control wells.

        Args:
            adata (anndata.AnnData): Multidimensional morphological data.
            group_var (str): Find similarity between groups. Could be treatment conditions for example.
            other_group_vars (list): Other variables that define groups that a similar.
            method (str): Method for similarity/ distance computation.
                Should be one of: pearson, spearman, kendall, euclidean, mahalanobis.
            use_rep (str): Calculate similarity/distance representation of X in .obsm.
            n_pcs (int): Number principal components to use if use_pcs is 'X_pca'.
            control_id (str): Name of control wells in group_var.
            sim_matrix (pandas.DataFrame): Use this matrix to compute reproducibility.
            return_scores (bool): If True, no anndata.AnnData object is returned, but the scores directly.

        Returns:
        anndata.AnnData
        .uns['eval']['effect']['percentiles']: pandas.DataFrame with effect scores and conditions.
        .uns['eval']['effect']['method']: Method used to calculate similarity.
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

    assert control_id in adata.obs[group_var].tolist(), f"control_id not in {group_var}: {control_id}"

    # check method
    avail_methods = ['pearson', 'spearman', 'kendall', 'euclidean', 'mahalanobis']
    method = method.lower()
    assert method in avail_methods, f"method should be in {avail_methods}, " \
                                    f"instead got {method}"

    # calculate correlation matrix
    if sim_matrix is None:
        corr_df = similarity_matrix(adata,
                                    method=method,
                                    group_var=group_var,
                                    other_group_vars=other_group_vars,
                                    use_rep=use_rep,
                                    n_pcs=n_pcs,
                                    show=False)
    else:
        assert isinstance(sim_matrix, pd.DataFrame), f"sim_matrix expected to be type(pandas.DataFrame), " \
                                                     f"instead got {type(sim_matrix)}"
        corr_df = sim_matrix

    # get upper triangular matrix
    corr_tri_df = corr_df.where(~np.tril(np.ones(corr_df.shape)).astype(np.bool))

    # get null distribution
    null_dist_df = corr_tri_df.loc[control_id, control_id]

    if other_group_vars is not None:
        null_stds = []
        null_means = []
        for groups, group_df in null_dist_df.groupby(other_group_vars):
            group_df = group_df[groups]
            null_stds.append(np.nanstd(group_df, ddof=1))
            null_means.append(np.nanmean(group_df))

        # take mean of grouped statistics
        null_std = np.mean(null_stds)
        null_mean = np.mean(null_means)
    else:
        # get statistics from null distribution
        null_std = np.nanstd(null_dist_df, ddof=1)
        null_mean = np.nanmean(null_dist_df)

    # get mean similarity for every group
    groups = group_var
    if other_group_vars:
        groups = [group_var] + other_group_vars

    effect_df = corr_tri_df.groupby(groups).apply(lambda x: np.nanmean(x[control_id]))
    effect_df.rename('effect', inplace=True)

    # calculate z_scores
    effect_df = (effect_df - null_mean) / null_std
    # get percentile
    effect_df = effect_df.apply(lambda x: 1 - norm.sf(abs(x)))

    if return_scores:
        return effect_df

    # update unstructured data
    if 'eval' not in adata.uns:
        adata.uns['eval'] = {}
    adata.uns['eval']['effect'] = {'percentiles': effect_df, 'method': method}

    return adata
