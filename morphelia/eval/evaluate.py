import numpy as np
from morphelia.eval import similarity_matrix
from scipy.stats import norm


def reproducibility(adata,
                    group_var='Metadata_Treatment',
                    other_group_vars=None,
                    method='pearson'):
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

        Returns:
            pd.DataFrame: Treatments/Concentrations and their reproducibility score.
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
    corr_df = similarity_matrix(adata,
                                method=method,
                                group_var=group_var,
                                other_group_vars=other_group_vars,
                                show=False)

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
    repro_score_df = corr_tri_df.groupby(groups).apply(lambda x: np.nanmean(x[x.name]))
    repro_score_df.rename('reproducibility', inplace=True)

    # calculate z_scores
    repro_score_df = (repro_score_df - null_mean) / null_std
    # get percentile
    repro_score_df = repro_score_df.apply(lambda x: 1 - norm.sf(abs(x)))

    return repro_score_df


def effect(adata,
           group_var='Metadata_Treatment',
           other_group_vars=None,
           control_id='ctrl',
           method='pearson'):
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
            control_id (str): Name of control wells in group_var.
            Should be one of: pearson, spearman, kendall, euclidean, mahalanobis.

        Returns:
            pd.DataFrame: Treatments/Concentrations and their reproducibility score.
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
    corr_df = similarity_matrix(adata,
                                method=method,
                                group_var=group_var,
                                other_group_vars=other_group_vars,
                                show=False)

    # get upper triangular matrix
    corr_tri_df = corr_df.where(~np.tril(np.ones(corr_df.shape)).astype(np.bool))

    null_dist_df = corr_tri_df.loc[control_id, control_id]

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

    return effect_df
