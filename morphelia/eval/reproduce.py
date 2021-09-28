import numpy as np
from morphelia.eval import similarity_matrix
from scipy.stats import norm


def repro_score(adata,
                treat_var='Metadata_Treatment',
                dose_var=None):
    """Computes reproducibility metric for well with certain treatments and doses for
    different plates and batches.
    First, the mean and standard deviations for all random correlation pairs are calculated.
    Then the median Pearson correlation of a certain treatment group is compared to the overall distribution.

        Args:
            adata (anndata.AnnData): Multidimensional morphological data.
            treat_var (str): Variable for treatment conditions.
            dose_var (str): Variable for doses.
            k (int): Number of samples to draw from random pairwise pearson correlations.

        Returns:
            pd.DataFrame: Treatments and their reproducibility score.
    """
    # check variables
    assert treat_var in adata.obs.columns, f"treat_var not in observations: {treat_var}"
    if dose_var is not None:
        assert dose_var in adata.obs.columns, f"dose_var not in observations: {dose_var}"

    # calculate correlation matrix
    corr_df = similarity_matrix(adata,
                                method='pearson',
                                treat_var=treat_var,
                                dose_var=dose_var,
                                show=False)

    # get upper triangular matrix
    corr_tri_df = corr_df.where(~np.tril(np.ones(corr_df.shape)).astype(np.bool))

    # get statistics from all correlations
    all_std = np.nanstd(corr_tri_df.values, ddof=1)
    all_mean = np.nanmean(corr_tri_df)

    # get mean correlation for every group
    groups = treat_var
    if dose_var:
        groups = [treat_var, dose_var]
    repro_score_df = corr_tri_df.groupby(groups).apply(lambda x: np.nanmedian(x[x.name]))
    repro_score_df.rename('repro_p_value', inplace=True)

    # calculate z_scores
    repro_score_df = (repro_score_df - all_mean) / all_std
    # get p-value
    repro_score_df = repro_score_df.apply(lambda x: norm.sf(abs(x)))

    return repro_score_df
