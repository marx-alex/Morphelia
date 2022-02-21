import logging
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2

from morphelia.tools.utils import get_subsample

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def feature_lmem(
    adata,
    treat_var="Metadata_Treatment",
    ctrl_id="ctrl",
    fixed_var="Metadata_Concentration",
    rand_var="BatchNumber",
    method="bonferroni",
    alpha=0.05,
    r2_thresh=0.7,
    drop=False,
    subsample=False,
    sample_size=10000,
    seed=0,
):
    r"""
    Use Linear Mixed Models to select features that have an effect compared to
    the control and are dependent on dose.

    For every treatment and feature a Linear Mixed Model is fitted:

    .. math::
        Y_{ij} = \beta_{0} + \beta_{1}X_{ij} + \gamma_{1i}X_{ij} + \epsilon_{ij}

    The model describes the relation between a fixed (independent) variable and
    a dependent variable while considering random effects.
    We use the control condition and dose steps as independent variables assuming
    a linear equidistant dependency. The model is fitted for every feature
    as dependent variable.

    R-squared and p-values are collected for every variable and treatment.
    The latter is corrected for multiple testing
    with different treatments by a given method (i.g. bonferroni).

    .. math::
        R^{2} = 1 - \frac{\sum(y-\hat{y}^2}{y-\bar{y}}

    R-squared values are then combined by mean-aggregation.

    Features can be dropped directly based on the given thresholds.
    Features are assumed to be significant, if at least on p-value is lower than the given alpha
    or the combined R-squared value is above the given threshold for R-squared.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        treat_var (str): Treatment variable in .obs.
        ctrl_id (str): Name of control condition stored in treat_var.
        fixed_var (str): Name of variable with fixed effect.
        rand_var (list, str): Name or list of variables with random effects.
        method (str): Method to correct for multiple testing.
            Can be 'bonferroni', 'bh' for Benjamin/Hochberg or any other method
            used by statsmodels.stats.multitest.multipletests
        alpha (float): Alpha value for significance.
        r2_thresh (float): Threshold for R-squared.
        drop (bool): Drop features based on thresholds.
            Stores dropped features in .uns['lmm_dropped']
        subsample (bool): Use method on subsample of the data.
        sample_size (int): Size of subsample.
        seed (int): Seed for reproducibility.

    Returns:
        anndata.AnnData

    """
    # check variables
    assert treat_var in adata.obs.columns, f"treat_var not in .obs: {treat_var}"
    assert fixed_var in adata.obs.columns, f"fixed_var not in .obs: {fixed_var}"
    if isinstance(rand_var, str):
        rand_var = [rand_var]
    assert all(
        rv in adata.obs.columns for rv in rand_var
    ), f"rand_var not in .obs: {rand_var}"
    # control group should be in treat_var
    unique_treats = list(adata.obs[treat_var].unique())
    if ctrl_id not in unique_treats:
        warnings.warn(
            f"Control group not in {treat_var}: {unique_treats}. Control will not be included in dose steps."
        )
    else:
        unique_treats.remove(ctrl_id)

    # evtl subsample data
    if subsample:
        adata_ss = get_subsample(adata, sample_size=sample_size, seed=seed)
    else:
        adata_ss = adata.copy()

    # cache evaluation metrics
    r2_metric = pd.DataFrame(0, index=unique_treats, columns=adata_ss.var_names)
    p_metric = pd.DataFrame(0, index=unique_treats, columns=adata_ss.var_names)

    # iterate over treatments and get cross-validated
    # R-squared scores for every treatment and feature
    for treat in unique_treats:
        logger.info(f"Fit LME on {treat} [{treat_var}]")
        adata_treat = adata_ss[
            (adata_ss.obs[treat_var] == treat) | (adata_ss.obs[treat_var] == ctrl_id)
        ].copy()
        r2, p = multivariate_lmem_fit(
            adata_treat, fixed_effect=fixed_var, rand_effect=rand_var
        )
        # store output
        r2_metric.loc[treat, :] = r2
        p_metric.loc[treat, :] = p

    # correct for multiple testing
    method = method.lower()
    if method == "bh":
        method = "fdr_bh"

    def corr_func(x):
        return multipletests(x, alpha=alpha, method=method)[1]

    p_metric = p_metric.apply(corr_func, axis=0)

    # add metrics to .var
    r2_mean = r2_metric.mean()
    r2_std = r2_metric.std()
    adata.var["r2_mean"] = r2_mean
    adata.var["r2_std"] = r2_std
    for row_ix, row in r2_metric.iterrows():
        col_name = f"r2_{row_ix}"
        adata.var[col_name] = row
    for row_ix, row in p_metric.iterrows():
        col_name = f"p_{row_ix}"
        adata.var[col_name] = row

    # get masks based on decisions
    r2_mask = r2_mean > r2_thresh
    p_mask = (p_metric < alpha).any(axis=0)
    combined_mask = np.logical_or(r2_mask, p_mask)
    adata.var["r2_mask"] = r2_mask
    adata.var["p_mask"] = p_mask
    adata.var["lme_combined_mask"] = combined_mask

    if drop:
        dropped_feats = adata.var_names[~combined_mask]
        logger.info(f"Dropped {len(dropped_feats)} features: {dropped_feats}")
        adata.uns["lmm_dropped"] = dropped_feats
        adata = adata[:, combined_mask].copy()

    return adata


def multivariate_lmem_fit(
    adata, fixed_effect="Metadata_Concentration", rand_effect="BatchNumber"
):
    # store p-values from log-likelihood ratio test
    ps = []
    # store r_squared values
    r2 = []

    # get effects
    unique_x = adata.obs[fixed_effect].unique()
    unique_x = np.sort(unique_x)
    x_mapping = {v: i for i, v in enumerate(unique_x)}
    x = adata.obs[fixed_effect].map(x_mapping)
    logger.info(f"Independent variables: {x_mapping}")
    x = x.to_numpy()[:, None]  # --> must be 2d
    z = adata.obs[rand_effect]
    if isinstance(z, pd.DataFrame):
        # merge columns
        z = z.apply(lambda col: "_".join(col.astype(str)), axis=1)
    z = z.astype("category")
    z = np.asarray(z.cat.codes)

    for feat_ix, feat in tqdm(
        enumerate(adata.var_names),
        desc="Fitting LME-Model on every feature...",
        total=len(adata.var_names),
    ):
        y = adata[:, feat].X.copy()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # model
            lmm = sm.MixedLM(endog=y, exog=x, groups=z)

            try:
                f_lmm = lmm.fit(reml=True)

                # r-squared with random effects
                re = f_lmm.random_effects
                rev = [re[g]["Group Var"] for g in z]
                y_pred = f_lmm.predict(exog=x) + rev
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                ps.append(f_lmm.pvalues[0])  # --> pvalue of the coefficient
                r2.append(r_squared)

            except (np.linalg.LinAlgError, ValueError):
                logger.info(f"Raised exception at features: {feat}")
                ps.append(1)
                r2.append(0)

    ps = np.asarray(ps)
    r2 = np.asarray(r2)
    r2 = np.clip(r2, a_min=0, a_max=None)
    return r2, ps


def log_likelihood_ratio_test(ll0, ll1):
    lrt = np.abs(ll1 - ll0) * 2
    p = chi2.sf(lrt, 1)
    return p
