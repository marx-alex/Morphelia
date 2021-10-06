# internal libraries
import warnings
from morphelia.preprocessing.basic import drop_nan

# external libraries
import statsmodels.formula.api as smf
from tqdm import tqdm
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

from morphelia.tools._utils import _get_subsample


def svm_rfe(adata,
            treat_var='Metadata_Treatment',
            kernel='linear',
            C=1,
            subsample=False,
            sample_size=1000,
            seed=0,
            drop=True,
            verbose=False,
            **kwargs):
    """
    Support-vector-machine-based recursive-feature elimination.

    Recursively Remove features with low weights from classification if SVM.
    Features are trained to classify treat_var (i.g. treatment condition in .obs).

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        treat_var (str): Name of annotation with treatments.
        C (float): Regularization parameter for SVC.
        subsample (bool): If True, fit models on subsample of data.
        sample_size (int): Size of supsample.
            Only if subsample is True.
        seed (int): Seed for subsample calculation.
            Only if subsample is True.
        kernel (str): Kernel type to be used for SVC.
        drop (bool): Drop least important features.
        verbose (bool)
        **kwargs (dict): Keyword arguments passed to sklearn.feature_selection.RFE.

    Returns:
        anndata.AnnData
        .uns['svm_rfe_feats']: Dropped features with low importance from classification task.
        .var['svm_rfe_feats']: True for features with low importance.
            Only if drop is False.
    """
    # get subsample
    if subsample:
        adata_ss = _get_subsample(adata,
                                  sample_size=sample_size,
                                  seed=seed)

    # check treat_var
    assert treat_var in adata.obs.columns, f"treat_var not in .obs: {treat_var}"

    # create RFE object
    svc = SVC(kernel=kernel, C=C)
    rfe = RFE(estimator=svc, **kwargs)
    if subsample:
        y = adata_ss.obs[treat_var].to_numpy()
        selector = rfe.fit(adata_ss.X, y)
    else:
        y = adata.obs[treat_var].to_numpy()
        selector = rfe.fit(adata.X, y)

    mask = selector.support_
    drop_feats = adata.var_names[~mask]

    if drop:
        adata = adata[:, mask].copy()
        adata.uns['svm_rfe_feats'] = drop_feats
    else:
        adata.var['svm_rfe_feats'] = ~mask

    if verbose:
        print(f"Dropped {len(drop_feats)} with low weights from SVC: {drop_feats}")

    return adata


def lmm_feat_select(adata,
                    treat_var='Metadata_Treatment',
                    conc_var=None,
                    group_var='BatchNumber',
                    time_var='Metadata_Time',
                    n_features_to_select=None,
                    subsample=False,
                    sample_size=1000,
                    seed=0,
                    drop=True,
                    verbose=False,
                    **kwargs):
    """Feature selection based on linear mixed models.

    For every variable in .X a lmm is calculated with the variable
    as dependent, another specified observation variable as fixed effect
    and a group observation variable as random effect.
    The models with best sum of squared residuals are selected.


    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        treat_var (str): Treatment variable in .obs as fixed effect.
        conc_var (str): Concentration variable in .obs as fixed effect.
        group_var (str): Variable in .obs to group data for
            random effects.
        time_var (str): Time variable in .obs to add as fixed effect.
        n_features_to_select (int): Number of features to select.
            If None, half of the features are selected.
        subsample (bool): If True, fit models on subsample of data.
        sample_size (int): Size of supsample.
            Only if subsample is True.
        seed (int): Seed for subsample calculation.
            Only if subsample is True.
        drop (bool): Drop features with p-value above threshold.
        verbose (bool)
        **kwargs (dict): Keyword arguments passed to statsmodels.formula.api.mixedlm.fit()

    Returns:
        anndata.AnnData
        .uns['lmm_feats']: Dropped features.
        .var['lmm_feats']: True for features to exclude.
            Only if drop is False.
    """
    # check that variables in by are in anndata
    all_vars = [treat_var, group_var]
    if time_var is not None:
        all_vars.append(time_var)
    if conc_var is not None:
        all_vars.append(conc_var)
    if not all(var in adata.obs.columns for var in all_vars):
        raise KeyError(f"One or all variables not in anndata object: {all_vars}")

    # check if any columns of adata.X contain nan values
    if np.isnan(adata.X).any():
        warnings.warn("NaN values detected in anndata object. "
                      "Tries to drop NaN-containing columns.")
        adata = drop_nan(adata)

    # get subsample
    if subsample:
        adata_ss = _get_subsample(adata,
                                  sample_size=sample_size,
                                  seed=seed)
    else:
        adata_ss = adata.copy()

    # check that fixed vars have more than one category
    if len(adata_ss.obs[treat_var].unique()) == 1:
        raise ValueError(f"Only one category in fixed variables: {adata.obs[treat_var].unique()}.")

    # cache sum of squared residuals for every variable
    ssrs = []

    # cache data to create formula
    data = adata_ss.to_df()
    data['treat'] = adata_ss.obs[treat_var]
    data['group'] = adata_ss.obs[group_var]

    if time_var is not None:
        data['time'] = adata_ss.obs[time_var]
    if conc_var is not None:
        data['conc'] = adata_ss.obs[conc_var]

    # iterate over all variables in .var
    var_list = adata.var_names
    for var in tqdm(var_list, desc=f"Fitting LMMs for {len(var_list)} features."):
        # create linear mixed effect model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            # get formulas
            if time_var is not None:
                if conc_var is not None:
                    formula = f"{var} ~ treat + conc + time"
                else:
                    formula = f"{var} ~ treat + time"
            else:
                formula = f"{var} ~ treat"
            lmm = smf.mixedlm(formula, data=data, groups='group')
            results = lmm.fit(**kwargs)

        # sum of squared residuals
        ssr = np.sum(np.square(results.resid))
        ssrs.append(ssr)

    # select models with best ssr
    ssrs_ix = np.argsort(ssrs)
    if n_features_to_select is not None:
        assert isinstance(n_features_to_select, int), f"n_features_to_select expected to be type(int), " \
                                                      f"instead got {type(n_features_to_select)}"
    else:
        n_features_to_select = len(ssrs_ix) // 2

    ssrs_ix = ssrs_ix[:n_features_to_select]
    mask = np.array([True if ix in ssrs_ix else False for ix, _ in enumerate(adata.var_names)])

    if drop:
        drop_feats = var_list[~mask]
        adata = adata[:, mask].copy()
        adata.uns['lmm_feats'] = drop_feats

        if verbose:
            print(f"Dropped {len(drop_feats)} features: {drop_feats}")
    else:
        adata.var['lmm_feats'] = ~mask

    return adata
