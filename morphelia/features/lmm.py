# internal libraries
import warnings
from morphelia.preprocessing.basic import drop_nan
import os

# external libraries
import statsmodels.api as sm
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def lmm_feat_select(adata,
                    fixed_var='Metadata_Treatment',
                    group_var='BatchNumber',
                    time_var='Metadata_Time',
                    time_series=False,
                    subsample=1000,
                    seed=0,
                    drop=False,
                    p_thresh=0.05,
                    show=False,
                    save=False):
    """Calculates a linear mixed effect model with a single variable as
    dependent, batch as grouping and treatments as indipendent variables.
    The null hypothesis is then tested, if all treatment trends are the same.
    The alternative is that at least one trend is different than the others.
    This is done by wald test using a F-distribution.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        fixed_var (str): Variable in adata.obs for fixed effects.
        group_var (str): Variable in adata.obs to group data for
            random effects.
        time_var (str): Variable in adata.obs to add as effect.
        time_series (bool): Includes time_var in model if True.
        subsample (int): If given, retrieves subsample of all cell data for speed.
        seed (int): Seed for subsample calculation.
        drop (bool): Drop features with p-value above threshold.
        p_thresh (float): Value to use as threshold if drop is True.
        show (bool): True to get figure.
        save (str): Path where to save figure.

    Returns:
        anndata.AnnData
        .var['lmm_wald_p']: p-values from wald test for every feature.
        .uns['lmm_dropped']: Features that have been dropped due to a p_value above p_thresh.
            Only if drop is True.
    """
    # check that variables in by are in anndata
    if time_series:
        all_vars = [fixed_var, group_var, time_var]
    else:
        all_vars = [fixed_var, group_var]
    if not all(var in adata.obs.columns for var in all_vars):
        raise KeyError(f"One or all variables not in anndata object: {all_vars}")

    # check if any columns of adata.X contain nan values
    if np.isnan(adata.X).any():
        warnings.warn("NaN values detected in anndata object. "
                      "Tries to drop NaN-containing columns.")
        adata = drop_nan(adata)

    # get subsample
    if subsample is not None:
        assert isinstance(subsample, int), f"expected type for subsample is int, instead got {type(subsample)}"
        # get samples
        np.random.seed(seed)
        X_len = adata.shape[0]
        if subsample > X_len:
            subsample = X_len
        sample_ix = np.random.randint(X_len, size=subsample)
        try:
            adata_ss = adata[sample_ix, :]
        except:
            adata_ss = adata.X.copy()
    else:
        adata_ss = adata.X.copy()

    # check that fixed vars have more than one category
    if len(adata_ss.obs[fixed_var].unique()) == 1:
        raise ValueError(f"Only one category in fixed variables: {adata.obs[fixed_var].unique()}.")

    # cache p-values from wald tests of joint hypothesis testing
    wald_ps = []

    # cache data to create formula
    data = pd.DataFrame({'fixed': adata_ss.obs[fixed_var],
                         'group': adata_ss.obs[group_var]})

    if time_series:
        data['time'] = adata_ss.obs[time_var]

    # iterate over all variables in adata.var
    var_list = adata.var_names
    for var in tqdm(var_list, desc=f"Fitting LMMs for {len(var_list)} features."):
        data['var'] = adata_ss[:, var].X

        # create linear mixed effect model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            if time_series:
                lmm = sm.MixedLM.from_formula('var ~ fixed : time', data=data, groups='group')
            else:
                lmm = sm.MixedLM.from_formula('var ~ fixed', data=data, groups='group')
            results = lmm.fit(method='lbfgs')

        first_treat = lmm.exog_names[1]
        contrasts = [f"{treat} - {first_treat}" for treat in lmm.exog_names[2:]]
        contrasts = ", ".join(contrasts)

        wald_str = results.wald_test(contrasts).summary()
        wald_p = float(wald_str.split(',')[1].strip().split('=')[1])
        wald_ps.append(wald_p)

    # add p-value from wald test to variables
    adata.var['lmm_wald_p'] = wald_ps
    if drop:
        assert 0 <= p_thresh <= 1, 'p_thresh expected to be a value between 0 and 1, ' \
                                  f'instead got {p_thresh}'
        drop_feats = [feat for feat, p in zip(adata.var_names, wald_ps) if p > p_thresh]
        adata = adata[:, adata.var['lmm_wald_p'] < p_thresh].copy()
        adata.uns['lmm_dropped'] = drop_feats

    # plotting
    if show:
        sns.set_theme()
        feat_ixs = list(range(adata.shape[1]))
        legend = ['Treatment effect' if elem < 0.05 else 'No treatment effect' for elem in wald_ps]
        
        plt.figure(figsize=(8, 6))
        pl = sns.scatterplot(x=feat_ixs, y=wald_ps, hue=legend, palette=sns.color_palette('husl', 2)[::-1],
                             alpha=0.6, edgecolors=None)
        plt.yscale('log')
        plt.gca().invert_yaxis()
        pl.set_xlabel('Feature')
        pl.set_ylabel('p-value (Wald test)')
        plt.title(f'LMM Feature Selection on subsample of {subsample} cells')
        plt.axhline(0.05, linestyle='dotted', color='black', label=f'Significance level: {0.05}')
        lgd = plt.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False)

        # save
        if save:
            try:
                plt.savefig(os.path.join(save, "feature_correlation.png"),
                            bbox_extra_artists=[lgd], bbox_inches='tight')
            except OSError:
                print(f'Can not save figure to {save}.')

    return adata
