# internal libraries
import warnings
from morphelia.preprocessing.pp import drop_nan

# external libraries
from statsmodels.formula.api import mixedlm
from tqdm import tqdm
import numpy as np


def lmm_feat_select(adata, fixed_var='Metadata_Treatment', group_var='BatchNumber',
                    time_var='Metadata_Time', time_series=False):
    """Calculates a linear mixed effect model with a single variable as
    dependent, batch as grouping and treatments as indipendent variables.
    The null hypothesis is then tested, if all treatments trends are the same.
    The alternative is that at least one trend is different than the others.
    This is done by wald test using a F-distribution.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        fixed_var (str): Variable in adata.obs for fixed effects.
        group_var (str): Variable in adata.obs to group data for
            random effects.
        time_var (str): Variable in adata.obs to add as effect.
        time_series (bool): Includes time_var in model if True.
    """
    # check that variables in by are in anndata
    if time_series:
        all_vars = [fixed_var, group_var, time_var]
    else:
        all_vars = [fixed_var, group_var]
    if not all(var in adata.obs.columns for var in all_vars):
        raise KeyError(f"One or all variables not in anndata object: {all_vars}")

    # check if any columns of adata.X contain nan values
    if np.isnan:
        warnings.warn("NaN values detected in anndata object. "
                      "Tries to drop NaN-containing columns.")
        adata = drop_nan(adata)

    # cache p-values from wald tests of joint hypothesis
    wald_ps = []

    # cache data to create formula
    data = {'fixed': adata.obs[fixed_var].astype('category'),
            'group': adata.obs[group_var].astype('category')}
    if time_series:
        data['time'] = adata.obs[time_var]

    # iterate over all variables in adata.var
    var_list = adata.var_names
    for var in tqdm(var_list, desc=f"Fitting LMMs for {len(var_list)} features."):
        data['var'] = adata[:, var].X

        # create linear mixed effect model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            if time_series:
                lmm = mixedlm('var ~ fixed : time', data, groups='group')
            else:
                lmm = mixedlm('var ~ fixed', data, groups='group')
            results = lmm.fit(method='lbfgs')

        first_treat = lmm.exog_names[1]
        contrasts = [f"{treat} - {first_treat}" for treat in lmm.exog_names[2:]]
        contrasts = ", ".join(contrasts)

        wald_str = results.wald_test(contrasts).summary()
        wald_p = float(wald_str.split(',')[1].strip().split('=')[1])
        wald_ps.append(wald_p)

    # add p-value from wald test to variables
    adata.var['wald_p'] = wald_ps

    return adata
