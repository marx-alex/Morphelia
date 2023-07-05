from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests

from morphelia.tools import choose_representation


def rank_feats_groups(
        adata,
        group_by: str,
        use_rep: Optional[str] = None,
        n_feats: Optional[int] = None,
        groups: Optional[Sequence] = None,
        reference: Optional[str] = None,
        method: str = 'wilcoxon',
        corr_method: str = 'fdr_bh',
        copy: bool = False
):

    avail_methods = ['t-test', 'wilcoxon']
    assert method in avail_methods, f'Method must be one of {avail_methods}'

    adata = adata.copy() if copy else adata

    # choose groups
    y_groups = adata.obs[group_by].to_numpy()
    unique_groups = np.sort(np.unique(y_groups))
    if groups is not None:
        assert all(g in unique_groups for g in groups), f'All items in {groups} must be in .obs[{group_by}]'
        unique_groups = np.sort(np.array(groups))
    if reference is not None:
        assert reference in unique_groups, f'Reference ({reference}) must be in {unique_groups}'
    groups_mask = np.isin(y_groups, unique_groups)
    y_groups = y_groups[groups_mask]

    # choose features
    X_features = choose_representation(adata, rep=use_rep)
    X_features = X_features[groups_mask, :]
    if use_rep is None:
        feature_names = adata.var_names
    else:
        feature_names = [f'{use_rep}_{i}' for i in range(X_features.shape[1])]
    if n_feats is not None:
        X_features = X_features[:, :n_feats]
        feature_names = feature_names[:n_feats]

    # choose method
    if method == 'wilcoxon':
        test_fn = stats.ranksums
    elif method == 't-test':
        test_fn = stats.ttest_ind
    else:
        raise NotImplementedError(f'Test not implemented: {method}')

    # perform test
    idx_iterables = [unique_groups, ['score', 'pvalue', 'pcorr']]
    result_idx = pd.MultiIndex.from_product(idx_iterables, names=["group", "scores"])
    result = pd.DataFrame(index=result_idx, columns=feature_names)

    for group in tqdm(unique_groups, desc='Perform test for every group'):
        X_group = X_features[y_groups == group, :]
        if reference is not None:
            X_ref = X_features[y_groups == reference, :]
        else:
            X_ref = X_features[y_groups != group]

        with np.errstate(invalid="ignore"):
            scores, pvalues = test_fn(X_group, X_ref, axis=0)

        scores[np.isnan(scores)] = 0
        pvalues[np.isnan(pvalues)] = 1

        # perform correction
        _, pcorr, _, _ = multipletests(
            pvalues, alpha=0.05, method=corr_method
        )

        # store result
        result.loc[(group, 'score')] = scores
        result.loc[(group, 'pvalue')] = pvalues
        result.loc[(group, 'pcorr')] = pcorr

    uns_key = 'rank_feats_groups'
    adata.uns[uns_key] = {}
    adata.uns[uns_key]['params'] = dict(
        group_by=group_by,
        reference=reference,
        groups=unique_groups,
        method=method,
        use_rep=use_rep,
        corr_method=corr_method,
    )

    adata.uns[uns_key]['result'] = result

    return adata if copy else None
