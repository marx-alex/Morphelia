# internal libraries
import warnings
import logging
from morphelia.preprocessing.basic import drop_nan

# external libraries
import statsmodels.formula.api as smf
from tqdm import tqdm
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

from morphelia.tools.utils import get_subsample

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


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

    Recursively Remove features with low weights from classification with SVM.
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
        adata_ss = get_subsample(adata,
                                 sample_size=sample_size,
                                 seed=seed)
    else:
        adata_ss = adata.copy()

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
        logger.info(f"Dropped {len(drop_feats)} with low weights from SVC: {drop_feats}")

    return adata
