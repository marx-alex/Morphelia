# internal libraries
import logging

# external libraries
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import anndata as ad

from morphelia.tools.utils import get_subsample

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def svm_rfe(
    adata: ad.AnnData,
    treat_var: str = "Metadata_Treatment",
    kernel: str = "linear",
    C: int = 1,
    subsample: bool = False,
    sample_size: int = 1000,
    seed: int = 0,
    drop: bool = True,
    verbose: bool = False,
    **kwargs,
) -> ad.AnnData:
    """Support-vector-machine-based recursive-feature elimination.

    Recursively remove features with low weights from classification with SVM.
    Features are trained to classify `treat_var` (i.g. treatment condition in `.obs`).

    Results are stored as:
        .uns['svm_rfe_feats']: Dropped features with low importance from classification task.

        .var['svm_rfe_feats']: True for features with low importance. Only if drop is False.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    treat_var : str
        Name of annotation with treatments
    C : float
        Regularization parameter for SVC
    subsample : bool
        If True, fit models on subsample of data
    sample_size : int
        Size of supsample. Only if subsample is True.
    seed : int
        Seed for subsample calculation. Only if subsample is True.
    kernel : str
        Kernel type to be used for SVC
    drop : bool
        Drop least important features
    verbose : bool
    **kwargs
        Keyword arguments passed to `sklearn.feature_selection.RFE`

    Returns
    -------
    anndata.AnnData
        AnnData object wihtout dropped features if `drop` is True

    Raises
    -------
    AssertionError
        If `treat_var` is not in `.obs`

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np
    >>> import pandas as pd

    >>> data = np.random.rand(12, 5)
    >>> obs = pd.DataFrame({
    >>>     'treatment': [
    >>>         'ctrl', 'ctrl', 'ctrl', 'ctrl', 'ctrl', 'ctrl',
    >>>         'adrenalin', 'adrenalin', 'adrenalin', 'adrenalin', 'adrenalin', 'adrenalin'
    >>>     ]
    >>> })
    >>> adata = ad.AnnData(data, obs=obs)
    >>> mp.ft.svm_rfe(adata, treat_var='treatment')
    AnnData object with n_obs × n_vars = 12 × 2
        obs: 'treatment'
        uns: 'svm_rfe_feats'
    """
    # get subsample
    if subsample:
        adata_ss = get_subsample(adata, sample_size=sample_size, seed=seed)
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
        adata.uns["svm_rfe_feats"] = drop_feats
    else:
        adata.var["svm_rfe_feats"] = ~mask

    if verbose:
        logger.info(
            f"Dropped {len(drop_feats)} with low weights from SVC: {drop_feats}"
        )

    return adata
