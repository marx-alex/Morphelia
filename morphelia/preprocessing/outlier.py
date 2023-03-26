import numpy as np
import logging
from typing import Union, Tuple, Optional

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import SGDOneClassSVM
from sklearn.svm import OneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
import anndata as ad

from morphelia.tools.utils import choose_representation

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def outlier_detection(
    adata: ad.AnnData,
    method: str = "if",
    use_rep: Optional[str] = None,
    n_pcs: int = 50,
    drop: bool = True,
    verbose: bool = False,
    nystroem_kwargs: Optional[dict] = None,
    **kwargs,
) -> Union[ad.AnnData, Tuple[ad.AnnData, np.ndarray]]:
    """Simple wrapper for sklearn's IsolationForest.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    method : str
        Outlier detection method. The following scikit-learn algorithms are available:
        Robust covariance (`rc`), One-Class SVM, (`oc-svm`), One-Class SVM (SGD) (`oc-svm-sgd`),
        Isolation Forest (`if`), Local Outlier Factor (`lof`)
    use_rep : str
        Representation to use for outlier detection
    n_pcs : int
        Numer of principal components if representation is `X_pca`
    drop : bool
        Drop outliers
    verbose : bool
    nystroem_kwargs : dict
        If method is One-Class SVM (SGD) a approximated feature map is constructed for an arbitrary kernel.
        These keyword arguments are passed to the kernel map function.
    kwargs
        Keyword arguments for `sklearn.ensemble.IsolationForest`

    Returns
    -------
    anndata.AnnData
        AnnData object without dropped features if `drop` is True

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np

    >>> data = np.random.rand(10, 5)
    >>> adata = ad.AnnData(data)
    >>> mp.pp.outlier_detection(adata)
    AnnData object with n_obs × n_vars = 5 × 5
    """
    if nystroem_kwargs is None:
        nystroem_kwargs = dict()
    nystroem_kwargs.setdefault("random_state", 0)
    nystroem_kwargs.setdefault("gamma", 0.1)

    avail_methods = ["rc", "oc-svm", "oc-svm-sgd", "if", "lof"]
    assert (
        method in avail_methods
    ), f"method must be one of {avail_methods}, instead got {method}"
    if method == "rc":
        kwargs.setdefault("random_state", 0)
        clf = EllipticEnvelope(**kwargs)
    elif method == "oc-svm":
        clf = OneClassSVM(**kwargs)
    elif method == "oc-svm-sgd":
        kwargs.setdefault("random_state", 0)
        clf = make_pipeline(Nystroem(**nystroem_kwargs), SGDOneClassSVM(**kwargs))
    elif method == "if":
        kwargs.setdefault("random_state", 0)
        clf = IsolationForest(**kwargs)
    elif method == "lof":
        clf = LocalOutlierFactor(**kwargs)
    else:
        raise NotImplementedError(f"Method {method} not implemented: {avail_methods}")

    if use_rep is not None:
        X = choose_representation(adata, rep=use_rep, n_pcs=n_pcs)
    else:
        X = adata.X

    o = clf.fit_predict(X)

    o = np.clip(o, a_min=0, a_max=None).astype(bool)

    if verbose:
        logger.info(
            f"Outliers: {np.sum(~o)}, Not Outlier: {np.sum(o)}, Outlier Fraction: {np.sum(~o)/len(o):.5f}"
        )

    if drop:
        adata = adata[o, :].copy()
        return adata
    else:
        return adata, o
