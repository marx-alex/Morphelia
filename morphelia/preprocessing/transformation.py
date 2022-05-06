import numpy as np
import anndata as ad
from sklearn.preprocessing import QuantileTransformer, PowerTransformer


def transform(adata: ad.AnnData, method: str = "power", **kwargs):
    """Feature transformation.

    Transformation of not normally distributed data.
    Several methods are available.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    method : str
        One of the following method to use for transformation:
        `sqrt`: Square root transformation
        `log`: Logarithmic transformation
        `log1p`: Natural logarithm of one plus the input array log(1 + x)
        `quantil`: Quantile transformation
        `power`: Power transformation (Yeo-Johnson and Box-Cox, default is Box-Cox)
    ** kwargs
        Arguments passed to transformer.

    Returns
    -------
    anndata.AnnData
        Transformed AnnData object

    Raises
    -------
    AssertionError
        If method is unknown

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np

    >>> data = np.random.rand(5, 1)
    >>> adata = ad.AnnData(data)
    >>> adata.X
    array([[0.7252543 ],
           [0.50132436],
           [0.95608366],
           [0.6439902 ],
           [0.42385504]], dtype=float32)

    >>> adata = mp.pp.transform(
    >>>     adata,
    >>>     method="log"
    >>> )
    >>> adata.X
    array([[-0.32123291],
           [-0.690502  ],
           [-0.04490986],
           [-0.44007176],
           [-0.85836375]], dtype=float32)
    """
    # define transformer
    method = method.lower()

    avail_methods = ["sqrt", "log", "log1p", "quantile", "power"]
    assert method in avail_methods, (
        f"Method must be one of {avail_methods}, " f"instead got {method}"
    )

    if method == "sqrt":
        adata.X = np.sqrt(adata.X, **kwargs)

    elif method == "log":
        adata.X = np.log(adata.X, **kwargs)

    elif method == "log1p":
        adata.X = np.log1p(adata.X, **kwargs)

    elif method == "quantile":
        kwargs.setdefault("random_state", 0)
        transformer = QuantileTransformer(**kwargs)
        adata.X = transformer.fit_transform(adata.X)

    elif method == "power":
        kwargs.setdefault("method", "box_cox")
        kwargs.setdefault("standardize", False)
        transformer = PowerTransformer(**kwargs)
        adata.X = transformer.fit_transform(adata.X)

    return adata
