import numpy as np
from sklearn.preprocessing import QuantileTransformer, PowerTransformer


def transform(adata, method="power", **kwargs):
    """
    Transformation of not normally distributed data.
    Several methods are available.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        method (str): One of the following method to use for transformation:
            sqrt: Square root transformation.
            log: Logarithmic transformation.
            log1p: Natural logarithm of one plus the input array log(1 + x).
            quantile: Quantile transformation.
            power: Power transformation (Yeo-Johnson and Box-Cox, default is Box-Cox).
        ** kwargs: Arguments passed to transformer.
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
