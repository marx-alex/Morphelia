import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import median_abs_deviation as mad

from typing import Union


class RobustMAD(BaseEstimator, TransformerMixin):
    """Robust data transformation.

    Class to perform a "Robust" normalization by using
    the median and median absolute deviation.

    .. math::

        scaled = \\frac{(x - median)}{mad}

    Parameters
    ----------
    scale : str
        Scale passed to scipy.stats.median_abs_deviation
    eps : float or int
        Avoids division by zero

    References
    ----------
    .. [1] Caicedo, J., Cooper, S., Heigwer, F. et al. Data-analysis strategies for image-based cell profiling.
       Nat Methods 14, 849–863 (2017). https://doi.org/10.1038/nmeth.4397
    .. [2] https://github.com/cytomining/pycytominer/blob/master/pycytominer/operations/transform.py

    Examples
    --------
    >>> import morphelia as mp
    >>> import numpy as np

    >>> data = np.random.rand(3, 3)
    >>> scaler = mp.tl.RobustMAD()
    >>> scaler.fit(data)
    >>> scaled = scaler.transform(data)
    >>> scaled
    array([[ 0.90040852, -4.05326241,  0.        ],
           [-0.67448975,  0.67448975,  0.67448975],
           [ 0.        ,  0.        , -0.7910895 ]])
    """

    def __init__(self, scale: str = "normal", eps: Union[float, int] = 0) -> None:
        self.scale = scale
        self.eps = eps

    def fit(self, X: np.ndarray) -> None:
        """Computes the median and mad for every feature.

        Parameters
        ----------
        X : numpy.ndarray
            Array to fit

        Returns
        -------
        None
        """
        # Get the mean of the features (columns) and center if specified
        self.median = np.nanmedian(X, axis=0)
        self.mad = mad(X, axis=0, nan_policy="omit", scale=self.scale)
        return None

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Applies RobustMAD and transforms the data

        Parameters
        ----------
        X : numpy.ndarray
            Array that will be transformed
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return (X - self.median) / (self.mad + self.eps)


class MedianPolish:
    """Fits an additive model using Tukey's median polish algorithm.

    Parameters
    ----------
    max_iterations : int
        Maximum number of iterations
    method : str
        Uses either `mean` or `median` to aggregate the data

    References
    ----------
    .. [1] https://github.com/borisvish/Median-Polish
    """

    def __init__(self, max_iterations: int = 10, method: str = "median") -> None:
        self.max_iterations = max_iterations
        avail_methods = ["median", "mean"]
        method = method.lower()
        assert method in avail_methods, f"method not in {avail_methods}: {method}"
        self.method = method

    def median_polish(self, X: np.ndarray):
        """Implements Tukey's median polish alghoritm for additive models

        The default is median aggregation, alternative is mean aggregation can be used.
        The latter would give results equal ANOVA.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be transformed

        Returns
        -------
        int, int, int, numpy.ndarray, numpy.ndarray
            Grand effect, row effect, column effect, transformed data, original data
        """

        if isinstance(X, np.ndarray):
            X_org = X
            X = X_org.copy()
        else:
            raise TypeError("Expected the argument to be a numpy.ndarray.")

        grand_effect = 0
        avg_row_effects = 0
        avg_col_effects = 0
        row_effects = np.zeros(shape=X.shape[0])
        col_effects = np.zeros(shape=X.shape[1])

        for i in range(self.max_iterations):
            if self.method == "median":
                row_avg = np.median(X, 1)
                row_effects += row_avg
                avg_row_effects = np.median(row_effects)
            elif self.method == "mean":
                row_avg = np.mean(X, 1)
                row_effects += row_avg
                avg_row_effects = np.mean(row_effects)
            grand_effect += avg_row_effects
            row_effects -= avg_row_effects
            X -= row_avg[:, np.newaxis]

            if self.method == "median":
                col_avg = np.median(X, 0)
                col_effects += col_avg
                avg_col_effects = np.median(col_effects)
            elif self.method == "mean":
                col_avg = np.mean(X, 0)
                col_effects += col_avg
                avg_col_effects = np.mean(col_effects)

            X -= col_avg

            grand_effect += avg_col_effects

        return grand_effect, col_effects, row_effects, X, X_org
