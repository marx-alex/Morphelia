import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import median_abs_deviation as mad


class RobustMAD(BaseEstimator, TransformerMixin):
    """
    Class to perform a "Robust" normalization with respect to median and mad
        scaled = (x - median) / mad

    Class is adopted from pycytominer:
    https://github.com/cytomining/pycytominer/blob/master/pycytominer/operations/transform.py
    """

    def __init__(self, scale="normal", eps=0):
        self.scale = scale
        self.eps = eps

    def fit(self, X):
        """
        Compute the median and mad to be used for later scaling.

        Args:
        X (numpy.ndarray): Array to fit with transform by RobustMAD
        """
        # Get the mean of the features (columns) and center if specified
        self.median = np.nanmedian(X, axis=0)
        self.mad = mad(X, axis=0, nan_policy="omit", scale=self.scale)
        return self

    def transform(self, X):
        """
        Apply the RobustMAD calculation.

        Args:
        X (numpy.ndarray): Array to apply RobustMAD scaling.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return (X - self.median) / (self.mad + self.eps)


class MedianPolish:
    """Fits an additive model using Tukey's median polish algorithm.
    This class is taken from borisvish: https://github.com/borisvish/Median-Polish
    """

    def __init__(self, max_iterations=10, method="median"):
        """Store values for maximum iterations and method."""
        self.max_iterations = max_iterations
        self.method = method

    def median_polish(self, X):
        """
        Implements Tukey's median polish alghoritm for additive models
        method - default is median, alternative is mean. That would give us result equal ANOVA.
        """

        if isinstance(X, np.ndarray):
            X_org = X
            X = X_org.copy()
        else:
            raise TypeError("Expected the argument to be a numpy.ndarray.")

        grand_effect = 0
        median_row_effects = 0
        median_col_effects = 0
        row_effects = np.zeros(shape=X.shape[0])
        col_effects = np.zeros(shape=X.shape[1])

        for i in range(self.max_iterations):
            if self.method == "median":
                row_medians = np.median(X, 1)
                row_effects += row_medians
                median_row_effects = np.median(row_effects)
            elif self.method == "average":
                row_medians = np.average(X, 1)
                row_effects += row_medians
                median_row_effects = np.average(row_effects)
            grand_effect += median_row_effects
            row_effects -= median_row_effects
            X -= row_medians[:, np.newaxis]

            if self.method == "median":
                col_medians = np.median(X, 0)
                col_effects += col_medians
                median_col_effects = np.median(col_effects)
            elif self.method == "average":
                col_medians = np.average(X, 0)
                col_effects += col_medians
                median_col_effects = np.average(col_effects)

            X -= col_medians

            grand_effect += median_col_effects

        return grand_effect, col_effects, row_effects, X, X_org
