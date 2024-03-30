from typing import Union, Optional
from scipy import linalg

import numpy as np
import anndata as ad


class TVN:
    """
    Handling batch effects with Typical Variation Normalization

    References
    ----------
    Improving Phenotypic Measurements in High-Content Imaging Screens
    D. Michael Ando, Cory Y. McLean, Marc Berndl
    bioRxiv 161422; doi: https://doi.org/10.1101/161422
    """

    def __init__(
        self,
        adata: ad.AnnData,
        treat_key: str,
        control_name: str,
        time_key: Optional[str] = None,
        ref_time: Optional[Union[int, float]] = None,
        epsilon: float = 1e-5,
    ):
        """
        Parameters
        ----------
        adata : anndata.AnnData
            Multidimensional morphological data
        treat_key : str
            Treatment information in .obs
        control_name : str
            Name of control group in .obs[treat_key]
        time_key : str, optional
            Time information in .obs
        ref_time : int, float, optional
            Reference time in .obs[time_key]
        epsilon : float
            Small value to avoid division by zero
        """
        self.adata = adata
        self.treat_key = treat_key
        self.control_name = control_name
        self.time_key = time_key
        self.ref_time = ref_time

        # Other variables
        self.epsilon = epsilon
        self._mean = None
        self._w = None

    def fit(self):
        # Control data
        if self.time_key is not None:
            x = self.adata[
                (self.adata.obs[self.treat_key] == self.control_name)
                & (self.adata.obs[self.time_key] == self.ref_time)
            ].X.copy()
        else:
            x = self.adata[self.adata.obs[self.treat_key] == self.control_name].X.copy()
        # Center data
        self._mean = np.mean(x, axis=0)
        x = x - self._mean

        # Covariance matrix
        cov = np.dot(x.T, x) / x.shape[0]

        # Whitening matrix
        eigen_vectors, eigen_values, _ = np.linalg.svd(cov)
        eigen_values_dm = np.diag(1 / np.sqrt(eigen_values + self.epsilon))
        w = np.dot(eigen_values_dm, eigen_vectors.T)
        self._w = w
        return None

    def transform(self):
        assert (self._w is not None) and (
            self._mean is not None
        ), "class has not been fitted"
        self.adata.X = self.adata.X - self._mean
        self.adata.X = np.dot(self.adata.X, self._w.T)
        return self.adata

    def fit_transform(self):
        self.fit()
        return self.transform()


class CORAL:
    """
    Handling batch effects with Correlation Alignment (CORAL)

    References
    ----------
    Sun, B., Feng, J., & Saenko, K. (2017).
    Correlation alignment for unsupervised domain adaptation.
    Domain adaptation in computer vision applications, 153-171.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        batch_key: str,
        treat_key: str,
        control_name: str,
        time_key: Optional[str] = None,
        ref_time: Optional[Union[int, float]] = None,
        regul_weight: Union[float, int] = 1,
    ):
        """
        Parameters
        ----------
        adata : anndata.AnnData
            Multidimensional morphological data
        batch_key : str
            Batch information in .obs
        treat_key : str
            Treatment information in .obs
        control_name : str
            Name of control group in .obs[treat_key]
        time_key : str, optional
            Time information in .obs
        ref_time : int, float, optional
            Reference time in .obs[time_key]
        regul_weight : float, int
            Weights for the regularization of the covariance matrices
        """
        self.adata = adata
        self.treat_key = treat_key
        self.control_name = control_name
        self.time_key = time_key
        self.ref_time = ref_time

        # Batch information
        self.batch_key = batch_key
        self.batches = adata.obs[batch_key].unique()
        self.n_batches = len(self.batches)

        # Other variables
        self.regul_weight = regul_weight
        self._rt = None
        self._rs = None

    def fit(self):
        # Control data
        # Control data
        if self.time_key is not None:
            x = self.adata[
                (self.adata.obs[self.treat_key] == self.control_name)
                & (self.adata.obs[self.time_key] == self.ref_time)
            ].X.copy()
            b = (
                self.adata[
                    (self.adata.obs[self.treat_key] == self.control_name)
                    & (self.adata.obs[self.time_key] == self.ref_time)
                ]
                .obs[self.batch_key]
                .values
            )
        else:
            x = self.adata[self.adata.obs[self.treat_key] == self.control_name].X.copy()
            b = (
                self.adata[self.adata.obs[self.treat_key] == self.control_name]
                .obs[self.batch_key]
                .values
            )

        # Covariance matrices
        cov = np.cov(x, rowvar=False)
        # Regularization
        cov = cov + (self.regul_weight * np.eye(x.shape[1]))
        # Matrix square root
        cov = linalg.sqrtm(cov)
        # Make real
        if np.iscomplexobj(cov):
            cov = cov.real

        cov_batches = {}
        for batch in self.batches:
            cov_batch = np.cov(x[b == batch, :], rowvar=False)
            cov_batch = cov_batch + (self.regul_weight * np.eye(x.shape[1]))
            cov_batch = linalg.inv(linalg.sqrtm(cov_batch))
            if np.iscomplexobj(cov_batch):
                cov_batch = cov_batch.real
            cov_batches[batch] = cov_batch

        self._rt = cov
        self._rs = cov_batches
        return None

    def transform(self):
        assert (self._rt is not None) and (
            self._rs is not None
        ), "class has not been fitted"
        for batch in self.batches:
            x = self.adata[self.adata.obs[self.batch_key] == batch, :].X.copy()
            x = np.dot(x, self._rs[batch])  # whitening
            x = np.dot(x, self._rt)  # re-coloring
            self.adata[self.adata.obs[self.batch_key] == batch, :].X = x
        return self.adata

    def fit_transform(self):
        self.fit()
        return self.transform()
