import logging
from typing import Optional

import scanpy as sc
import numpy as np
import anndata as ad
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def choose_representation(
    adata: ad.AnnData, rep: Optional[str] = None, n_pcs: Optional[int] = None
) -> np.ndarray:
    """Fetch a representation from an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    rep : str, optional
        Key in `.obsm`
    n_pcs : int, optional
        Number of principal components to return if representation is `X_pca`

    Returns
    -------
    numpy.ndarray
        The chosen representation of the AnnData object

    Raises
    -------
    ValueError
        If `rep` is None and `X_pca` is not in `.obsm`
    ValueError
        If `rep` is not None and not in `.obsm`

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np

    >>> data = np.random.rand(5, 5)
    >>> rep = np.random.rand(5, 5)  # synthetic representation of the data
    >>> adata = ad.AnnData(data)
    >>> adata.obsm['X_rep'] = rep  # add representation to adata

    >>> fetched_rep = mp.tl.choose_representation(adata, rep='X_rep')  # fetch X_rep from adata
    >>> np.all(fetched_rep == rep)  # fetched representation should equal the original one
    True
    """
    # use X_pca by default
    if rep is None:
        if "X_pca" not in adata.obsm.keys():
            logger.warning("Found no PC representation. Trying to compute PCA...")
            sc.tl.pca(adata)

        if "X_pca" in adata.obsm.keys():

            if n_pcs is not None and n_pcs > adata.obsm["X_pca"].shape[1]:
                logger.warning(
                    f"Number n_pcs {n_pcs} is larger than PCs in X_pca, "
                    f"use number of PCs in X_pca instead {adata.obsm['X_pca'].shape[1]}"
                )
                n_pcs = adata.obsm["X_pca"].shape[1]

            # return .X if rep is None
            if n_pcs == 0:
                X = adata.X.copy()
            else:
                # return pcs
                X = adata.obsm["X_pca"][:, :n_pcs].copy()

        else:
            raise ValueError("Did not found X_pca in .obsm")

    else:
        if rep == "X_pca":
            if n_pcs is not None:
                X = adata.obsm[rep][:, :n_pcs].copy()
            else:
                X = adata.obsm[rep].copy()

        elif rep in adata.obsm.keys():
            X = adata.obsm[rep].copy()

        elif rep == "X":
            X = adata.X.copy()

        else:
            raise ValueError(f"Did not find rep in .obsm: {rep}")

    return X


def choose_layer(
    adata: ad.AnnData, obsm: Optional[str] = None, copy=False
) -> np.ndarray:
    """Fetch a layer from an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    obsm : str, optional
        Key in `.obsm`
    copy : bool
        Return a copy

    Returns
    -------
    numpy.ndarray
        The chosen layer of the AnnData object
    """
    if obsm is not None:
        if copy:
            return adata.obsm[obsm].copy()
        else:
            return adata.obsm[obsm]

    if copy:
        return adata.X.copy()
    else:
        return adata.X


def get_subsample(
    adata: ad.AnnData, sample_size: Optional[int] = None, seed: int = 0
) -> ad.AnnData:
    """Draws n (sample_size) random samples from an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    sample_size : int, optional
        Number of samples
    seed : int
        Seed for reproducibility of subsample

    Returns
    -------
    anndata.AnnData
        Subsample of the original AnnData object

    Raises
    -------
    AssertionError
        If `sample_size` is not an integer

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np

    >>> data = np.random.rand(5, 5)
    >>> adata = ad.AnnData(data)
    >>> sample = mp.tl.get_subsample(adata, sample_size=3)  # sample of size 3
    >>> sample
    AnnData object with n_obs × n_vars = 3 × 5
    """
    if sample_size is None:
        return adata
    else:
        assert isinstance(sample_size, int), (
            f"expected type(int) for sample_size, " f"instead got {type(sample_size)}"
        )
        # get samples
        np.random.seed(seed)
        N = len(adata)
        if sample_size >= N:
            logger.warning(
                "sample_size exceeds available samples, draws all samples instead."
            )
            return adata
        else:
            rng = np.random.default_rng()
            sample_ixs = rng.choice(N, size=sample_size, replace=False)
            adata_ss = adata[sample_ixs, :].copy()

    return adata_ss


def encode_labels(
    adata: ad.AnnData, key: str = "Metadata_Treatment", sfx: str = "_Enc"
) -> ad.AnnData:
    """Label encoding of any categorical variable in `.obs`.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    key : str
        Name of observation in .obs
    sfx : str
        Suffix for new observation

    Returns
    -------
    anndata.AnnData
        AnnData object with encoded observation

    Raises
    ------
    AssertionError
        If `key` is not in `.obs`

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np
    >>> import pandas as pd

    >>> data = np.random.rand(5, 5)
    >>> obs = pd.DataFrame({
    >>>     'category': ['cell1', 'cell1', 'cell1', 'cell2', 'cell2'],
    >>> })
    >>> adata = ad.AnnData(data, obs=obs)
    >>> adata.strings_to_categoricals()

    >>> adata = mp.tl.encode_labels(adata, key='category')
    >>> adata.obs
        category	category_Enc
    0	cell1	    0
    1	cell1	    0
    2	cell1	    0
    3	cell2	    1
    4	cell2	    1
    """
    assert key in adata.obs.columns, f"label_var not in .obs: {key}"
    x = adata.obs[key].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(x)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    new_var = key + sfx
    adata.obs[new_var] = y
    if "le_map" in adata.uns:
        adata.uns["le_map"][key] = le_name_mapping
    else:
        adata.uns["le_map"] = {key: le_name_mapping}
    return adata
