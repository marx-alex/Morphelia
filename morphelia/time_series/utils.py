import logging
from typing import Optional, Union, Tuple

import numpy as np
import anndata as ad

from morphelia.tools import choose_representation

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def make_3d(
    adata: ad.AnnData,
    time_var: str = "Metadata_Time",
    tree_var: str = "Metadata_Trace_Tree",
    use_rep: Optional[str] = None,
    n_pcs: int = 50,
):
    """Return three dimensional representation of data.

    If the AnnData object contains time-series information,
    knowing the time and lineage tree of each cell, the data can
    be transformed to an 3d-array.
    This function can make a 3d-array from the `.X` or any representation
    in `.obsm`.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological time-series data
    time_var : str
        Variable in `.obs` with time information
    tree_var : str
        Variable in `.obs` with lineage information
    use_rep : str, optional
        Representation of the AnnData object to use.
        Used `.X` by default.
    n_pcs : int
        Number of PCs to use if use_rep is `X_pca`

    Returns
    -------
    numpy.array
        Array of shape `[samples, time, features]`

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np
    >>> import pandas as pd

    >>> data = np.random.rand(6, 5)
    >>> obs = pd.DataFrame({
    >>>     'time': [0, 1, 2, 0, 1, 2],
    >>>     'lineage': [0, 0, 0, 1, 1, 1]
    >>> })
    >>> adata = ad.AnnData(data, obs=obs)

    >>> three_d = mp.ts.make_3d(adata, time_var='time', tree_var='lineage')
    >>> three_d.shape
    (2, 3, 5)
    """
    adata = adata[np.isfinite(adata.obs[tree_var])]

    if use_rep is None:
        use_rep = "X"
    X = choose_representation(adata, rep=use_rep, n_pcs=n_pcs)

    trees, trees_ix = np.unique(adata.obs[tree_var], return_inverse=True)
    time, time_ix = np.unique(adata.obs[time_var], return_inverse=True)

    X_traj = np.zeros((len(trees), len(time), adata.n_vars), dtype=X.dtype)
    X_traj[trees_ix, time_ix, :] = X

    return X_traj


class Adata3D:
    """Create a three-dimensional representation of time-series data.

    If the AnnData object contains time-series information,
    knowing the time and lineage tree of each cell, the data can
    be transformed to an 3d-array.
    This class can make a 3d-array from the `.X` or any representation
    in `.obsm`.

    Parameters
    ----------
    time_var : str
        Variable in `.obs` with time information
    tree_var : str
        Variable in `.obs` with lineage information
    use_rep : str, optional
        Representation of the AnnData object to use.
        Used `.X` by default.
    y_var : str, optional
        A label in `.obs` that should be transformed to 3D
        similarly to the representation.
        If `y_var` is not None `to_3d` returns two arrays.
    n_pcs : int
        Number of principal components to use if use_rep is "X_pca"

    Attributes
    ----------
    y : numpy.ndarray or None
        An observation from `.obs`
    X : numpy.ndarray
        A representation from the AnnData object
    n_vars : int
        Number of variables of `X`
    trees : numpy.ndarray
        Unique lineage trees
    trees_ix : numpy.ndarray
        Indices of unique lineage trees
    time : numpy.ndarray
        Unique time points
    time_ix : numpy.ndarray
        Indices of unique time points

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np
    >>> import pandas as pd

    >>> data = np.random.rand(6, 5)
    >>> obs = pd.DataFrame({
    >>>     'time': [0, 1, 2, 0, 1, 2],
    >>>     'lineage': [0, 0, 0, 1, 1, 1]
    >>> })
    >>> adata = ad.AnnData(data, obs=obs)

    >>> converter = mp.ts.Adata3D(adata, time_var='time', tree_var='lineage')
    >>> three_d = converter.to_3d()
    >>> three_d.shape
    (2, 3, 5)

    >>> two_d = converter.to_2d(three_d)
    >>> two_d.shape
    (6, 5)
    """

    def __init__(
        self,
        adata: ad.AnnData,
        time_var: str = "Metadata_Time",
        tree_var: str = "Metadata_Trace_Tree",
        use_rep: Optional[str] = None,
        y_var: Optional[str] = None,
        n_pcs: int = 50,
    ) -> None:
        adata = adata[np.isfinite(adata.obs[tree_var])]
        sorted_ix = adata.obs[time_var].argsort()
        adata = adata[sorted_ix, :]
        self.y = None
        if y_var is not None:
            assert y_var in adata.obs.columns, f"y_var not found in .obs: {y_var}"
            self.y = adata.obs[y_var].to_numpy()

        if use_rep is None:
            use_rep = "X"
        self.X = choose_representation(adata, rep=use_rep, n_pcs=n_pcs)
        self.n_vars = self.X.shape[-1]

        self.trees, self.trees_ix = np.unique(adata.obs[tree_var], return_inverse=True)
        self.time, self.time_ix = np.unique(adata.obs[time_var], return_inverse=True)

    def to_3d(
        self, return_y: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Convert data representation to 3D.

        Convert representation of shape `[samples, features]`
        to shape `[samples, time, features]`.

        Notes
        --------
        Each sample and time point should only have one datapoint.

        Parameters
        ----------
        return_y : bool
            Return annotation (`y`) as 3d array

        Returns
        -------
        numpy.ndarray
            Three dimensional representation if shape `[samples, time, features]`

        Raises
        -------
        AssertionError
            If annotation (`y`) are not consistent within lineage trees
        """
        X_3d = np.zeros(
            (len(self.trees), len(self.time), self.n_vars), dtype=self.X.dtype
        )
        X_3d[self.trees_ix, self.time_ix, :] = self.X

        if return_y:
            y_2d = np.zeros((len(self.trees), len(self.time)), dtype=self.y.dtype)
            y_2d[self.trees_ix, self.time_ix] = self.y
            assert np.all(
                y_2d.T == y_2d.T[0, :]
            ), "y label not consistent within samples!"
            y = y_2d[:, 0].flatten()
            return X_3d, y

        return X_3d

    def to_2d(self, X_3d: np.ndarray) -> np.ndarray:
        """Convert 3D-representation back to 2D.

        3D-representation of the AnnData object with shape `[samples, time, features]`
        is converted back to shape `[samples x features]` with this method.

        Notes
        --------
        Each sample and time point should only have one datapoint.

        Parameters
        ----------
        X_3d : numpy.ndarray
            Array of shape `[samples, time, features]`

        Returns
        -------
        numpy.ndarray
            2D representation of shape `[samples, features]`
        """
        assert (
            len(X_3d.shape) == 3
        ), f"Shape of X_3d must be three-dimensional, instead got shape {X_3d.shape}"

        X_2d = X_3d[self.trees_ix, self.time_ix, :].copy()

        return X_2d


def vectorize_emb(
    adata: ad.AnnData,
    use_rep: Optional[str] = None,
    n_pcs: Optional[int] = None,
    vkey: str = "X_vect",
    time_var: str = "Metadata_Time",
    tree_var: str = "Metadata_Track_Root",
    verbose: bool = False,
) -> ad.AnnData:
    """Vectorize an embedding of an AnnData object.

    If an AnnData object contains time-series data and information about time
    and cell tracks are known, this function can vectorize any representation of the object.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological time-series data
    use_rep : str, optional
        Representation to vectorize. `.X` is used by default
    n_pcs : int, optional
        Number of principal components to use if `use_rep` is `X_pca`
    vkey : str
        New keyword for the vectorized representation in `.obsm`
    time_var : str
        Variable in `.obs` with time information
    tree_var : str
        Variable in `.obs` if lineage tree information
    verbose : bool


    Returns
    -------
    anndata.AnnData
        AnnData object with additional vectorized embedding in `.obsm`

    Raises
    ------
    AssertionError
        If `time_var` or `tree_var` is not in `.obs`

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np
    >>> import pandas as pd

    >>> data = np.random.rand(6, 5)
    >>> obs = pd.DataFrame({
    >>>     'time': [0, 1, 2, 1, 2, 3],
    >>>     'lineage': [0, 0, 0, 1, 1, 1]
    >>> })
    >>> adata = ad.AnnData(data, obs=obs)

    >>> adata = mp.ts.vectorize_emb(adata, tree_var='lineage', time_var='time')  # vectorize .X
    >>> adata
    AnnData object with n_obs × n_vars = 6 × 5
        obs: 'time', 'lineage'
        obsm: 'X_vect'
    """
    # check vars
    assert time_var in adata.obs.columns, f"time_var not in .obs: {time_var}"
    assert tree_var in adata.obs.columns, f"tree_var not in .obs: {tree_var}"

    len_before = len(adata)
    adata = adata[np.isfinite(adata.obs[tree_var])]
    if verbose:
        logger.info(
            f"{len_before - len(adata)} samples deleted, "
            f"because they were not connected in a tree"
        )

    # conver to 3D
    converter = Adata3D(adata, time_var, tree_var, use_rep=use_rep, n_pcs=n_pcs)

    X = converter.to_3d()  # [N, T, F]

    # vectorize
    X[:, :-1, :] = X[:, 1:, :] - X[:, :-1, :]
    X[:, -1, :] = 0

    # back to 2d
    X = converter.to_2d(X)
    adata.obsm[vkey] = X
    return adata
