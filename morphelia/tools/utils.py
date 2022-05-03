import logging
import scanpy as sc
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def choose_representation(adata, rep=None, n_pcs=None):
    """Get representation of multivariate data.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        rep (str): Key in .obsm.
        n_pcs (int): Number of principal components to return.

    Returns:
        numpy.ndarray
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


def get_subsample(adata, sample_size=None, seed=0):
    """Draws n (sample_size) random samples from adata.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        sample_size (int): Number of samples.
        seed (int): Seed for reproducibility of subsample.

    Returns:
        anndata.AnnData
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


def make_3d(
    adata,
    time_var="Metadata_Time",
    tree_var="Metadata_Trace_Tree",
    use_rep=None,
    n_pcs=50,
):
    """Return three dimensional representation of data.

    Args:
        time_var (str): Variable in .obs with timesteps
        tree_var (str): Variable in .obs with tree identifier
        use_rep (bool): Make representation of data 3d
        n_pcs (int): Number of PCs to use if use_rep is "X_pca"

    Returns:
        np.array of shape [samples x timesteps x features]
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
    def __init__(
        self,
        adata,
        time_var="Metadata_Time",
        tree_var="Metadata_Trace_Tree",
        use_rep=None,
        y_var=None,
        n_pcs=50,
    ):
        """Return three dimensional representation of data.

        Args:
            time_var (str): Variable in .obs with timesteps
            tree_var (str): Variable in .obs with tree identifier
            use_rep (bool): Make representation of data 3d
            n_pcs (int): Number of PCs to use if use_rep is "X_pca"

        Returns:
            np.array of shape [samples x timesteps x features]
        """
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

    def to_3d(self, return_y=False):
        """
        Convert adata.X of shape [samples x features] to shape [samples x timesteps x features].
        Cave: Each sample and timestep should only have one datapoint.

        Args:
            return_y (bool): Return y labels

        Returns:
            (np.array): Three dimensional representation of adata.X --> [Samples x Time x Features]
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

    def to_2d(self, X_3d):
        """
        Convert three dimensional representation of adata with shape [samples x timesteps x features]
        back to shape [samples x features] and add representation to adata.obsm.

        Cave: Each sample and timestep should only have one datapoint.

        Args:
            X_3d (np.array): Array of shape [samples x timesteps x features]

        Returns:
            (np.array)
        """
        assert (
            len(X_3d.shape) == 3
        ), f"Shape of X_3d must be three-dimensional, instead got shape {X_3d.shape}"

        X_2d = X_3d[self.trees_ix, self.time_ix, :].copy()

        return X_2d


def encode_labels(adata, key="Metadata_Treatment", sfx="_Enc"):
    """Encode categorical label in adata.obs.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        key (str): Name of observation in .obs.
        sfx (str): Suffix for new observation.

    Returns:
        (anndata.AnnData)
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


def vectorize_emb(
    adata,
    use_rep=None,
    n_pcs=None,
    vkey="X_vect",
    time_var="Metadata_Time",
    tree_var="Metadata_Track_Root",
    verbose=False,
):
    """
    Vectorize embedding.

    :param adata:
    :param use_rep:
    :param n_pcs:
    :param vkey:
    :param time_var:
    :param tree_var:
    :param verbose:
    :return:
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
