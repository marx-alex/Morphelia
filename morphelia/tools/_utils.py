import warnings
import scanpy as sc


def _choose_representation(adata,
                           rep=None,
                           n_pcs=None):
    """Get representation of multivariate data.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        rep (str): Key in .obsm.
        n_pcs (int): Number of principal components to return.

    Returns:
        numpy.ndarray
    """
    # return .X if rep is None
    if rep is None and n_pcs == 0:
        X = adata.X

    # use X_pca by default
    if rep is None:
        if 'X_pca' not in adata.obsm.keys():
            warnings.warn("Found no PC representation. Trying to compute PCA...")
            sc.tl.pca(adata)

        if 'X_pca' in adata.obsm.keys():

            if n_pcs is not None and n_pcs > adata.obsm['X_pca'].shape[1]:
                warnings.warn(f"Number n_pcs {n_pcs} is larger than PCs in X_pca, "
                              f"use number of PCs in X_pca instead {adata.obsm['X_pca'].shape[1]}")
                n_pcs = adata.obsm['X_pca'].shape[1]

            # return pcs
            X = adata.obsm['X_pca'][:, :n_pcs]

        else:
            raise ValueError("Did not found X_pca in .obsm")

    else:
        if rep == 'X_pca':
            if n_pcs is not None:
                X = adata.obsm[rep][:, :n_pcs]
            else:
                X = adata.obsm[rep]

        elif rep in adata.obsm.keys():
            X = adata.obsm[rep]

        elif rep == 'X':
            X = adata.X

        else:
            raise ValueError(f"Did not find rep in .obsm: {rep}")

    return X
