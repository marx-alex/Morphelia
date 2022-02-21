import numpy as np

from morphelia.external import waypoint_sampling
from morphelia.tools import choose_representation
from sklearn.metrics import pairwise_distances


def get_quiver(adata, X="X_nne", V="X_vect", n_waypoints=50, min_cells=20):
    """
    Get grid parameters to draw a quiver plot.
    Use morphelia.tools.vectorize_emb beforehand.

    :param adata:
    :param X:
    :param V:
    :param n_waypoints:
    :param min_cells:
    :return:
    """
    assert X in adata.obsm.keys(), f"X not in .obsm: {X}"
    assert V in adata.obsm.keys(), f"V not in .obsm: {V}"

    X = choose_representation(adata, rep=X)
    assert len(X.shape) == 2, f"X must be 2-dimensional, instead got shape: {X.shape}"
    V = choose_representation(adata, rep=V)
    assert len(V.shape) == 2, f"V must be 2-dimensional, instead got shape: {V.shape}"

    # get waypoints
    wps = waypoint_sampling(X, n_waypoints=n_waypoints)
    waypoints = X[wps, :]

    # compute distance waypoints
    wp_dists = pairwise_distances(X, X[wps, :])
    # closest waypoint for every cell
    wp_ixs = np.argmin(wp_dists, axis=1)

    # find average vector for every waypoint
    wp_vect = np.zeros((len(wps), 2))
    for ix in range(len(wps)):
        if np.sum(wp_ixs == ix) > min_cells:
            wp_vect[ix, :] = np.nanmean(V[wp_ixs == ix, :], axis=0)

    vect_sum = np.sum(wp_vect, axis=1)
    zero_vect = vect_sum != 0

    return (
        waypoints[zero_vect, 0],
        waypoints[zero_vect, 1],
        wp_vect[zero_vect, 0],
        wp_vect[zero_vect, 1],
    )


def get_streamline(adata, X="X_nne", V="X_vect", grid_dim=50, min_cells=5):
    """
    Get grid parameters to draw a streamplot.
    Use morphelia.tools.vectorize_emb beforehand.

    :param adata:
    :param X:
    :param V:
    :param grid_dim:
    :param min_cells:
    :return:
    """
    assert X in adata.obsm.keys(), f"X not in .obsm: {X}"
    assert V in adata.obsm.keys(), f"V not in .obsm: {V}"

    X = choose_representation(adata, rep=X)
    assert len(X.shape) == 2, f"X must be 2-dimensional, instead got shape: {X.shape}"
    V = choose_representation(adata, rep=V)
    assert len(V.shape) == 2, f"V must be 2-dimensional, instead got shape: {V.shape}"

    # get boundary vals
    max_val = np.max(X, axis=0)
    max_x, max_y = max_val[0], max_val[1]
    min_val = np.min(X, axis=0)
    min_x, min_y = min_val[0], min_val[1]

    # get meshgrid of waypoints and their indices
    wp_x = np.linspace(min_x, max_x, grid_dim)
    wp_y = np.linspace(min_y, max_y, grid_dim)
    wp_ix = np.arange(grid_dim)

    wp_X, wp_Y = np.meshgrid(wp_x, wp_y)
    wps = np.column_stack((wp_X.flatten(), wp_Y.flatten()))
    wp_X_ix, wp_Y_ix = np.meshgrid(wp_ix, wp_ix)
    wp_ix = np.column_stack((wp_X_ix.flatten(), wp_Y_ix.flatten()))

    # compute distance waypoints
    wp_dists = pairwise_distances(X, wps)
    # closest waypoint for every cell
    wp_nrbs = np.argmin(wp_dists, axis=1)

    # find average vector for every waypoint
    Xv = np.zeros((grid_dim, grid_dim))
    Yv = np.zeros((grid_dim, grid_dim))
    for ix in range(len(wps)):
        if np.sum(wp_nrbs == ix) > min_cells:
            grid_x, grid_y = wp_ix[ix, 0], wp_ix[ix, 1]
            vect = np.nanmean(V[wp_nrbs == ix, :], axis=0)
            Xv[grid_y, grid_x] = vect[0]
            Yv[grid_y, grid_x] = vect[1]

    return wp_X, wp_Y, Xv, Yv
