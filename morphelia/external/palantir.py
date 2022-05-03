import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse.linalg import eigs
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances, euclidean_distances
from scipy.sparse import csgraph, find, csr_matrix
from scipy.stats import pearsonr, entropy, norm
import networkx as nx
from pygam import LinearGAM, s

import time
import warnings
import logging
from copy import deepcopy
from collections import OrderedDict

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


class Palantir:
    """
    This is an implementation of the Palantir framework, which is described here:
    Setty, M., Kiseliovas, V., Levine, J. et al.
    Characterization of cell fate probabilities in single-cell data with Palantir.
    Nat Biotechnol 37, 451–460 (2019). https://doi.org/10.1038/s41587-019-0068-4

    Some fragments of the code are taken from the original implementation:
    https://github.com/dpeerlab/Palantir

    The code is licensed under the GPL-2.0 License.
    """

    def __init__(
        self,
        n_waypoints=100,
        n_neighbors=10,
        start_cell=None,
        terminal_states=None,
        n_jobs=None,
        verbose=False,
    ):
        self.n_waypoints = n_waypoints
        self.n_neighbors = n_neighbors
        self.start_cell = start_cell
        self.n_jobs = n_jobs
        self.verbose = verbose

        if isinstance(terminal_states, (tuple, list)):
            terminal_states = np.array(terminal_states)
        elif isinstance(terminal_states, np.ndarray):
            pass
        else:
            raise TypeError(
                f"terminal_states must be of type list, tuple or numpy.ndarray, "
                f"instead got {type(terminal_states)}"
            )
        self._terminal_states = terminal_states

        # results
        self._waypoints = None
        self._pseudotime = None
        self._W = None
        self._entropy = None
        self._branch_probs = None

    @property
    def waypoints(self):
        return self._waypoints

    @property
    def terminal_states(self):
        return self._terminal_states

    @property
    def pseudotime(self):
        return self._pseudotime

    @property
    def branch_probs(self):
        return self._branch_probs

    @branch_probs.setter
    def branch_probs(self, branch_probs):
        self._branch_probs = branch_probs

    @property
    def entropy(self):
        return self._entropy

    @entropy.setter
    def entropy(self, entropy):
        self._entropy = entropy

    @property
    def branch_label(self):
        return np.argmax(self.branch_probs.to_numpy(), axis=1)

    def fit(self, X, emb=None):
        start = time.time()

        if isinstance(X, np.ndarray):
            X = X.copy()
        elif isinstance(X, ad.AnnData):
            assert (
                emb is not None
            ), "X is of type anndata.AnnData, please provide the embedding name in .obsm"
            X = X.obsm[emb].copy()
        elif isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        else:
            raise TypeError(
                "X must be of type numpy.ndarray, pandas.DataFrame or anndata.AnnData, "
                f"instead got {type(X)}"
            )

        if self.verbose:
            logger.info(f"Creating {self.n_waypoints} waypoints")
        self._waypoints = waypoint_sampling(X, n_waypoints=self.n_waypoints)

        if self.start_cell is None:
            self.start_cell = find_start_cell(X)
        assert isinstance(self.start_cell, int), (
            "start cell must be a y index for X of type(int), "
            f"instead got {type(self.start_cell)}"
        )

        if self.start_cell not in self.waypoints:
            self._waypoints.append(self.start_cell)
        if self._terminal_states is not None:
            self._waypoints = list(set(self._waypoints + list(self._terminal_states)))

        if self.verbose:
            logger.info("Pseudotime computation")
        self._pseudotime, self._W = compute_pseudotime(
            X,
            self.start_cell,
            self._waypoints,
            n_neighbors=self.n_neighbors,
            verbose=self.verbose,
        )

        if self.verbose:
            logger.info("Entropy and branch probabilities")
        _branch_probs = differentiation_prob(
            X,
            self._waypoints,
            self._pseudotime,
            terminal_states=self._terminal_states,
        )
        self._terminal_states = list(_branch_probs.columns)

        if self.verbose:
            logger.info("Project results to all cells")
        self._branch_probs = pd.DataFrame(
            np.dot(self._W.T, _branch_probs), columns=self._terminal_states
        )
        self._entropy = self._branch_probs.apply(entropy, axis=1)

        end = time.time()
        if self.verbose:
            logger.info(f"Time for computation: {end - start} seconds")

        return self

    def annotate_data(self, adata):
        """
        Annotate anndata.AnnData class with Palantir class instances.

        :param adata:
        :return:
        """
        assert isinstance(adata, ad.AnnData), (
            f"adata must be of type anndata.AnnData, " f"instead got {type(adata)}"
        )

        adata.obs["pseudotime"] = self.pseudotime
        adata.obs["branch"] = self.branch_label
        adata.obs["branch"] = adata.obs["branch"].astype("category")
        adata.obs["entropy"] = self.entropy.to_numpy()

        uns = {
            "waypoints": self.waypoints,
            "terminal_states": self.terminal_states,
            "start_cell": self.start_cell,
        }
        adata.uns["palantir"] = uns

        for branch in self.branch_probs:
            adata.obs[f"branch_prob_{branch}"] = self.branch_probs[branch].to_numpy()

        return adata

    def branch_dist(self, X, treat_var="Metadata_Treatment", cutoff=0.7):
        """
        Convenience function to get distribution of labels per branch.

        :param X:
        :param treat_var:
        :param cutoff:
        :return:
        """

        if isinstance(X, ad.AnnData):
            assert treat_var in X.obs.columns, f"treat_var not in .obs: {treat_var}"
            X = X.obs[treat_var].to_numpy()

        branch_labels = self.branch_label.copy()
        branch_probs = self.branch_probs.values

        branches = list(range(len(self.branch_probs.columns)))
        treats = np.unique(X)

        count_matrix = np.zeros((len(branches), len(treats)))
        count_matrix = pd.DataFrame(count_matrix, columns=treats, index=branches)

        for branch in branches:
            if cutoff is None:
                mask = branch_labels == branch
            else:
                mask = branch_probs[:, branch] > cutoff

            branch_treats = X[mask]
            unique, counts = np.unique(branch_treats, return_counts=True)

            for i, treat in enumerate(unique):
                count_matrix.loc[branch, treat] += counts[i]

        dist_matrix = count_matrix.div(count_matrix.sum(axis=0), axis=1).fillna(0)
        # dist_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0).fillna(0)
        dist_matrix.name = "dist"
        count_matrix.name = "count"

        return count_matrix, dist_matrix

    def compute_trends(self, X, feats, branches=None):
        """
        For every given feature and branch fit a Generalized Additive Model to show feature trends.

        :param X:
        :param feats:
        :param branches:
        :return:
        """
        # check given variables
        if isinstance(feats, str):
            feats = [feats]
        else:
            assert isinstance(
                feats, (tuple, list)
            ), f"feats should be of type str, list or tuple, instead got {type(feats)}"

        if isinstance(X, ad.AnnData):
            assert all(
                feat in X.var_names for feat in feats
            ), f"Not all features given by feats found in .var_names: {feats}"
            X = X[:, feats].X.copy()

        # compute trends for all branches
        if branches is None:
            branches = list(range(len(self.branch_probs.columns)))

        # store results in dict
        results = OrderedDict()
        for branch in branches:
            results[branch] = OrderedDict()
            # branch weights
            weights = self.branch_probs.iloc[:, branch]
            # pseudotime range for branch
            br_max_pt = self.pseudotime[weights > 0.7].max()
            branch_pt = np.linspace(0, br_max_pt, 500)
            results[branch]["pseudotime"] = branch_pt

            # fit GAM for every feature
            for ix, feat in enumerate(feats):
                results[branch][feat] = OrderedDict()
                y = X[:, ix]
                # fit GAM
                gam = LinearGAM(s(0))
                gam.fit(self.pseudotime, y, weights=weights)

                # get predictions and confidence intervals
                y_predict = gam.predict(branch_pt)
                y_intervals = gam.confidence_intervals(branch_pt)

                # store values
                results[branch][feat]["trends"] = y_predict
                results[branch][feat]["ci"] = y_intervals

        return results


def waypoint_sampling(X, n_waypoints=100):
    """
    Min-max sampling of waypoints in a two dimensional embedding.

    :param X:
    :param n_waypoints:
    :return:
    """
    # store waypoints and initiate distances
    wps = []
    N = X.shape[0]
    dists = np.zeros((N, n_waypoints))

    # random sampling of first waypoint
    first_wp = np.random.randint(N)
    wps.append(first_wp)

    # distance of all cells to first waypoint
    dists[:, 0] = euclidean_distances(X, X[wps[0], :].reshape(1, -1)).flatten()

    for ix in range(1, n_waypoints):
        # get minimum distances to current set of waypoints
        min_dists = dists[:, :ix].min(axis=1)

        # choose maximum distance as new waypoint
        new_wp = np.where(min_dists == min_dists.max())[0][0]
        wps.append(new_wp)

        # add distances to new waypoint
        dists[:, ix] = euclidean_distances(X, X[wps[ix], :].reshape(1, -1)).flatten()

    # filter unique waypoints
    wps = list(set(wps))

    return wps


def find_start_cell(X):
    """
    Return cell closest to origin

    :param X:
    :return:
    """
    dist_from_orig = np.sqrt(np.sum(X ** 2, axis=1))
    return int(np.argmin(dist_from_orig))


def compute_pseudotime(
    X, start_cell, wps, n_neighbors=10, n_jobs=None, max_iter=25, verbose=False
):
    """
    Pseudotime computation for cells embedded in X.
    Pseudotime increases with distance from start_cell.

    :param X:
    :param start_cell:
    :param wps:
    :param n_neighbors:
    :param n_jobs:
    :param max_iter:
    :param verbose:
    :return:
    """
    # k nearest neighbors to connect cells
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, metric="euclidean", n_jobs=n_jobs
    ).fit(X)

    adj = nbrs.kneighbors_graph(X, mode="distance")

    # connect graph where disconnected
    adj = connect_graph(adj, X, start_cell)

    # compute waypoint distances to all other cells
    D = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(shortest_path_helper)(adj, cell) for cell in wps
    )
    D = np.stack(D)

    # determining the perspective matrix
    # waypoint weights
    sdv = np.std(D) * 1.06 * D.size ** (-1 / 5)
    W = np.exp(-0.5 * np.power((D / sdv), 2))
    # stochastize the matrix
    W = W / W.sum(axis=0)

    # initialize pseudotime to start cell distances
    pseudotime = D[wps.index(start_cell), :]
    converged = False

    # iteratively update perspective and determine pseudotime
    iter = 1
    while not converged and iter < max_iter:
        # align perspective matrix to distance matrix
        P = deepcopy(D)
        for wp in wps[1:]:
            # wp position relative to start
            wp_dist = pseudotime[wp]

            # converting all distance smaller than wp_dist to negative
            before_mask = pseudotime < wp_dist
            P[wps.index(wp), before_mask] = -D[wps.index(wp), before_mask]

            # align to start
            P[wps.index(wp), :] = P[wps.index(wp), :] + wp_dist

        # weighted pseudotime
        new_traj = np.sum(P * W, axis=0)

        # check for convergence
        corr = pearsonr(pseudotime, new_traj)[0]
        if verbose:
            logger.info(f"Correlation at iteration {iter:d}: {corr:.4f}")
        if corr > 0.9999:
            converged = True

        pseudotime = new_traj
        iter += 1

    # scale pseudotime
    pseudotime = (pseudotime - np.min(pseudotime)) / np.ptp(pseudotime)

    return pseudotime, W


def connect_graph(adj, X, start_cell):
    """
    Reconnect a partially disconnected graph by finding disconnected nodes and
    attaching them to furthest disconnected nodes.

    :param adj:
    :param X:
    :param start_cell:
    :return:
    """

    # adjacency matrix to graph
    G = nx.Graph(adj)
    # find unreachable nodes
    dists = pd.Series(nx.single_source_dijkstra_path_length(G, start_cell))
    unreachable_nodes = list(set(range(X.shape[0])) - set(dists.index))

    if len(unreachable_nodes) > 0:
        warnings.warn(
            f"Found {len(unreachable_nodes)} disconnected nodes, consider increasing n_neighbors"
        )

        # reconnect isolated cells
        while len(unreachable_nodes) > 0:
            farthest_reachable = dists.loc[dists == dists.max()]
            farthest_reachable = farthest_reachable.index

            # compute distance to unreachable_nodes
            unreachable_dists = pairwise_distances(
                X[unreachable_nodes, :],
                X[farthest_reachable, :].reshape(1, -1),
            )

            unreachable_dists = pd.DataFrame(
                unreachable_dists,
                index=unreachable_nodes,
                columns=farthest_reachable,
            )

            # add edge between unreachable and its closest farthest reachable
            closest_nodes = unreachable_dists.idxmin(axis=1).to_list()
            closest_node_dist = unreachable_dists.min(axis=1)
            adj[closest_nodes, unreachable_nodes] = closest_node_dist

            # recompute isolates based on new adjacency matrix
            G = nx.Graph(adj)
            dists = pd.Series(nx.single_source_dijkstra_path_length(G, start_cell))
            unreachable_nodes = set(range(X.shape[0])).difference(dists.index)

    return adj


def shortest_path_helper(adj, cell):
    return csgraph.dijkstra(adj, False, cell)


def differentiation_prob(
    X, wps, pseudotime, terminal_states=None, n_neighbors=10, n_jobs=None
):
    """
    Differentiation probability for cells of reaching a certain terminal cell state.
    Probability is defined to be the stationary distribution in a markov process.

    :param X:
    :param wps:
    :param pseudotime:
    :param terminal_states:
    :param n_neighbors:
    :param n_jobs:
    :return:
    """
    T = construct_markov_chain(X, wps, pseudotime, n_neighbors, n_jobs)

    # find terminal states
    if terminal_states is None:
        terminal_states = terminal_states_from_markov_chain(X, wps, T, pseudotime)

    wps = np.array(wps)
    abs_states = np.where(np.isin(wps, terminal_states))[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # terminal states are absorption states and should have no outgoing edges
        T[abs_states, :] = 0
        # diagonals as 1s
        T[abs_states, abs_states] = 1

    # getting list of transition states
    trans_states = list(set(range(len(wps))).difference(abs_states))

    # Q matrix
    Q = T[trans_states, :][:, trans_states]
    # fundamental matrix
    mat = np.eye(Q.shape[0]) - Q.todense()
    N = np.linalg.inv(mat)

    # absorption probabilities
    branch_probs = np.dot(N, T[trans_states, :][:, abs_states].todense())
    branch_probs = pd.DataFrame(
        branch_probs, index=wps[trans_states], columns=wps[abs_states]
    )
    branch_probs[branch_probs < 0] = 0

    bp = pd.DataFrame(0, index=terminal_states, columns=terminal_states)
    bp.values[range(len(terminal_states)), range(len(terminal_states))] = 1
    branch_probs = branch_probs.append(bp.loc[:, branch_probs.columns])

    return branch_probs.loc[wps, :]


def terminal_states_from_markov_chain(X, wps, T, pseudotime):
    """
    Choosing terminal states from the boundaries of the embedding.

    :param X:
    :param wps:
    :param T:
    :param pseudotime:
    :return:
    """
    # boundaries of embedding
    ix_max = set(np.argmax(X[wps, :], axis=0))
    ix_min = set(np.argmin(X[wps, :], axis=0))
    boundaries = list(ix_max.union(ix_min))
    boundaries = [wps[i] for i in boundaries]

    vals, vecs = eigs(T.T, 10)
    ranks = np.abs(np.real(vecs[:, np.argsort(vals)[-1]]))
    ranks_median = np.median(ranks)

    # cutoff and intersection with the boundary cells
    cutoff = norm.ppf(
        0.9999,
        loc=ranks_median,
        scale=np.median(np.abs((ranks - ranks_median))),
    )

    # connected components of cells beyond cutoff
    cells = np.array(wps)[ranks > cutoff]

    # find connected components
    T_dense = pd.DataFrame(T.todense(), index=wps, columns=wps)
    G = nx.from_pandas_adjacency(T_dense.loc[cells, cells])
    cells = [
        list(i)[np.argmax(pseudotime[list(i)])] for i in nx.connected_components(G)
    ]

    # nearest embedded space boundaries
    terminal_states = [
        np.argmin(pairwise_distances(X[boundaries, :], X[i, :].reshape(1, -1)))
        for i in cells
    ]

    terminal_states = [boundaries[i] for i in terminal_states]
    terminal_states = np.unique(terminal_states)

    return terminal_states


def construct_markov_chain(X, wps, pseudotime, n_neighbors=10, n_jobs=None):
    """
    Construction of a markov chain with cell states and their transition probabilities.

    :param X:
    :param wps:
    :param pseudotime:
    :param n_neighbors:
    :param n_jobs:
    :return:
    """
    # build knn graph
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, metric="euclidean", n_jobs=n_jobs
    ).fit(X[wps, :])

    adj = nbrs.kneighbors_graph(X[wps, :], mode="distance")
    dist, ixs = nbrs.kneighbors(X[wps, :])

    # standard deviation allowing for "back" edges
    adpative_k = np.min([int(np.floor(n_neighbors / 3)) - 1, 30])
    adaptive_std = np.ravel(dist[:, adpative_k])

    # directed graph construction
    # pseudotime position of all the neighbors
    pt_nbrs = np.take(pseudotime, np.take(wps, ixs))

    # remove edges that point backwards in pseudotime except for edges that are within
    # the computed standard deviation
    pt_margin = pseudotime[wps] - adaptive_std
    rem_edges = pt_nbrs < pt_margin[:, np.newaxis]
    x, y = rem_edges.nonzero()

    # determine the indices and update adjacency matrix
    adj[x, ixs[x, y]] = 0

    # affinity matrix and markov chain
    i, j, v = find(adj)

    aff = np.exp(
        -(v ** 2) / (adaptive_std[i] ** 2) * 0.5
        - (v ** 2) / (adaptive_std[j] ** 2) * 0.5
    )
    W = csr_matrix((aff, (i, j)), [len(wps), len(wps)])

    # transition matrix
    D = np.ravel(W.sum(axis=1))
    i, j, v = find(W)
    T = csr_matrix((v / D[i], (i, j)), [len(wps), len(wps)])

    return T
