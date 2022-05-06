import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances

from typing import Optional


class FateCluster:
    """Find clusters in a fate map by one directional random walks.

    Given one start cell and several terminal cells, this class calculates
    random walks from the start cell until a terminal cell is reached.
    Walks are only possible in one direction (away from the start cell).
    A fate probability is assigned to each cell for all terminal cells based on the
    fate of random walks that passed a cell.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors for the affinity matrix
    neighbor_params : dict, optional
        Keyword arguments that are passed to sklearn.neighbors.NearestNeighbors
    n_iter : int
        Number of random walks
    centroid : numpy.ndarray, optional
        The centroid of the fate map
    centroid_method : str
        In case no `centroid` is given, the `mean` or
        `median` of all data points can be used as centroid
    centroid_fun_params : dict, optional
        Keyword arguments that are passed to the `centroid_method`
        in case no `centroid` is give
    random_state : int, optional
        Random initialization
    verbose : bool

    Raises
    ------
    ValueError
        If `centroid` is None and `centroid_method` is unknown
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        neighbor_params: Optional[dict] = None,
        n_iter: int = 100,
        centroid: np.array = Optional[None],
        centroid_method: str = "median",
        centroid_fun_params: dict = Optional[None],
        random_state: int = Optional[None],
        verbose=False,
    ):
        self.n_neighbors = n_neighbors
        if neighbor_params is None:
            neighbor_params = {}
        self.neighbor_params = neighbor_params
        self.n_iter = n_iter
        self.verbose = verbose

        self.centroid = centroid

        avail_centroid_methods = ["mean", "median"]
        centroid_method = centroid_method.lower()
        if centroid_method == "mean":
            self.centroid_fun = np.nanmean
            self.centroid_fun_params = {"axis": 0}
        elif centroid_method == "median":
            self.centroid_fun = np.nanmedian
            self.centroid_fun_params = {"axis": 0}
        elif callable(centroid_method) or centroid_method is None:
            self.centroid_fun = centroid_method
            if centroid_fun_params is None:
                centroid_fun_params = {}
            self.centroid_fun_params = centroid_fun_params
        else:
            raise ValueError(
                f"centroid_method must be one of {avail_centroid_methods} or callable, "
                f"instead got {centroid_method}"
            )

        self.n_fates = None
        self.terminal_cells = None
        self.fate_prob = None
        self.classes = None

        np.random.seed(random_state)

    @property
    def result(self) -> pd.DataFrame:
        """Get class results.

        Returns
        -------
        pandas.DataFrame
            DataFrame with fate counts per terminal cell
        """
        unique, counts = np.unique(self.classes, return_counts=True)
        return pd.DataFrame(
            {
                "tc_index": self.terminal_cells,
                "label": unique,
                "counts": counts,
            }
        )

    def fit(self, X: np.ndarray):
        """Compute random walks from centroid to terminal cells.

        This method calculates the fate probability for each cell
        to end up in a terminal state and the class based on the maximum probability.

        Parameters
        ----------
        X : numpy.ndarray
            Array of shape `[observations, features]`
        """

        # find centroid
        if self.centroid is not None:
            centroid = self.centroid
        else:
            centroid = self.centroid_fun(X, **self.centroid_fun_params)
            self.centroid = centroid
        if len(centroid.shape) == 1:
            centroid = centroid.reshape(1, -1)

        terminal_cells = []
        fate_count = None

        neigh = NearestNeighbors(n_neighbors=self.n_neighbors, **self.neighbor_params)
        neigh.fit(X)

        if self.verbose:
            iterator = tqdm(range(self.n_iter))
        else:
            iterator = range(self.n_iter)

        for _ in iterator:

            poi = centroid  # --> point of interest
            poi_dist = 0  # --> distance of poi from centroid
            poi_ix = None  # --> initiate
            footprint = np.zeros(X.shape[0]).reshape(-1, 1)

            # start one directional random walk
            while True:
                nbs = neigh.kneighbors(poi, self.n_neighbors, return_distance=False)

                nbs = nbs.flatten()
                cent_dist = euclidean_distances(X[nbs, :], centroid).flatten()
                # only consider points that are further away from centroid
                next_points_ix = np.argwhere(cent_dist > poi_dist)
                n_next_points = len(next_points_ix)
                if n_next_points > 0:
                    # choose a random point
                    next_point_ix = int(
                        next_points_ix[int(np.random.choice(n_next_points, 1))]
                    )
                    poi_ix = int(nbs[next_point_ix])
                    # store footprint and update poi
                    footprint[poi_ix, :] += 1
                    poi = X[poi_ix, :].reshape(1, -1)
                    poi_dist = float(cent_dist[next_point_ix])

                else:
                    # if no point is further away, a terminal cell is reached
                    assert (
                        poi_ix is not None
                    ), "something wrong with X, probably not enough samples"
                    if poi_ix not in terminal_cells:
                        terminal_cells.append(poi_ix)
                        if fate_count is None:
                            fate_count = footprint
                        else:
                            fate_count = np.concatenate((fate_count, footprint), axis=1)

                    else:
                        fate_ix = int(np.argwhere(np.asarray(terminal_cells) == poi_ix))
                        fate_count[:, fate_ix] += footprint.flatten()

                    break

        assert (
            fate_count is not None
        ), "can not compute fate probability, n_iter should be at least 1"
        fc_row_sum = fate_count.sum(axis=1)
        fate_prob = fate_count / fc_row_sum[:, np.newaxis]

        self.n_fates = len(terminal_cells)
        self.terminal_cells = terminal_cells
        self.fate_prob = fate_prob
        self.classes = np.argmax(fate_prob, axis=1)

        return None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenient funtion to fit an array an return the classes (based on fate probabilities).

        Parameters
        ----------
        X : numpy.ndarray
            Array of shape `[observations, features]`
        """
        self.fit(X)
        return self.classes
