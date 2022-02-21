import scipy as sc
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class StateTransitionModel:
    """
    Model for state transitions.
    """

    def __init__(
        self, adata, state_var="leiden", time_var="Metadata_Time", rep="X_umap"
    ):
        self.adata = adata
        assert (
            state_var in self.adata.obs.columns
        ), f"state variable not in .obs: {state_var}"
        assert (
            time_var in self.adata.obs.columns
        ), f"time variable not in .obs: {time_var}"
        self.state_var = state_var
        self.time_var = time_var
        assert rep in self.adata.obsm, f"Representation not found: {rep}"
        self.rep = rep

        # state statistics
        self.state_means = self._get_state_means()
        self.state_dists = self._get_state_dists()

    def _get_state_means(self):
        return np.stack(
            self.adata.obs.groupby(self.state_var).apply(
                lambda x: np.mean(self.adata[x.index, :].obsm[self.rep], axis=0)
            )
        )

    def _get_state_dists(self):
        return euclidean_distances(self.state_means, self.state_means)

    def transportation_problem(self, supply, demand):
        n = len(demand)
        m = len(supply)

        b_ub = np.append(supply, -demand)

        nz = 2 * n * m
        irow = np.zeros(nz, dtype=int)
        jcol = np.zeros(nz, dtype=int)
        value = np.zeros(nz)
        for i in range(n):
            for j in range(m):
                k = n * i + j
                k1 = 2 * k
                k2 = k1 + 1
                irow[k1] = i
                jcol[k1] = k
                value[k1] = 1.0
                irow[k2] = n + j
                jcol[k2] = k
                value[k2] = -1.0

        A_ub = sc.sparse.coo_matrix((value, (irow, jcol)))

        res = sc.optimize.linprog(
            c=np.reshape(self.state_dists, m * n),
            A_ub=A_ub,
            b_ub=b_ub,
            options={"sparse": True},
        )

        return res.x.reshape(m, n)

    def state_dist(self):
        """
        Proportion of cell in states for all time points.
        Sum of proportions for one time point should equal 1.
        """
        state_dist = self.adata.obs.groupby(self.time_var)[self.state_var].apply(
            lambda x: x.value_counts() / x.count()
        )
        state_dist = state_dist.unstack(level=1)
        return state_dist
