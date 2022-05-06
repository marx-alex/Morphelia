import scipy as sc
import anndata as ad
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class StateTransitionModel:
    """Model state transitions over time.

    Individual cell may change their state over time.
    This class provides a framework to model the state distribution transition.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological time-series data
    state_var : str
        Variable in `.obs` that contains states
    time_var : str
        Variable in `.obs` that contains time information
    rep : str
        Representation of the AnnData object to use.
        Default is `X_umap`.

    Raises
    -------
    AssertionError
        If `state_var` of `time_var` is not in `.obs`
    AssertionError
        If representation is not found

    Attributes
    ----------
    adata : anndata.AnnData
        AnnData object
    state_var : str
        State variable in adata
    time_var : str
        Time variable in adata
    rep : str
        Representation in adata
    state_means : numpy.ndarray
        Means per state
    state_dists : numpy.ndarray
        Distances between state means

    Examples
    --------
    >>> import anndata as ad
    >>> import morphelia as mp
    >>> import numpy as np
    >>> import pandas as pd

    >>> data = np.random.rand(9, 5)
    >>> rep = np.random.rand(9, 5)
    >>> obs = pd.DataFrame({
    >>>     'time': [0, 1, 2, 0, 1, 2, 0, 1, 2],
    >>>     'state': [0, 1, 0, 1, 0, 0, 1, 1, 1]
    >>> })
    >>> adata = ad.AnnData(data, obs=obs)
    >>> adata.obsm['X_rep'] = rep

    >>> trans = mp.ts.StateTransitionModel(adata, state_var='state', time_var='time', rep='X_rep')
    >>> trans.state_dist()  # state distribution
        0	        1
    time
    0	0.333333	0.666667
    1	0.333333	0.666667
    2	0.666667	0.333333
    """

    def __init__(
        self,
        adata: ad.AnnData,
        state_var: str = "leiden",
        time_var: str = "Metadata_Time",
        rep: str = "X_umap",
    ) -> None:
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

    def transportation_problem(
        self, supply: np.ndarray, demand: np.ndarray
    ) -> np.ndarray:
        """Solves the transportation problem between supply and demand.

        Parameters
        ----------
        supply : np.ndarray
            Supply
        demand : np.ndarray
            Demand

        Returns
        -------
        np.ndarray
            Optimal allocation of supply to demand
        """
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
        """The proportion of cell in states for all time points.

        The sum of proportions for one time point should equal 1.

        Returns
        -------
        pandas.DataFrame
            Distribution of states for every time point
        """
        state_dist = self.adata.obs.groupby(self.time_var)[self.state_var].apply(
            lambda x: x.value_counts() / x.count()
        )
        state_dist = state_dist.unstack(level=1)
        return state_dist
