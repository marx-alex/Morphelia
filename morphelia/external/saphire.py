import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.ticker import MaxNLocator
from hmmlearn import hmm

from typing import Tuple, Optional

plt.style.use("seaborn-darkgrid")


class BaseTraj:
    """Base trajectory class.

    Parameters
    ----------
    model : hmmlearn.hmm.GaussianHMM
        Hidden markov model
    X : numpy.ndarray
        Single-cell embedding

    Raises
    ------
    AssertionError
        If single-cell embedding does not have two dimensions

    Attributes
    ----------
    model : hmmlearn.hmm.GaussianHMM
        Hidden markov model
    X : numpy.ndarray
        Single-cell embedding
    means : numpy.ndarray
        Model means
    states : numpy.ndarray
        Predicted model states
    n_states : int
        Number of model states
    trans : numpy.ndarray
        Model transition matrix
    """

    def __init__(self, model: hmm.GaussianHMM, X: np.ndarray) -> None:
        self.model = model
        assert len(X.shape) == 2, f"X should be 2-d, instead got shape {X.shape}"
        self.X = X
        self.means = self.model.means_.copy()
        self.states = self.model.predict(X)
        self.n_states = len(np.unique(self.states))
        self.trans = self.model.transmat_.copy()

    def rho_dt_bins(
        self, rho: np.ndarray, theta: np.ndarray, dt: np.ndarray, bins: int = 12
    ) -> Tuple[np.ndarray, list, list]:
        """
        Bin rho values and dwell time on polar coordinates.

        Parameters
        ----------
        rho : numpy.ndarray
            Polar space rho
        theta : numpy.ndarray
        Polar space theta
        dt : numpy.ndarray
            Transition dwell time
        bins : int
            Number of bins for the radar plot

        Returns
        -------
        numpy.ndarray, list, list
            Bin means, rhos and transition dwell times
        """
        bins = np.linspace(-np.pi, np.pi, bins + 1)
        bin_means = (bins[:-1] + bins[1:]) / 2
        bin_ix = np.digitize(theta, bins)
        bin_rd = [
            rho[(bin_ix == i) & (rho > 0)].mean()
            if len(rho[(bin_ix == i) & (rho > 0)]) > 0
            else 0
            for i in range(1, len(bins))
        ]
        bin_dt = [
            dt[(bin_ix == i) & (dt > 0)].sum()
            if len(dt[(bin_ix == i) & (dt > 0)]) > 0
            else 0
            for i in range(1, len(bins))
        ]
        return bin_means, bin_rd, bin_dt

    def transition_vectors(self):
        """Transition vectors between states on polar coordinates.

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Rhos and thetas in polar space
        """
        mu_x, mu_y = self.means[:, 0], self.means[:, 1]
        mu_x_dist = mu_x - mu_x[:, np.newaxis]
        mu_y_dist = mu_y - mu_y[:, np.newaxis]

        dist_vect = np.column_stack((mu_x_dist.flatten(), mu_y_dist.flatten()))
        trans_rho, trans_theta = self.cart2pol(dist_vect)
        trans_rho = (
            trans_rho.reshape((self.n_states, self.n_states)) * self.design_transition()
        ).flatten()
        return trans_rho, trans_theta

    def design_transition(self, thresh: float = 0.1):
        """Design transition matrix with zero diagonals.

        Parameters
        ----------
        thresh : float
            Transitions below thresh are 0, over thresh are 1

        Returns
        -------
        numpy.ndarray
            Design transition matrix
        """
        design_trans = self.trans
        diag_ix = np.diag_indices(len(design_trans))
        design_trans[diag_ix] = 0
        design_trans[design_trans < thresh] = 0
        design_trans[design_trans >= thresh] = 1
        return design_trans

    def norm_trans_time(self):
        """Normalized transition dwell time.

        Returns
        -------
        numpy.ndarray
            Normalized transition dwell time
        """
        unique, counts = np.unique(self.states, return_counts=True)
        sort_ix = unique.argsort()
        counts = counts[sort_ix]
        # normalize by transition probability
        dt = (counts * self.design_transition()).flatten()

        return dt / dt.sum()

    def norm_state_time(self):
        """Normalized state dwell time.

        Returns
        -------
        numpy.ndarray
            Normalized state dwell time
        """
        unique, counts = np.unique(self.states, return_counts=True)
        sort_ix = unique.argsort()
        counts = counts[sort_ix]
        return counts / counts.sum()

    @staticmethod
    def cart2pol(arr):
        """Cartesion space to polar space.

        Parameters
        ----------
        numpy.ndarray, numpy.ndarray
            Arrays of shape `[n_state, dims]`
        """
        x, y = arr[:, 0], arr[:, 1]
        rho = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        return rho, theta


class PhenoSign(BaseTraj):
    """Phenotypic Signature class.

    Parameters
    ----------
    model : hmmlearn.hmm.GaussianHMM
        Hidden markov model
    X : numpy.ndarray
        Single-cell embedding

    Attributes
    ----------
    bin_means : numpy.ndarray
        Bin means for radar plot
    signature : numpy.ndarray
        Phenotypic signature of shape `[4, n_bins`] with
        state radial distances, state dwell times,
        transition distances, transition dwell times
    """

    def __init__(self, model, X) -> None:
        super(PhenoSign, self).__init__(model, X)
        self.bin_means, self.signature = self.get_signature()

    def get_signature(self):
        """Calculate phenotypic signature for a given model.

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Bin means, array of shape `[4, n_bins]` with
            state radial distances, state dwell times,
            transition distances, transition dwell times

        Raises
        ------
        AssertionError
            If state bin means do not match transition bin means
        """
        # states
        mu_rho, mu_theta = self.cart2pol(self.means)
        state_dt = self.norm_state_time()
        bin_means_1, state_rd_bins, state_dt_bins = self.rho_dt_bins(
            mu_rho, mu_theta, state_dt
        )

        # transitions
        trans_rho, trans_theta = self.transition_vectors()
        trans_dt = self.norm_trans_time()
        bin_means_2, trans_rd_bins, trans_dt_bins = self.rho_dt_bins(
            trans_rho, trans_theta, trans_dt
        )

        assert (bin_means_1 == bin_means_2).all(), (
            "state and transition vectors are binned differently and can"
            "not be concatenated."
        )

        return bin_means_1, np.vstack(
            (state_rd_bins, state_dt_bins, trans_rd_bins, trans_dt_bins)
        )


class Saphire(PhenoSign):
    """Implementation of the SAPHIRE algorithm.

    This class can be used to visualize Hidden Markov Models and time series data.

    Parameters
    ----------
    model : hmmlearn.hmm.GaussianHMM
        Hidden markov model
    X : numpy.ndarray
        Single-cell embedding

    References
    ----------
    .. [1] Gordonov S, Hwang MK, Wells A, Gertler FB, Lauffenburger DA,
       Bathe M. Time series modeling of live-cell shape dynamics for
       image-based phenotypic profiling. Integr Biol (Camb). 2016;8(1):73-90.
    """

    def __init__(self, model, X):
        super(Saphire, self).__init__(model, X)

    def plot_traj(
        self, projection: str = "cartesian", ymax: Optional[float] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot cell trajectories.

        Parameters
        ----------
        projection : str
            `cartesian` or `polar`
        ymax : int, optional
            Limit for y-axis

        Returns
        -------
        matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
            Figure and Axes

        Raises
        ------
        AssertionError
            If projection is unknown
        """
        avail_proj = ["cartesian", "polar"]
        projection = projection.lower()
        assert projection in avail_proj, f"projection unknown: {projection}"
        if projection == "cartesian":
            projection = None

        cmap = plt.get_cmap("binary")
        cmap = _truncate_colormap(cmap, minval=0.2)

        if projection == "polar":
            y, x = self.cart2pol(self.X)
            y_mu, x_mu = self.cart2pol(self.means)
        else:
            x, y = self.X[:, 0], self.X[:, 1]
            x_mu, y_mu = self.means[:, 0], self.means[:, 1]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": projection})
        ax.scatter(x, y, c=self.states, cmap="Set1", zorder=2)
        traj = ax.scatter(
            x_mu,
            y_mu,
            c=np.unique(self.states),
            cmap="Set1",
            s=200,
            zorder=2,
            edgecolor="black",
            alpha=0.6,
        )
        legend = ax.legend(
            *traj.legend_elements(),
            loc="upper right",
            bbox_to_anchor=(1.2, 0.94),
            title="States",
        )
        ax.add_artist(legend)
        if ymax is not None:
            ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        _colorline(x, y, cmap=cmap, zorder=1)
        norm = mpl.colors.Normalize(vmin=0, vmax=48)
        cax = fig.add_axes([0.94, 0.15, 0.05, 0.3])
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            orientation="vertical",
            label="Time",
        )
        plt.show()

        return fig, ax

    def plot_states(self, ymax: Optional[float] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot cell states.

        Parameters
        ----------
        ymax : int, optional
            Limit for y-axis

        Returns
        -------
        matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
            Figure and Axes
        """
        bin_rd, bin_dt = self.signature[0, :], self.signature[1, :]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
        cmap = plt.get_cmap("Oranges")
        N = 12
        width = (2 * np.pi) / N
        ax.bar(self.bin_means, bin_rd, width=width, color=cmap(bin_dt))
        if ymax is not None:
            ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cax = fig.add_axes([0.94, 0.15, 0.05, 0.3])
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            orientation="vertical",
            label="Increasing state dwell time",
            ticks=[0, 0.5, 1],
        )

        return fig, ax

    def plot_transition(self, ymax=None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot transition between cell states.

        Parameters
        ----------
        ymax : int, optional
            Limit for y-axis

        Returns
        -------
        matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
            Figure and Axes
        """
        bin_rd, bin_dt = self.signature[2, :], self.signature[3, :]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
        cmap = plt.get_cmap("Blues")
        N = 12
        width = (2 * np.pi) / N
        ax.bar(self.bin_means, bin_rd, width=width, color=cmap(bin_dt))
        if ymax is not None:
            ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cax = fig.add_axes([0.94, 0.15, 0.05, 0.3])
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            orientation="vertical",
            label="Increasing transition dwell time",
            ticks=[0, 0.5, 1],
        )

        return fig, ax


def _colorline(
    x,
    y,
    z=None,
    cmap=plt.get_cmap("copper"),
    norm=plt.Normalize(0.0, 1.0),
    linewidth=3,
    alpha=1.0,
    zorder=1,
):
    """
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = _make_segments(x, y)
    lc = mcoll.LineCollection(
        segments,
        array=z,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def _make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def _truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    https://stackoverflow.com/a/18926541
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap
