from typing import Optional, Union, Tuple
import math
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats, optimize
from skimage import filters
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def cell_motility(
    adata: ad.AnnData,
    x_loc: Optional[str] = None,
    y_loc: Optional[str] = None,
    use_rep: Optional[str] = None,
    track_id: str = "Metadata_Track",
    time_var: str = "Metadata_Time",
    state_var: Optional[str] = None,
    fpu: int = 1,
    min_len: Optional[int] = 30,
    msd_max_tau: Optional[int] = None,
    kurtosis_max_tau: Optional[int] = 3,
    autocorr_max_tau: Optional[int] = 10,
    dependence_max_tau: Optional[int] = 3,
    store_vars: Optional[Union[str, list]] = None,
) -> pd.DataFrame:
    """Calculates cell motility parameters based on the x- and y-coordinates of single cells over time.
    Instead of location parameters also other variables can be analyzed.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    x_loc : str, optional
        Location of cell on x-coordinate
    y_loc : str, optional
        Location of cell on y-coordinate
    use_rep : str, optional
        If specified, a representation in `.obsm` is used as path, `x_loc` and `y_loc` is ignored.
    track_id : str
        Name of track identifiers in '.obs'
    time_var : str
        Name of time variable in '.obs'
    state_var : str, optional
        Variable in `.obs` with information about cell states.
        If this variable is given,  state features are calculated as well.
    fpu : int
        Frames per unit
    min_len : int, optional
        Minimum length of track to consider. Only used if `add_mot` is False.
    msd_max_tau : int
        Maximal tau for Mean Squared Displacement
    kurtosis_max_tau : int
        Maximal tau for the calculation of the kurtosis of the displacement distribution
    autocorr_max_tau : int
        Maximal tau for Autocorrelation
    dependence_max_tau : int
        Maximal lag for all signed dependence measures
    store_vars : str, list
        Store additional variables for every track

    Returns
    -------
    pd.DataFrame
        Motility measurements for every cell track with given length
    """
    if isinstance(store_vars, str):
        store_vars = [store_vars]

    if state_var is not None:
        unique_states = np.sort(np.unique(adata.obs[state_var]))
    else:
        unique_states = None

    if min_len is not None:
        taus = [msd_max_tau, autocorr_max_tau, kurtosis_max_tau, dependence_max_tau]
        taus = [tau for tau in taus if tau is not None]
        assert all([tau < min_len for tau in taus]), f"All taus must be smaller than min_len ({min_len})"

    output = []

    for track, sdata in tqdm(adata.obs.groupby(track_id)):
        sdata = sdata.sort_values(time_var)
        index = sdata.index
        if use_rep is not None:
            path = adata[index, :].obsm[use_rep][:, :2]
        else:
            path = sdata[[x_loc, y_loc]].values
        ts_len = len(sdata)

        accept = True if min_len is None else ts_len >= min_len
        if accept:
            if state_var is None:
                mot = CellMotility(
                    path=path,
                    fpu=fpu,
                    msd_max_tau=msd_max_tau,
                    kurtosis_max_tau=kurtosis_max_tau,
                    autocorr_max_tau=autocorr_max_tau,
                )
            else:
                states = adata[index, :].obs[state_var].to_numpy()
                mot = CellStateMotility(
                    path=path,
                    states=states,
                    unique_states=unique_states,
                    fpu=fpu,
                    msd_max_tau=msd_max_tau,
                    kurtosis_max_tau=kurtosis_max_tau,
                    autocorr_max_tau=autocorr_max_tau,
                    dependence_max_tau=dependence_max_tau
                )

            results = mot.result()
            # add id and other information to results
            track_info = pd.Series({"Track": track})
            if store_vars is not None:
                track_info = pd.concat((track_info, sdata.iloc[0, :][store_vars]))
            results = pd.concat((track_info, results))

            output.append(results)

    output = pd.DataFrame(output)
    return output


class CellMotility:
    """Class for motility calculations of cell trajectories.

    Parameters
    ----------
    path : numpy.ndarray
        Trajectory coordinates with dimensions of path length x 2
    fpu : int
        Frames per unit
    msd_max_tau : int
        Maximal lag for the calculation of mean squared displacement
    kurtosis_max_tau : int
        Maximal tau for the calculation of the kurtosis of the displacement distribution
    autocorr_max_tau : int
        Maximal lag for the autocorrelation and circular autocorrelation calculation
    """

    def __init__(
        self,
        path: np.ndarray,
        fpu: int = 1,
        msd_max_tau: Optional[int] = None,
        kurtosis_max_tau: Optional[int] = 3,
        autocorr_max_tau: Optional[int] = 10,
    ):
        assert path.shape[-1] == 2, "Class only accepts 2d coordinates"
        self.path = path
        self.n = len(path)
        # frames per unit
        self.fpu = fpu

        # assign the max tau for msd calculation
        if msd_max_tau is None:
            msd_max_tau = self.n - 1
        else:
            assert msd_max_tau < self.n, "msd_max_tau must be smaller than path length"
        self.msd_max_tau = msd_max_tau

        # assign the max tau for kurtosis calculation
        if kurtosis_max_tau is None:
            kurtosis_max_tau = self.n - 1
        else:
            assert (
                kurtosis_max_tau < self.n
            ), "kurtosis_max_tau must be smaller than path length"
        self.kurtosis_max_tau = kurtosis_max_tau

        # assign the max tau for autocorrelation calculation
        if autocorr_max_tau is None:
            autocorr_max_tau = self.n - 1
        else:
            assert (
                autocorr_max_tau < self.n
            ), "autocorr_max_tau must be smaller than path length"
        self.autocorr_max_tau = autocorr_max_tau

        # distance and speed metrics
        self.displacements = self.calc_displacements()
        self.cum_displacements = self._cum_displacements()
        self.speed = self._speed()

    def result(self) -> pd.Series:
        """Returns all motility parameters.

        Returns
        -------
        pandas.Series
            Series with all motility parameters for the given trajectory path
        """
        otsu, active_frac, active_avg_speed = self.speed_otsu()
        (
            avg_speed,
            std_speed,
            median_speed,
            mad_speed,
            min_speed,
            max_speed,
        ) = self.speed_props()
        avg_angle, std_angle = self.angle_props()
        (
            avg_displacement,
            std_displacement,
            skew_displacement,
        ) = self.displacement_props()
        result = pd.Series(
            {
                "Length": self.n,
                "AccumDisplacement": self.cum_displacements[-1],
                "DisplacementFromOrigin": self.displacements_from_origin()[-1],
                "DirectionalityRatio": self.directionality_ratio(),
                "GyrationRadius": self.gyration_radius(),
                "AvgSpeed": avg_speed,
                "StdSpeed": std_speed,
                "MedianSpeed": median_speed,
                "MADSpeed": mad_speed,
                "MinSpeed": min_speed,
                "MaxSpeed": max_speed,
                "OtsuSpeed": otsu,
                "ActiveFrac": active_frac,
                "ActiveAvgSpeed": active_avg_speed,
                "AvgAngle": avg_angle,
                "StdAngle": std_angle,
                "Linearity": self.linearity(),
                "SquaredSpearmanRho": self.spearman(),
                "MSDalpha": self.msd_alpha(),
                "Persistence": self.persistence(),
                "NonGaussAlpha2": self.non_gauss_alpha2(),
                "AvgDisplacement": avg_displacement,
                "StdDisplacement": std_displacement,
                "SkewDisplacement": skew_displacement,
                "Hurst": self.hurst_mandelbrot(),
            }
        )

        kurtosis = self.displacement_kurtosis()
        for i in range(self.kurtosis_max_tau):
            result[f"DisplacementKurtosis_{i+1}"] = kurtosis[i]

        ac = self.autocorrelation()
        for i in range(self.autocorr_max_tau):
            result[f"Autocorrelation_{i+1}"] = ac[i]

        angle_ac = self.angle_autocorrelation()
        for i in range(self.autocorr_max_tau):
            result[f"AngleAutocorrelation_{i + 1}"] = angle_ac[i]

        return result

    def calc_displacements(self, tau=1) -> np.ndarray:
        """Displacements.

        Parameters
        ----------
        tau : int
            Time lag

        Returns
        -------
        numpy.ndarray
            Array of length path length - 1
        """
        distance = np.sqrt(
            np.sum((self.path[: (self.n - tau), :] - self.path[tau:, :]) ** 2, axis=1)
        )
        return distance

    def _cum_displacements(self) -> np.ndarray:
        """Cumulative Distance from origin."""
        return np.cumsum(self.displacements)

    def displacements_from_origin(self) -> np.ndarray:
        """Distance from origin.

        Returns
        -------
        numpy.ndarray
            Distance for all path coordinates from the first coordinate
        """
        distance = np.sqrt(np.sum((self.path[0, :] - self.path) ** 2, axis=1))
        return distance

    def directionality_ratio(self) -> float:
        r"""Directionality Ratio.

        .. math::
            Directionality\:ratio = \frac{Cell\:displacement\:from\:origin}{Cell\:accumulated\:displacement}

        Returns
        -------
        float
            Directionality ratio
        """
        return self.displacements_from_origin()[-1] / self.cum_displacements[-1]

    def gyration_radius(self) -> float:
        r"""Radius of Gyration.

        Calculates the Radius of Gyration which is defined as the root mean squared distance of the cells
        to their center. The center is the mean position.

        .. math::
            R_g = \sqrt{\sum_{i=0}^{n-1}{\frac{1}{n}(x(t_i)-x_0)^2}}

        Returns
        -------
        float
            Radius of Gyration
        """
        center_of_mass = np.mean(self.path, axis=0)
        return np.sqrt(
            np.sum(np.sum((self.path - center_of_mass) ** 2, axis=1) / self.n)
        )

    def _speed(self) -> np.ndarray:
        floor = len(self.displacements) // self.fpu
        dist = self.displacements[: self.fpu * floor]
        return np.sum(dist.reshape(-1, self.fpu), axis=1)

    def speed_props(self) -> Tuple[float, float, float, float, float, float]:
        """Calculates mean, standard deviation, median, median absolute deviation, minimum and maximum speed.

        Returns
        -------
        tuple
            Mean, standard deviation, median, median absolute deviation, minimum and maximum speed
        """
        return (
            self.speed.mean(),
            self.speed.std(),
            float(np.median(self.speed)),
            stats.median_abs_deviation(self.speed),
            np.min(self.speed),
            np.max(self.speed),
        )

    def speed_otsu(self) -> Tuple[float, float, float]:
        """Speed Otsu theshold.

        Calculates an Otsu threshold of all velocities to distinguish slow from fast states.
        The threshold is returned together with the proportion of both states and the average
        speed of the active state.

        Returns
        -------
        tuple
            Otsu threshold, Fraction of active states, Average speed in active state
        """
        otsu = filters.threshold_otsu(self.speed)
        active_frac = len(self.speed[self.speed > otsu]) / len(self.speed)
        active_avg_speed = self.speed[self.speed > otsu].mean()
        return otsu, active_frac, active_avg_speed

    def angle_props(self) -> Tuple[float, float]:
        r"""Calculates the properties of the angle distribution.

        Angles are calculates as following:

        .. math::
            \varphi_i = \arctan2(x_i, y_i)

        The circular mean is defined as:

        .. math::
            {\overline{\varphi}} = \arctan2(\overline{x}, \overline{y})

        The circular standard deviation is defined as:

        .. math::
            \sigma_{\varphi} = \sqrt{-2 \ln(\overline{R})}

        with:

        .. math::
            \overline{R} = \frac{1}{N} \sum _{i=1}^{N} \sqrt{x_{i}^{2} + y_{i}^{2}}

        Returns
        -------
        tuple
            Mean and standard deviation of the angle distribution.
        """
        vect = np.diff(self.path, axis=0)
        phi = np.arctan2(vect[1], vect[0])
        circ_mean = stats.circmean(phi, low=-np.pi, high=np.pi)
        circ_std = stats.circstd(phi, low=-np.pi, high=np.pi)
        return circ_mean, circ_std

    def angle_autocorrelation(self):
        r"""Circular autocorrelation of angles.

        The angles of cell movement is the movement direction on a uniform polar system.
        We consider the underlying process to be stationary. The circular autocorrelation function is then defined as:

        .. math::
            R_c(k) := R_c(\phi_0, \phi_k), \; k \geq  0

        The circular correlation coefficient as introduced by Fisher and Lee (1983) can be written as:

        .. math::
            R_c(k) = \frac{E[\cos(\phi_0)\cos(\phi_k)] \cdot E[\sin(\phi_0)\sin(\phi_k)] -
            E[\sin(\phi_0)\cos(\phi_k)] \cdot E[\cos(\phi_0)\sin(\phi_k)]}
            {(1 - E[\cos(\phi_0)^2]) \cdot E[\cos(\phi_0)^2] - (E[\sin(\phi_0) \cos(\phi_0)])^2}

        Returns
        -------
        numpy.ndarray
            Circular autocorrelation for all lags between 1 and autocorr_max_tau

        References
        ----------
        Fisher, N. I., & Lee, A. J. (1983). A Correlation Coefficient for Circular Data.
        Biometrika, 70(2), 327–332. https://doi.org/10.2307/2335547
        Holzmann, H., Munk, A., Suster, M. et al. Hidden Markov models for circular and linear-circular time series.
        Environ Ecol Stat 13, 325–347 (2006). https://doi.org/10.1007/s10651-006-0015-7
        """

        vect = np.diff(self.path, axis=0)
        phi = np.arctan2(vect[:, 1], vect[:, 0])  # angles

        ac = []
        for tau in range(1, self.autocorr_max_tau + 1):
            phi_0 = phi[: len(phi) - tau]
            phi_t = phi[tau:]

            sin_phi_0, cos_phi_0 = np.sin(phi_0), np.cos(phi_0)
            sin_phi_t, cos_phi_t = np.sin(phi_t), np.cos(phi_t)

            numerator = np.mean(cos_phi_0 * cos_phi_t) * np.mean(
                sin_phi_0 * sin_phi_t
            ) - np.mean(sin_phi_0 * cos_phi_t) * np.mean(cos_phi_0 * sin_phi_t)
            denominator = (1 - np.mean(cos_phi_0 ** 2)) * np.mean(cos_phi_0 ** 2) - (
                np.mean(sin_phi_0 * cos_phi_0)
            ) ** 2
            ac.append(numerator / denominator)
        return np.array(ac)

    def linearity(self) -> float:
        """Linearity.

        R-squared value of a linear regression of the path coordinates.

        Returns
        -------
        float
            R-squared value
        """
        _, _, r_value, _, _ = stats.linregress(self.path[:, 0], self.path[:, 1])
        return r_value ** 2

    def spearman(self) -> float:
        """Squared Spearman Rho.

        Squared Spearman Rho of the path coordinates.

        Returns
        -------
        float
            Squared Spearman Rho
        """
        if len(np.unique(self.path[:, 0])) == 1 or len(np.unique(self.path[:, 1])) == 1:
            # no displacement in one coordinate
            rho = 0.0
        else:
            rho, _ = stats.spearmanr(self.path)
        return rho ** 2

    def squared_displacement(self) -> np.ndarray:
        """Squared displacement from the origin.

        Returns
        -------
        numpy.ndarray
            Squared displacement
        """
        return np.sum((self.path - self.path[0, :]) ** 2, axis=1)

    def _msd(self, tau):
        """MSD of a given path."""
        assert tau < self.n, "tau must be smaller then path length"
        sd = np.sum((self.path[: (self.n - tau), :] - self.path[tau:, :]) ** 2, axis=1)
        return np.mean(sd)

    def _msd_distribution(self):
        msd_distribution = []
        for tau in range(1, self.msd_max_tau + 1):
            msd_distribution.append(self._msd(tau))
        return np.array(msd_distribution)

    def msd_alpha(self) -> float:
        r"""Mean squared displacement alpha value.

        Slope of the mean squared displacement in relation to tau in log-log space.
        This is a indication of the underlying diffusion process:
            alpha > 1: superdiffusion
            alpha = 1: normal diffusion
            alpha < 1: subdiffusion

        The mean squared displacement with different values for tau is calculated as following:

        .. math::
            MSD_{\tau}=\langle (x_{t+\tau}-x_t)^2\rangle

        Returns
        -------
        float
            Mean squared displacement alpha value
        """
        msd_distribution = self._msd_distribution()
        # return nan if msd distribution ha zeros
        if sum(msd_distribution == 0) > 0:
            return np.nan

        tau = np.arange(1, self.msd_max_tau + 1)
        msd_alpha, _, _, _, _ = stats.linregress(
            np.log(tau), np.log(msd_distribution)
        )  # slope in log-log space
        return msd_alpha

    def _vacf(self):
        """Velocity Autocorrelation Function"""
        vect = np.diff(self.path, axis=0)
        va = np.sum(vect * vect[0, :], axis=1)
        va = va / va[0]
        return va

    def persistence(self) -> float:
        r"""Calculates the inverse of the decay rate of the velocity autocorrelation function,
        also known as cell persistence.

        The velocity autocorrelation function is the following:

        .. math::
            R_v(t) = \frac{v_{t_0} \cdot v_{t+t_0}}{v_{t_0} \cdot v_{t_0}}

        The exponential decay rate tau is calculated with:

        .. math::
            R(t) = \exp^{-\frac{t}{\tau}}

        Returns
        -------
        float
            Persistence of cell migration

        References
        ----------
        Tee JY, Mackay-Sim A. Directional Persistence of Cell Migration in Schizophrenia Patient-Derived Olfactory Cells.
        International Journal of Molecular Sciences. 2021; 22(17):9177. https://doi.org/10.3390/ijms22179177
        """
        va = self._vacf()
        stop = (1 / self.fpu) * (len(va) - 1)
        xdata = np.linspace(0, stop, len(va))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, _ = optimize.curve_fit(exponential_decay, xdata, va)

        return popt[0]

    def displacement_kurtosis(self) -> np.ndarray:
        r"""Calculates the kurtosis of the displacement distribution. We consider different time lags tau
        for the calculation of the displacements. A higher kurtosis indicates a levy's flight.

        Kurtosis is the standard fourth central moment:

        .. math::
            Kurtosis(d_{\tau}) = \langle d_i^4 - \bar{d} \rangle - 3

        Returns
        -------
        numpy.ndarray
            Kurtosis values for every time lag
        """
        kurtosis = []
        for tau in range(1, self.kurtosis_max_tau + 1):
            displacements = self.calc_displacements(tau)
            kurtosis.append(stats.kurtosis(displacements))
        return np.array(kurtosis)

    def non_gauss_alpha2(self) -> float:
        r"""Calculates the non-gaussian parameter a2 for the displacement distribution.
        The a2 parameter is the fourth moment of the distribution relative to its second moment.
        The non-Gaussian parameter measures how much the tail of the distribution deviates from Gaussian, and is zero
        for a Gaussian distribution.
        Broader distributions (but with the same standard deviation) result in higher values of a2.
        Levy-like motion would be expected to have alpha_2 > 0.

        .. math::
            \alpha_2 = \frac{\langle d_i^4 \rangle}{3 {\langle d_i^2 \rangle}^2} - 1

        with:
        d: displacement
        < and > denotes averaging over the distribution

        Returns
        -------
        float
            Non-gaussian alpha 2

        References
        ----------
        Parry BR, Surovtsev IV, Cabeen MT, O'Hern CS, Dufresne ER, Jacobs-Wagner C.
        The bacterial cytoplasm has glass-like properties and is fluidized by metabolic activity.
        Cell. 2014 Jan 16;156(1-2):183-94. doi: 10.1016/j.cell.2013.11.028
        """
        return (
            np.mean(self.displacements ** 4)
            / (3 * np.mean(self.displacements ** 2) ** 2)
        ) - 1

    def displacement_props(self) -> Tuple[float, float, float]:
        """Calculates the mean, standard deviation and the skewness of the displacement distribution.

        Returns
        -------
        tuple
            Mean, standard deviation and skewness of displacement distribution
        """
        return (
            self.displacements.mean(),
            self.displacements.std(),
            float(stats.skew(self.displacements)),
        )

    def hurst_mandelbrot(self) -> float:
        """Calculates the Hurst coefficient for cell movement.

        The Hurst coefficient is estimated with Mandelbrot's rescaled range method (see references).

        H : 0.5 - 1 ; long-term positive autocorrelation
        H : 0.5 ; fractal Brownian motion
        H : 0-0.5 ; long-term negative autocorrelation

        Returns
        -------
        float
            Hurst coefficient

        References
        ----------
        Mandelbrot, B. B., and Wallis, J. R. (1969),
        Some long-run properties of geophysical records, Water Resour. Res., 5( 2), 321– 340,
        doi:10.1029/WR005i002p00321
        """
        largest_pow2 = math.floor(math.log2(self.n - 1))
        ns = (self.n - 1) / (2 ** np.arange(largest_pow2))
        ns = ns.astype(int)

        RSl = []
        for n in ns:
            RS = []
            # create subsereies of size n
            for i in range((self.n - 1) // n):
                subseries = self.displacements[i * n : (n + (i * n))]

                # calculate rescaled range R/S for subseries n
                Z = np.cumsum(subseries - np.mean(subseries))
                R = Z.max() - Z.min()
                S = np.std(subseries)
                if S > 0:
                    RS.append(R / S)

            RSl.append(np.mean(RS))

        hurst, _, _, _, _ = stats.linregress(np.log(ns), np.log(RSl))
        return hurst

    def autocorrelation(self) -> np.ndarray:
        r"""Calculates autocorrelation of cell displacements.

        The autocorrelation is estimated with:

        .. math::
            R(\tau) = \frac{1}{(n - \tau)s^2} \sum_{t=1}^{n-\tau}(d_t - \bar{d})(d_{t+\tau} - \bar{d})

        Returns
        -------
        numpy.ndarray
            Autocorrelation for every lag tau
        """
        n = len(self.displacements)
        norm = self.displacements - np.mean(self.displacements)

        acs = []
        # calculate autocorrelations
        for tau in range(1, self.autocorr_max_tau + 1):
            ac = (
                np.sum(norm[: (n - tau)] * norm[tau:])
                / (n - tau)
                / np.var(self.displacements)
            )
            acs.append(ac)

        return np.array(acs)


def exponential_decay(x, a):
    with np.errstate(over='ignore'):
        decay = np.exp(-x / a)
    return decay


class CellStateMotility(CellMotility):
    """Class for motility calculations of cell trajectories using location and state information.

    Parameters
    ----------
    path : numpy.ndarray
        Trajectory coordinates with dimensions of path length x 2
    states : numpy.ndarray
        Cell states
    unique_states : numpy.ndarray, optional
        If not all possibly states occur in this trajectory, the actual states can be defined here.
        Otherwise, it is assumed stat all possible states occur in the trajectory.
    fpu : int
        Frames per unit
    msd_max_tau : int
        Maximal lag for the calculation of mean squared displacement
    kurtosis_max_tau : int
        Maximal tau for the calculation of the kurtosis of the displacement distribution
    autocorr_max_tau : int
        Maximal lag for the autocorrelation and circular autocorrelation calculation
    dependence_max_tau : int
        Maximal lag for all signed dependence measures
    """
    def __init__(
        self,
        path: np.ndarray,
        states: np.ndarray,
        unique_states: Optional[np.ndarray] = None,
        fpu: int = 1,
        msd_max_tau: Optional[int] = 30,
        kurtosis_max_tau: Optional[int] = 3,
        autocorr_max_tau: Optional[int] = 10,
        dependence_max_tau: Optional[int] = 3
    ):
        super().__init__(
            path=path,
            fpu=fpu,
            msd_max_tau=msd_max_tau,
            kurtosis_max_tau=kurtosis_max_tau,
            autocorr_max_tau=autocorr_max_tau
        )
        if not isinstance(states, np.ndarray):
            states = np.array(states)
        if not np.issubdtype(states.dtype, np.integer):
            states = states.astype(int)
        self.states = states

        if unique_states is None:
            unique_states = np.sort(np.unique(states))
        else:
            unique_states = np.sort(unique_states)
        if not np.issubdtype(unique_states.dtype, np.integer):
            unique_states = unique_states.astype(int)
        self.unique_states = unique_states
        self.n_states = len(unique_states)

        # assign the max tau for signed dependence calculation
        if dependence_max_tau is None:
            dependence_max_tau = self.n - 1
        else:
            assert (
                    dependence_max_tau < self.n
            ), "dependence_max_tau must be smaller than path length"
        self.dependence_max_tau = dependence_max_tau

        # estimated marginal probabilities
        _, counts = np.unique(states, return_counts=True)
        self.marginal_prob = counts / np.sum(counts)

    def result(self) -> pd.Series:
        """Returns all state motility parameters.

        Returns
        -------
        pandas.Series
            Series with all state motility parameters for the given trajectory path
        """
        otsu, active_frac, active_avg_speed = self.speed_otsu()
        (
            avg_speed,
            std_speed,
            median_speed,
            mad_speed,
            min_speed,
            max_speed,
        ) = self.speed_props()
        avg_angle, std_angle = self.angle_props()
        (
            avg_displacement,
            std_displacement,
            skew_displacement,
        ) = self.displacement_props()
        (
            entropy,
            gini,
            chebycheff
        ) = self.state_dispersion()
        result = pd.Series(
            {
                "Length": self.n,
                "AccumDisplacement": self.cum_displacements[-1],
                "DisplacementFromOrigin": self.displacements_from_origin()[-1],
                "DirectionalityRatio": self.directionality_ratio(),
                "GyrationRadius": self.gyration_radius(),
                "AvgSpeed": avg_speed,
                "StdSpeed": std_speed,
                "MedianSpeed": median_speed,
                "MADSpeed": mad_speed,
                "MinSpeed": min_speed,
                "MaxSpeed": max_speed,
                "OtsuSpeed": otsu,
                "ActiveFrac": active_frac,
                "ActiveAvgSpeed": active_avg_speed,
                "AvgAngle": avg_angle,
                "StdAngle": std_angle,
                "Linearity": self.linearity(),
                "SquaredSpearmanRho": self.spearman(),
                "MSDalpha": self.msd_alpha(),
                "Persistence": self.persistence(),
                "NonGaussAlpha2": self.non_gauss_alpha2(),
                "AvgDisplacement": avg_displacement,
                "StdDisplacement": std_displacement,
                "SkewDisplacement": skew_displacement,
                "Hurst": self.hurst_mandelbrot(),
                "StatesEntropy": entropy,
                "StatesGini": gini,
                "StatesChebycheff": chebycheff,
                "StateTransitionProb": self.state_transition_prob(),
                "StatesCovered": self.states_covered()
            }
        )

        kurtosis = self.displacement_kurtosis()
        for i in range(self.kurtosis_max_tau):
            result[f"DisplacementKurtosis_{i+1}"] = kurtosis[i]

        ac = self.autocorrelation()
        for i in range(self.autocorr_max_tau):
            result[f"Autocorrelation_{i+1}"] = ac[i]

        angle_ac = self.angle_autocorrelation()
        for i in range(self.autocorr_max_tau):
            result[f"AngleAutocorrelation_{i + 1}"] = angle_ac[i]

        state_dwell_time = self.state_dwell_time()
        for i, state in enumerate(self.unique_states):
            result[f"StateDwellTime_{state}"] = state_dwell_time[i]

        state_stationary_dist = self.states_stationary_dist()
        for i, state in enumerate(self.unique_states):
            result[f"StateStationaryDist{state}"] = state_stationary_dist[i]

        # measures of serial dependence
        cohens_kappa = self.cohens_kappa()
        cramers_vi = self.cramers_vi()
        sakoda = self.sakoda()
        gk_tau = self.gk_tau()
        gk_lambda = self.gk_lambda()
        uncertainty_coeff = self.uncertainty_coeff()
        for i in range(self.dependence_max_tau):
            result[f"StatesCohenKappa_{i+1}"] = cohens_kappa[i]
            result[f"StatesCramerVi_{i + 1}"] = cramers_vi[i]
            result[f"StatesSakoda_{i + 1}"] = sakoda[i]
            result[f"StatesGKTau_{i + 1}"] = gk_tau[i]
            result[f"StatesGKLambda_{i + 1}"] = gk_lambda[i]
            result[f"StatesUncertainty_{i + 1}"] = uncertainty_coeff[i]

        return result

    def state_dwell_time(self) -> np.ndarray:
        """The state dwell time is the percentage a cell spend in a certain state.

        Returns
        -------
        numpy.ndarray
            Dwell time for every unique state
        """
        dwell_time = []
        for st in self.unique_states:
            dwell_time.append(np.count_nonzero(self.states == st))
        dwell_time = np.array(dwell_time)
        dwell_time = dwell_time / np.sum(dwell_time)
        return dwell_time

    def state_dispersion(self) -> Tuple[float, float, float]:
        r"""Estimated measures of dispersion for categorical time series:

        Entropy:

        .. math::
            \nu_E = -\frac{1}{ln(m+1)}\sum_j p_j ln(p_j)

        Gini index:

        .. math::
            \nu_G = \frac{m + 1}{m}(1 - \sum_j p_j^2)

        Chebycheff dispersion:

        .. math::
            \nu_C = \frac{m+1}{m} (1 - max_j p_j)

        Returns
        -------
        float, float, float
            Entropy, Gini index, Chebycheff dispersion
        """
        # marginal probabilities
        _, counts = np.unique(self.states, return_counts=True)
        p = counts / counts.sum()

        entropy = -(1 / np.log(self.n_states + 1)) * np.sum(p * np.log(p))
        gini = ((self.n_states + 1) / self.n_states) * (1 - np.sum(p ** 2))
        chebycheff = ((self.n_states + 1) / self.n_states) * (1 - np.max(p))
        return entropy, gini, chebycheff

    def state_transition_prob(self) -> float:
        """The transition probability is the number of state transitions divided
        by the number of transitions and non-transitions.

        Returns
        -------
        float
            State transition probability
        """
        transition = []
        for i in range(self.n - 1):
            if self.states[i] != self.states[i+1]:
                transition.append(1)
            else:
                transition.append(0)

        return sum(transition) / len(transition)

    def states_covered(self) -> float:
        """Fraction of unique states that the cell walks through.
        This is only unequal 1 if the number of unique states differs
        to the number of unique states of the trajectory.

        Returns
        -------
        float
            Fraction of states covered by the trajectory
        """
        return len(np.unique(self.states)) / len(self.unique_states)

    def _transition_matrix(self) -> np.ndarray:
        unique_states = np.sort(np.unique(self.states))
        state_idxs = {st: i for i, st in enumerate(unique_states)}
        n_states = len(unique_states)
        T = np.zeros((n_states, n_states))

        for state_a, state_b in zip(self.states[:-1], self.states[1:]):
            T[state_idxs[state_a], state_idxs[state_b]] += 1

        T_sum = T.sum(axis=1)
        T_sum[T_sum == 0] = 1   # avoid division by zero
        return T / T_sum.reshape(-1, 1)

    def states_stationary_dist(self) -> np.ndarray:
        """Stationary state distribution.

        Returns
        -------
        numpy.ndarray
            Stationary distribution
        """
        T = self._transition_matrix()
        eigenvals, eigenvects = np.linalg.eig(T.T)

        close_to_1_idx = np.argmax(eigenvals)
        target_eigenvect = eigenvects[:, close_to_1_idx]

        _stationary_dist = target_eigenvect / target_eigenvect.sum()
        _stationary_dist = _stationary_dist.real
        stationary_dist = []

        ix = 0

        for state in self.unique_states:
            if state in self.states:
                stationary_dist.append(_stationary_dist[ix])
                ix += 1
            else:
                stationary_dist.append(0)

        return np.array(stationary_dist)

    def cohens_kappa(self) -> np.ndarray:
        r"""Estimated Cohen's Kappa for the categorical time series of states and its lagged version.

        .. math::
            \kappa(k) = \frac{\sum_j (p_{jj}(k) - p_j^2)}{1 - \sum_j p_j^2}

        Returns
        -------
        numpy.ndarray
            Cohen's kappa for all given time lags

        References
        ----------
        Weiß CH, Göb R (2008). “Measuring serial dependence in categorical time series.” AStA Advances
        in Statistical Analysis, 92, 71–89
        """
        marg_prob = self.marginal_prob

        kappas = []
        for tau in range(1, self.dependence_max_tau + 1):
            if len(marg_prob) == 1:
                kappas.append(0)
            else:
                # estimated joint probabilities
                joint_prob = confusion_matrix(self.states[:-tau], self.states[tau:], normalize='all')
                joint_prob = joint_prob.diagonal()

                kappa = np.sum(joint_prob - (marg_prob ** 2))
                kappa = kappa / (1 - np.sum(marg_prob ** 2))

                kappas.append(kappa)

        return np.array(kappas)

    def cramers_vi(self) -> np.ndarray:
        r"""Estimated Cramer's Vi for the categorical time series of states and its lagged version.

        .. math::
            \upsilon(k) = \frac{\phi(k)}{\sqrt{m - 1}}

        with:

        .. math::
            \phi^2(k) = \sum_{ij} \frac{(p_{ij}(k) - p_i p_j)^2}{p_i p_j}

        Returns
        -------
        numpy.ndarray
            Cramer's vi

        References
        ----------
        Weiß CH, Göb R (2008). “Measuring serial dependence in categorical time series.” AStA Advances
        in Statistical Analysis, 92, 71–89
        """
        m = len(np.unique(self.states))

        # estimated marginal probabilities
        marg_prob = self.marginal_prob
        marg_prob = np.dot(marg_prob.reshape(-1, 1), marg_prob.reshape(1, -1))

        vis = []

        for tau in range(1, self.dependence_max_tau + 1):
            if m == 1:
                vis.append(0)
            else:
                # estimated joint probabilities
                joint_prob = confusion_matrix(self.states[:-tau], self.states[tau:], normalize='all')

                phi2 = np.sum(((joint_prob - marg_prob) ** 2) / marg_prob)

                vi = np.sqrt(phi2 / (m - 1))
                vis.append(vi)

        return np.array(vis)

    def sakoda(self) -> np.ndarray:
        r"""Estimated Sakoda measure for the categorical time series of states and its lagged version.

        .. math::
            p^*(k) = \sqrt{\frac{m \phi^2(k)}{(m - 1)(1 + \phi^2(k))}}

        with:

        .. math::
            \phi^2(k) = \sum_{ij} \frac{(p_{ij}(k) - p_i p_j)^2}{p_i p_j}

        Returns
        -------
        numpy.ndarray
            Sakoda measure

        References
        ----------
        Weiß CH, Göb R (2008). “Measuring serial dependence in categorical time series.” AStA Advances
        in Statistical Analysis, 92, 71–89
        """
        m = len(np.unique(self.states))

        # estimated marginal probabilities
        marg_prob = self.marginal_prob
        marg_prob = np.dot(marg_prob.reshape(-1, 1), marg_prob.reshape(1, -1))

        skds = []

        for tau in range(1, self.dependence_max_tau + 1):
            if m == 1:
                skds.append(0)
            else:
                # estimated joint probabilities
                joint_prob = confusion_matrix(self.states[:-tau], self.states[tau:], normalize='all')

                phi2 = np.sum(((joint_prob - marg_prob) ** 2) / marg_prob)

                # avoid division by zero
                skd = np.sqrt((m * phi2) / ((m - 1) * (1 + phi2)))
                skds.append(skd)

        return np.array(skds)

    def gk_tau(self) -> np.ndarray:
        r"""Estimated Goodman and Kruskal's Tau for the categorical time series of states and its lagged version.

        .. math::
            \tau(k) = \frac{\sum_{ij} \frac{p_{ij}(k)^2}{p_j} - \sum_i p_i^2}{1 - \sum_i p_i^2}

        Returns
        -------
        numpy.ndarray
            Goodman and Kruskal's Tau

        References
        ----------
        Weiß CH, Göb R (2008). “Measuring serial dependence in categorical time series.” AStA Advances
        in Statistical Analysis, 92, 71–89
        """
        marg_prob = self.marginal_prob

        tau_vals = []
        for tau in range(1, self.dependence_max_tau + 1):
            if len(marg_prob) == 1:
                tau_vals.append(0)
            else:
                # estimated joint probabilities
                joint_prob = confusion_matrix(self.states[:-tau], self.states[tau:], normalize='all')

                tau_val = np.sum((joint_prob ** 2) / marg_prob.reshape(1, -1)) - np.sum(marg_prob ** 2)
                tau_val = tau_val / (1 - np.sum(marg_prob ** 2))

                tau_vals.append(tau_val)

        return np.array(tau_vals)

    def gk_lambda(self) -> np.ndarray:
        r"""Estimated Goodman and Kruskal's Lambda for the categorical time series of states and its lagged version.

        .. math::
            \lambda(k) = \frac{\sum_{j} max_i p_{ij}(k) - max_i p_i}{1 - max_i p_i}

        Returns
        -------
        numpy.ndarray
            Goodman and Kruskal's Lambda

        References
        ----------
        Weiß CH, Göb R (2008). “Measuring serial dependence in categorical time series.” AStA Advances
        in Statistical Analysis, 92, 71–89
        """
        marg_prob = self.marginal_prob
        max_mp = np.max(marg_prob)

        lambda_vals = []

        for tau in range(1, self.dependence_max_tau + 1):
            if len(marg_prob) == 1:
                lambda_vals.append(0)
            else:
                # estimated joint probabilities
                joint_prob = confusion_matrix(self.states[:-tau], self.states[tau:], normalize='all')

                # maximum joint probabilities
                max_jp = joint_prob.max(axis=0)

                lambda_val = (np.sum(max_jp) - max_mp) / (1 - max_mp)
                lambda_vals.append(lambda_val)

        return np.array(lambda_vals)

    def uncertainty_coeff(self) -> np.ndarray:
        r"""Estimated Uncertainty coefficient for the categorical time series of states and its lagged version.

        .. math::
            u(k) = - \frac{\sum_{ij} p_{ij}(k) ln(\frac{p_{ij}(k)}{p_i p_j})}{\sum_i p_i ln(p_i)}

        Returns
        -------
        numpy.ndarray
            Uncertainty coefficient

        References
        ----------
        Weiß CH, Göb R (2008). “Measuring serial dependence in categorical time series.” AStA Advances
        in Statistical Analysis, 92, 71–89
        """
        marg_prob = self.marginal_prob
        marg_prob_matrix = np.dot(marg_prob.reshape(-1, 1), marg_prob.reshape(1, -1))

        coeffs = []

        for tau in range(1, self.dependence_max_tau + 1):
            if len(marg_prob) == 1:
                coeffs.append(0)
            else:
                # estimated joint probabilities
                joint_prob = confusion_matrix(self.states[:-tau], self.states[tau:], normalize='all')

                combined = joint_prob / marg_prob_matrix
                combined = np.log(combined, out=combined, where=combined > 0)

                coeff = np.sum(joint_prob * combined)

                coeff = - coeff / np.sum(marg_prob * np.log(marg_prob))
                coeffs.append(coeff)

        return np.array(coeffs)
