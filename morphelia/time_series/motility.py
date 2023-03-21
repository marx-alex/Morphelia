from typing import Optional, Union, Tuple
import math

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats, optimize
from skimage import filters
from tqdm import tqdm


def cell_motility(
    adata: ad.AnnData,
    x_loc: str,
    y_loc: str,
    track_id: str = "Metadata_Track",
    time_var: str = "Metadata_Time",
    fpu: int = 1,
    msd_max_tau: Optional[int] = 30,
    kurtosis_max_tau: Optional[int] = 3,
    autocorr_max_tau: Optional[int] = 10,
    store_vars: Optional[Union[str, list]] = None,
) -> pd.DataFrame:
    """
    Calculates cell motility parameters based on the x- and y-coordinates of single cells over time.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    x_loc : str
        Location of cell on x-coordinate
    y_loc : str
        Location of cell on y-coordinate
    track_id : str
        Name of track identifiers in '.obs'
    time_var : str
        Name of time variable in '.obs'
    fpu : int
        Frames per unit
    msd_max_tau : int
        Maximal tau for Mean Squared Displacement
    kurtosis_max_tau : int
        Maximal tau for the calculation of the kurtosis of the displacement distribution
    autocorr_max_tau : int
        Maximal tau for Autocorrelation
    store_vars : str, list
        Store additional variables for every track

    Returns
    -------
    pd.DataFrame
        Motility measurements for every cell track with given length
    """
    if isinstance(store_vars, str):
        store_vars = [store_vars]

    min_track_len = max(msd_max_tau, autocorr_max_tau) + 1

    output = []

    for track, sdata in tqdm(adata.obs.groupby(track_id)):
        sdata = sdata.sort_values(time_var)
        path = sdata[[x_loc, y_loc]].values
        ts_len = len(sdata)

        if ts_len >= min_track_len:
            mot = CellMotility(
                path=path,
                fpu=fpu,
                msd_max_tau=msd_max_tau,
                kurtosis_max_tau=kurtosis_max_tau,
                autocorr_max_tau=autocorr_max_tau,
            )
            results = mot.result()
            # add id
            results = pd.concat((pd.Series({"ID": track}), results))
            # add other variables
            for var in store_vars:
                results[var] = sdata.iloc[0, :][var]
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
        Maximal lag for the autocorrelation calculation
    """

    def __init__(
        self,
        path: np.ndarray,
        fpu: int = 1,
        msd_max_tau: Optional[int] = 30,
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
            stats.median_absolute_deviation(self.speed),
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
        mean_vect = vect.mean(axis=0)
        mean_phi = np.arctan2(mean_vect[1], mean_vect[0])
        rho = np.sqrt(np.sum(vect ** 2, axis=1))
        circular_std = np.sqrt(-2 * np.ln(rho.mean()))
        return (mean_phi, circular_std)

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
            Persistance of cell migration

        References
        ----------
        Tee JY, Mackay-Sim A. Directional Persistence of Cell Migration in Schizophrenia Patient-Derived Olfactory Cells.
        International Journal of Molecular Sciences. 2021; 22(17):9177. https://doi.org/10.3390/ijms22179177
        """
        va = self._vacf()
        stop = (1 / self.fpu) * (len(va) - 1)
        xdata = np.linspace(0, stop, len(va))
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
    return np.exp(-x / a)
