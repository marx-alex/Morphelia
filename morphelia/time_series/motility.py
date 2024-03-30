from typing import Optional, Union, Tuple, List
import warnings

import anndata as ad
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def motility_features(
    data: Union[ad.AnnData, pd.DataFrame],
    coords: Optional[List[str]] = None,
    use_rep: Optional[str] = None,
    rep_dims: Optional[int] = None,
    track_id: str = "Metadata_Track",
    time_var: str = "Metadata_Time",
    fpu: int = 1,
    min_len: Optional[int] = 30,
    msd_max_tau: Optional[int] = None,
    kurtosis_max_tau: Optional[int] = 3,
    autocorr_max_tau: Optional[int] = 10,
    store_vars: Optional[Union[str, list]] = None,
) -> pd.DataFrame:
    """
    This function calculates motility profiles based on n-dimensional trajectories of single cells.
    Instead of location parameters also other variables can be analyzed.

    Parameters
    ----------
    data : anndata.AnnData, pandas.DataFrame
        Trajectory data
    coords : list, tuple, optional
        Coordinates in n-dimensions
    use_rep : str, optional
        If specified, a representation in `.obsm` is used as path, 'coords' is ignored.
    rep_dims : int, optional
        Dimensions of the representation to use.
    track_id : str
        Name of track identifiers.
        Must be in '.obs' for AnnData objects.
    time_var : str
        Name of time variable. Must be in '.obs' for AnnData objects.
    fpu : int
        Frames per unit
    min_len : int, optional
        Minimum length of tracks to consider.
    msd_max_tau : int
        Maximal tau for Mean Squared Displacement
    kurtosis_max_tau : int
        Maximal tau for the calculation of the kurtosis of the displacement distribution
    autocorr_max_tau : int
        Maximal tau for autocorrelation
    store_vars : str, list
        Store additional variables for every track

    Returns
    -------
    pd.DataFrame
        Motility measurements for every cell track with given length
    """
    if use_rep is not None:
        assert isinstance(
            data, ad.AnnData
        ), "If 'use_rep' is not None, data must be an AnnData object"

    if min_len is not None:
        taus = [msd_max_tau, autocorr_max_tau, kurtosis_max_tau]
        taus = [tau for tau in taus if tau is not None]
        assert all(
            [tau < min_len for tau in taus]
        ), f"All taus must be smaller than min_len ({min_len})"

    if isinstance(store_vars, str):
        store_vars = [store_vars]

    # Group data by track
    if isinstance(data, pd.DataFrame):
        grouper = data.groupby(track_id)
    else:
        grouper = data.obs.groupby(track_id)

    output = []

    # Get motility data for every track
    for track, sdata in tqdm(grouper, desc="Motility Profiling"):
        sdata = sdata.sort_values(time_var)
        index = sdata.index
        if use_rep is not None:
            if rep_dims is None:
                path = data[index, :].obsm[use_rep]
            else:
                path = data[index, :].obsm[use_rep][:, :rep_dims]
        else:
            path = sdata[coords].values
        ts_len = len(sdata)

        accept = True if min_len is None else ts_len >= min_len
        if accept:
            mot = Motility(
                path=path,
                fpu=fpu,
                msd_max_tau=msd_max_tau,
                kurtosis_max_tau=kurtosis_max_tau,
                autocorr_max_tau=autocorr_max_tau,
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


class Motility:
    """Class for motility calculations of cell trajectories.

    Parameters
    ----------
    path : numpy.ndarray
        Trajectory coordinates with dimensions [path length x d]
    fpu : int
        Frames per unit
    msd_max_tau : int
        Maximal lag for the calculation of mean squared displacement
    kurtosis_max_tau : int
        Maximal tau for the calculation of the kurtosis of the displacement distribution
    autocorr_max_tau : int
        Maximal lag for the autocorrelation and circular autocorrelation calculation

    References
    ----------
    https://github.com/cellgeometry/heteromotility
    """

    def __init__(
        self,
        path: np.ndarray,
        fpu: int = 1,
        msd_max_tau: Optional[int] = None,
        kurtosis_max_tau: Optional[int] = 3,
        autocorr_max_tau: Optional[int] = 10,
    ):
        assert (
            len(path.shape) == 2
        ), f"path must have 2 dimensions, got {len(path.shape)}"
        assert path.shape[1] >= 2, "path must have at least 2 coordinates"
        self.path = path
        self.n = path.shape[0]
        self.dim = path.shape[1]
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
        (
            avg_speed,
            std_speed,
            median_speed,
            mad_speed,
            min_speed,
            max_speed,
        ) = self.speed_props()
        msd_alpha = self.msd_alpha()
        (
            act_ratio,
            avg_active_speed,
            avg_idling_speed,
            std_active_speed,
            std_idling_speed,
            v_thresh,
        ) = self.phase_props(msd_alpha=msd_alpha)
        avg_theta, active_theta, idling_theta = self.turn_angle_props(v_thresh=v_thresh)
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
                "ActiveRatio": act_ratio,
                "AvgActiveSpeed": avg_active_speed,
                "AvgIdlingSpeed": avg_idling_speed,
                "StdActiveSpeed": std_active_speed,
                "StdIdlingSpeed": avg_idling_speed,
                "AvgTurnAngle": avg_theta,
                "AvgActiveTurnAngle": active_theta,
                "AvgIdlingTurnAngle": idling_theta,
                "Linearity": self.mcor(),
                "Monotonicity": self.mcor_spearman(),
                "MSDalpha": msd_alpha,
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

        ac = self.displacement_autocorrelation()
        for i in range(self.autocorr_max_tau):
            result[f"DisplacementAC_{i+1}"] = ac[i]

        vac = self.velocity_autocorrelation()
        for i in range(self.autocorr_max_tau):
            result[f"VelocityAC_{i + 1}"] = vac[i]

        return result

    def calc_displacements(self, tau=1) -> np.ndarray:
        """Displacements.

        Parameters
        ----------
        tau : int
            Time lag

        Returns
        -------
        distance : numpy.ndarray
            Array of length n (path length) - 1
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
        return np.sqrt(np.mean(np.sum((self.path - center_of_mass) ** 2, axis=1)))

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

    def mcor(self) -> float:
        r"""
        We use the multi-way correlation coefficient (mcor) [Taylor 2020] as a measure of linearity.
        Mcor is applicable for the multidimensional and similar to Pearson's correlation coefficient
        in the 2-dimensional case.

        It is defined as:

        .. math::
            mcor[v_1,...,v_n] = \frac{1}{\sqrt{d}} \sqrt{\frac{1}{d-1}\sum_{i=0}^{d}(\lambda_i - \bar{\lambda})^2}

        where lambda_i is the ith eigenvalue of the empirical correlation matrix with
        column vectors v_i.

        Returns
        -------
        float
            Multi-way correlation coefficient

        References
        ----------
        Taylor, B.M. (2020). A Multi-Way Correlation Coefficient. arXiv: Methodology.
        """
        if np.any(np.std(self.path, axis=0) == 0):
            # no displacement in one coordinate
            mcor = np.nan
        else:
            try:
                evs = np.linalg.eigvals(np.corrcoef(self.path, rowvar=False))
                mcor = (1 / np.sqrt(self.dim)) * np.std(evs, ddof=1)
            except LinAlgError:
                mcor = np.nan
        return mcor

    def mcor_spearman(self) -> float:
        """
        This metrics works the same as the multi-way correlation coefficient
        in self.linearity. Instead of Pearson's correlation coefficient,
        Spearman's rank correlation coefficient is used to calculate the
        empirical correlation matrix.

        Returns
        -------
        float
            Multi-way correlation coefficient using Spearman's rank correlation coefficient

        References
        ----------
        Taylor, B.M. (2020). A Multi-Way Correlation Coefficient. arXiv: Methodology.
        """
        if np.any(np.std(self.path, axis=0) == 0):
            # no displacement in one coordinate
            mcor = np.nan
        else:
            if self.dim == 2:
                # Rho need to be converted into a correlation matrix
                rho = stats.spearmanr(self.path).statistic
                R = np.full(shape=(self.dim, self.dim), fill_value=rho)
                np.fill_diagonal(R, 1)
            else:
                # stats.spearmanr returns a correlation matrix
                R = stats.spearmanr(self.path)

            try:
                evs = np.linalg.eigvals(R)
                mcor = (1 / np.sqrt(self.dim)) * np.std(evs, ddof=1)
            except LinAlgError:
                mcor = np.nan

        return mcor

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
        This is an indication of the underlying diffusion process:
            alpha > 1: superdiffusion
            alpha = 1: normal diffusion
            alpha < 1: subdiffusion
            alpha = 2: ballistic motion

        The mean squared displacement with different values for tau is calculated as following:

        .. math::
            MSD_{\tau}=\langle (x_{t+\tau}-x_t)^2\rangle

        Returns
        -------
        float
            Mean squared displacement alpha value
        """
        msd_distribution = self._msd_distribution()
        # return nan if msd distribution has zeros
        if sum(msd_distribution == 0) > 0:
            return np.nan

        tau = np.arange(1, self.msd_max_tau + 1)
        msd_alpha, _, _, _, _ = stats.linregress(
            np.log(tau), np.log(msd_distribution)
        )  # slope in log-log space
        return msd_alpha

    def phase_props(
        self, msd_alpha: float = None
    ) -> Tuple[float, float, float, float, float, float]:
        r"""
        Cell movements are categorized into active and idling phases
        depending on a speed threshold.

        The threshold is defined by:

        .. math::
            v_{thr} = CDF_{v}^{-1} (1 - \frac{\alpha}{2})

        where alpha is the MSD alpha value.

        Parameters
        ----------
        msd_alpha : float
            Slope of the log-log MSD curve.
            If None, the alpha value is calculated from the trajectory.

        Returns
        -------
        float
            Proportion of time spent in active phase
        float
            Mean speed in active phase
        float
            Mean speed in idling phase
        float
            Speed standard deviation in active phase
        float
            Speed standard deviation in idling phase
        float
            Speed threshold to identivy active and idling cells

        References
        ----------
        Tee JY, Mackay-Sim A. Directional Persistence of Cell Migration in Schizophrenia Patient-Derived Olfactory Cells.
        International Journal of Molecular Sciences. 2021; 22(17):9177. https://doi.org/10.3390/ijms22179177
        """
        if msd_alpha is None:
            msd_alpha = self.msd_alpha()
        v = self.speed

        # Threshold to distinguish between active and idling phase
        v_sorted = np.sort(v)
        p = np.linspace(0, 1, len(v))
        cdf_inv = interp1d(p, v_sorted)
        p_thresh = np.clip(1 - (msd_alpha / 2), a_min=0, a_max=1)
        v_thresh = cdf_inv(p_thresh)
        v_active, v_idling = v[v >= v_thresh], v[v < v_thresh]
        act_ratio = len(v_active) / len(v)

        if len(v_active) > 0:
            avg_active_speed = float(np.mean(v_active))
            std_active_speed = float(np.std(v_active))
        else:
            avg_active_speed = 0.0
            std_active_speed = 0.0

        if len(v_idling) > 0:
            avg_idling_speed = float(np.mean(v_idling))
            std_idling_speed = float(np.std(v_idling))
        else:
            avg_idling_speed = 0.0
            std_idling_speed = 0.0

        return (
            act_ratio,
            avg_active_speed,
            avg_idling_speed,
            std_active_speed,
            std_idling_speed,
            v_thresh,
        )

    def turn_angle_props(self, v_thresh: float = None) -> Tuple[float, float, float]:
        """
        The mean turning angle for the whole cell trajectory as well as for the active
        and idling phase is calculated with possible values between 0 and 180 degrees.

        Parameters
        ----------
        v_thresh : float
            Speed threshold to identify active and idling cells.
            If None, the threshold is calculated.

        Returns
        -------
        float
            Mean turning angle
        float
            Mean turning angle in active phase
        float
            Mean turning angle in idling phase
        """
        if v_thresh is None:
            v_thresh = self.phase_props()[-1]
        v = self.speed

        # Vectorize the trajectory coordinates
        vect = self.path[:: self.fpu, :][1:, :] - self.path[:: self.fpu, :][:-1, :]
        # Vector Normalization
        unit_vect = vect / np.linalg.norm(vect, axis=1).reshape(-1, 1)
        # Calculate angles between vectors
        coeff = np.sum(unit_vect[:-1, :] * unit_vect[1:, :], axis=1)
        theta = np.arccos(np.clip(coeff, -1.0, 1.0))

        avg_theta = np.mean(theta) * 180 / np.pi

        if len(theta[v[:-1] >= v_thresh]) > 0:
            active_theta = np.mean(theta[v[:-1] >= v_thresh]) * 180 / np.pi
        else:
            active_theta = 0.0

        if len(theta[v[:-1] < v_thresh]) > 0:
            idling_theta = np.mean(theta[v[:-1] < v_thresh]) * 180 / np.pi
        else:
            idling_theta = 0.0

        return avg_theta, active_theta, idling_theta

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
            np.mean(self.displacements**4)
            / (3 * np.mean(self.displacements**2) ** 2)
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
            float(stats.skew(self.displacements, bias=False)),
        )

    def hurst_mandelbrot(self) -> float:
        """Calculates the Hurst coefficient for cell movement.

        The Hurst coefficient is estimated with Mandelbrot's rescaled range method (see references).

        H : 0.5 - 1 ; long-term positive autocorrelation
        H : 0.5 ; fractal Brownian motion
        H : 0-0.5 ; long-term negative autocorrelation

        This function raises a warning and return nan if the time series is too short (i.e. <= 15)
        because the linear regression will have < 3 points in those cases.

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

        N = len(self.displacements)
        # alpha is the largest power of 2 less n
        alpha = int(np.log2(N)) - 1
        if (2**alpha) >= N:
            alpha -= 1
        ns = N / (2 ** np.arange(alpha))
        ns = ns.astype(int)  # lengths of nonoverlapping time series
        nxs = N // ns  # number of time series

        rs_n = []

        for n, nx in zip(ns, nxs):
            x = self.displacements[: n * nx].reshape(nx, n)
            # 1. Calculate the mean for each time series of length n
            # and create a mean-adjusted series
            y = x - np.mean(x, axis=1).reshape(-1, 1)
            # 2. Calculate the cumulative deviate series z
            z = np.cumsum(y, axis=1)
            # 3. Calculate the range r and the standard deviation s
            r = np.max(z, axis=1) - np.min(z, axis=1)
            s = np.std(x, axis=1)
            if len(s[s != 0]) > 0:
                # 4. Calculate the rescaled range r/s
                # and average over all the partial time series of length n
                rs = np.mean(r[s != 0] / s[s != 0])
                rs_n.append(rs)

        if len(rs_n) >= 3:
            rs_n = np.array(rs_n)
            hurst, _, _, _, _ = stats.linregress(np.log2(ns), np.log2(rs_n))
        else:
            warnings.warn(
                "Linear regression can not be performed with < 3 points. "
                "The displacement time series might be too short or has too many parts with zero variance. "
                "The Hurst coefficient will be nan."
            )
            hurst = np.nan

        return hurst

    def displacement_autocorrelation(self) -> np.ndarray:
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

    def velocity_autocorrelation(self) -> np.ndarray:
        r"""The normalized velocity autocorrelation is a measure of cell persistence.

        It is calculated with:

        .. math::
            R_{vac}(\tau) = \frac{1}{N-\tau}(\sum_{t=0}^{N-\tau} v \cdot v_{t + \tau}) \times
            \frac{1}{Norm}

        with the normalization factor:

        .. math::
            Norm = \frac{1}{N} \sum_{t=0}^{N-1} \mid v_t \mid^2

        Returns
        -------
        numpy.ndarray
            Normalized velocity autocorrelation for every lag tau
        """
        # Vectorize the trajectory coordinates
        vect = self.path[:: self.fpu, :][1:, :] - self.path[:: self.fpu, :][:-1, :]
        n = len(vect)

        norm = np.mean(np.linalg.norm(vect, axis=1) ** 2)

        # Store correlation coefficients
        acs = []
        # Calculate Autocorrelations
        for tau in range(1, self.autocorr_max_tau + 1):
            coeff = np.sum(vect[: n - tau, :] * vect[tau:, :], axis=1)
            ac = np.mean(coeff) / norm
            acs.append(ac)

        return np.array(acs)


def state_motility_features(
    data: Union[ad.AnnData, pd.DataFrame],
    state_var: str,
    track_id: str = "Metadata_Track",
    time_var: str = "Metadata_Time",
    min_len: Optional[int] = 30,
    max_tau: Optional[int] = 5,
    store_vars: Optional[Union[str, list]] = None,
) -> pd.DataFrame:
    """
    This function calculates motility profiles based on one-dimensional discrete trajectories of single cells.

    Parameters
    ----------
    data : anndata.AnnData, pandas.DataFrame
        Trajectory data
    state_var : str
        Cell state variable. Must be in '.obs' for AnnData objects.
    track_id : str
        Name of track identifiers.
        Must be in '.obs' for AnnData objects.
    time_var : str
        Name of time variable. Must be in '.obs' for AnnData objects.
    min_len : int, optional
        Minimum length of tracks to consider.
    max_tau : int
        Maximal lag for all signed dependence measures
    store_vars : str, list
        Store additional variables for every track

    Returns
    -------
    pd.DataFrame
        Motility measurements for every cell track with given length
    """
    if min_len is not None:
        if max_tau is not None:
            assert (
                max_tau < min_len
            ), f"max_tau must be smaller than min_len ({min_len})"

    if isinstance(store_vars, str):
        store_vars = [store_vars]

    # Get all unique states
    if isinstance(data, pd.DataFrame):
        unique_states = np.sort(np.unique(data[state_var]))
    else:
        unique_states = np.sort(np.unique(data.obs[state_var]))

    # Group data by track
    if isinstance(data, pd.DataFrame):
        grouper = data.groupby(track_id)
    else:
        grouper = data.obs.groupby(track_id)

    output = []

    # Get motility data for every track
    for track, sdata in tqdm(grouper, desc="Motility Profiling"):
        sdata = sdata.sort_values(time_var)
        states = sdata[state_var].to_numpy()
        ts_len = len(sdata)

        accept = True if min_len is None else ts_len >= min_len
        if accept:
            states = sdata[state_var].to_numpy()
            mot = StateMotility(
                states=states, unique_states=unique_states, max_tau=max_tau
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


class StateMotility:
    """Class for motility features of discrete cell state trajectories.

    Parameters
    ----------
    states : numpy.ndarray
        State trajectory
    unique_states : numpy.ndarray, optional
        If not all possibly states occur in this trajectory, the actual states can be defined here.
        Otherwise, it is assumed stat all possible states occur in the trajectory.
    max_tau : int
        Maximal lag for all signed dependence measures

    References
    ----------
    Oriona, Á. L., & Fernández, J. A. V. (2023).
    Analyzing categorical time series with the R package ctsfeatures.
    arXiv preprint arXiv:2304.12332.

    Weiß CH, Göb R (2008). “Measuring serial dependence in categorical time series.” AStA Advances
    in Statistical Analysis, 92, 71–89
    """

    def __init__(
        self,
        states: np.ndarray,
        unique_states: Optional[np.ndarray] = None,
        max_tau: Optional[int] = 5,
    ):
        if not isinstance(states, np.ndarray):
            states = np.array(states).flatten()
        if not np.issubdtype(states.dtype, np.integer):
            states = states.astype(int)
        self.states = states
        self.n = len(states)

        if unique_states is None:
            unique_states = np.sort(np.unique(states))
        else:
            unique_states = np.sort(unique_states)
        if not np.issubdtype(unique_states.dtype, np.integer):
            unique_states = unique_states.astype(int)
        self.unique_states = unique_states
        self.n_states = len(unique_states)

        # assign the max tau for signed dependence calculation
        if max_tau is None:
            max_tau = self.n - 1
        else:
            assert max_tau < self.n, "max_tau must be smaller than path length"
        self.max_tau = max_tau

        # estimated marginal probabilities
        self.marginal_prob = self._marginal_prob()

    def result(self) -> pd.Series:
        """Returns all state motility parameters.

        Returns
        -------
        pandas.Series
            Series with all state motility parameters for the given trajectory path
        """
        (entropy, gini, chebycheff) = self.dispersion()
        result = pd.Series(
            {
                "Length": self.n,
                "Entropy": entropy,
                "Gini": gini,
                "Chebycheff": chebycheff,
                "TransitionProb": self.transition_prob(),
            }
        )

        for i, state in enumerate(self.unique_states):
            result[f"DwellTime_{state}"] = self.marginal_prob[i]

        state_stationary_dist = self.stationary_dist()
        for i, state in enumerate(self.unique_states):
            result[f"StationaryDist_{state}"] = state_stationary_dist[i]

        # Measures of serial dependence
        cohens_kappa = self.cohens_kappa()
        cramers_vi = self.cramers_vi()
        sakoda = self.sakoda()
        gk_tau = self.gk_tau()
        gk_lambda = self.gk_lambda()
        uncertainty_coeff = self.uncertainty_coeff()
        total_correlation = self.total_correlation()
        for i in range(self.max_tau):
            result[f"CohenKappa_{i+1}"] = cohens_kappa[i]
            result[f"CramerVi_{i + 1}"] = cramers_vi[i]
            result[f"Sakoda_{i + 1}"] = sakoda[i]
            result[f"GKTau_{i + 1}"] = gk_tau[i]
            result[f"GKLambda_{i + 1}"] = gk_lambda[i]
            result[f"Uncertainty_{i + 1}"] = uncertainty_coeff[i]
            result[f"TotalCorr_{i + 1}"] = total_correlation[i]

        return result

    def _marginal_prob(self, drop_zero: bool = False) -> np.ndarray:
        """
        Estimated marginal probabilities

        Parameters
        ----------
        drop_zero : bool
            Drop zero probabilities
        """
        unique, counts = np.unique(self.states, return_counts=True)
        state_counts = dict(zip(unique, counts))
        p = np.array(
            [
                state_counts[state] if state in state_counts.keys() else 0
                for state in self.unique_states
            ]
        )
        p = p / np.sum(p)

        if drop_zero:
            return p[p > 0]
        return p

    def _joint_prob(
        self, tau: Optional[int] = 1, drop_zero: bool = False
    ) -> np.ndarray:
        """
        Estimated joints probabilities for lag tau

        Parameters
        ----------
        tau : int
            Time lag
        drop_zero : bool
            Drop zero probabilities
        """
        if drop_zero:
            labels = np.unique(self.states)
        else:
            labels = self.unique_states
        return confusion_matrix(
            self.states[:-tau], self.states[tau:], labels=labels, normalize="all"
        )

    def _transition_matrix(self) -> np.ndarray:
        return confusion_matrix(
            self.states[:-1],
            self.states[1:],
            labels=self.unique_states,
            normalize="true",
        )

    def dispersion(self) -> Tuple[float, float, float]:
        r"""
        Estimated measures of dispersion for categorical time series.
        All have a range of [0, 1] and reach a minimal value of 0 in case
        of one-point-distributions and a maximum value of 1 in case of
        uniform distributions


        Entropy:

        .. math::
            \nu_E = -\frac{1}{ln(m+1)}\sum_j p_j ln(p_j)

        Gini index:

        .. math::
            \nu_G = \frac{m}{m-1}(1 - \sum_j p_j^2)

        Chebycheff dispersion:

        .. math::
            \nu_C = \frac{m}{m-1} (1 - max_j p_j)

        Returns
        -------
        float, float, float
            Entropy, Gini index, Chebycheff dispersion
        """
        p = self.marginal_prob[self.marginal_prob > 0]
        m = self.n_states

        entropy = -(1 / np.log(m)) * np.sum(p * np.log(p))
        gini = (m / (m - 1)) * (1 - np.sum(p**2))
        chebycheff = (m / (m - 1)) * (1 - np.max(p))
        return entropy, gini, chebycheff

    def transition_prob(self) -> float:
        """The transition probability is the number of state transitions divided
        by the number of transitions and non-transitions.

        Returns
        -------
        float
            State transition probability
        """
        transition = np.diff(self.states) == 0
        return 1 - (np.sum(transition) / len(transition))

    def stationary_dist(self) -> np.ndarray:
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

        stationary_dist = target_eigenvect / target_eigenvect.sum()

        return stationary_dist.real

    def cohens_kappa(self) -> np.ndarray:
        r"""Estimated Cohen's Kappa for the categorical time series of states and its lagged version.

        .. math::
            \kappa(k) = \frac{\sum_j (p_{jj}(k) - p_j^2)}{1 - \sum_j p_j^2}

        Returns
        -------
        numpy.ndarray
            Cohen's kappa for all given time lags
        """
        marg_prob = self.marginal_prob

        kappas = []
        for tau in range(1, self.max_tau + 1):
            if len(marg_prob) == 1:
                kappas.append(0)
            else:
                # estimated joint probabilities
                joint_prob = self._joint_prob(tau=tau)
                joint_prob = joint_prob.diagonal()

                kappa = np.sum(joint_prob - (marg_prob**2))
                # Avoid division by zero
                if np.sum(marg_prob**2) == 1:
                    kappa = 0.0
                else:
                    kappa = kappa / (1 - np.sum(marg_prob**2))

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
        """
        m = self.n_states

        # estimated marginal probabilities
        marg_prob = self.marginal_prob[self.marginal_prob > 0]
        marg_prob = np.dot(marg_prob.reshape(-1, 1), marg_prob.reshape(1, -1))

        vis = []

        for tau in range(1, self.max_tau + 1):
            if m == 1:
                vis.append(0)
            else:
                # estimated joint probabilities
                joint_prob = self._joint_prob(tau=tau, drop_zero=True)

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
        """
        m = self.n_states

        # estimated marginal probabilities
        marg_prob = self.marginal_prob[self.marginal_prob > 0]
        marg_prob = np.dot(marg_prob.reshape(-1, 1), marg_prob.reshape(1, -1))

        skds = []

        for tau in range(1, self.max_tau + 1):
            if m == 1:
                skds.append(0)
            else:
                # estimated joint probabilities
                joint_prob = self._joint_prob(tau=tau, drop_zero=True)

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
        """
        marg_prob = self.marginal_prob[self.marginal_prob > 0]

        tau_vals = []
        for tau in range(1, self.max_tau + 1):
            if len(marg_prob) == 1:
                tau_vals.append(0)
            else:
                # estimated joint probabilities
                joint_prob = self._joint_prob(tau=tau, drop_zero=True)

                tau_val = np.sum((joint_prob**2) / marg_prob.reshape(1, -1)) - np.sum(
                    marg_prob**2
                )
                tau_val = tau_val / (1 - np.sum(marg_prob**2))

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
        """
        marg_prob = self.marginal_prob[self.marginal_prob > 0]
        max_mp = np.max(marg_prob)

        lambda_vals = []

        for tau in range(1, self.max_tau + 1):
            if len(marg_prob) == 1:
                lambda_vals.append(0)
            else:
                # estimated joint probabilities
                joint_prob = self._joint_prob(tau=tau, drop_zero=True)

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
        """
        marg_prob = self.marginal_prob[self.marginal_prob > 0]
        marg_prob_matrix = np.dot(marg_prob.reshape(-1, 1), marg_prob.reshape(1, -1))

        coeffs = []

        for tau in range(1, self.max_tau + 1):
            if len(marg_prob) == 1:
                coeffs.append(0)
            else:
                # estimated joint probabilities
                joint_prob = self._joint_prob(tau=tau, drop_zero=True)

                combined = joint_prob / marg_prob_matrix
                combined = np.log(combined, out=combined, where=combined > 0)

                coeff = np.sum(joint_prob * combined)

                coeff = -coeff / np.sum(marg_prob * np.log(marg_prob))
                coeffs.append(coeff)

        return np.array(coeffs)

    def total_correlation(self) -> np.ndarray:
        r"""Estimated Uncertainty coefficient for the categorical time series of states and its lagged version.

        .. math::
            \Psi(k) = \frac{1}{m^2} \sum_{i, j}^{m} \psi_{i, j}(k)^2

        with:

        .. math::
            \psi_{i, j}(k) = \frac{p_{ij}(k)-p_i p_j}{\sqrt{p_i (1-p_i) p_j (1-p_j)}}


        Returns
        -------
        numpy.ndarray
            Total correlation
        """
        marg_prob = self.marginal_prob[self.marginal_prob > 0]
        marg_prob_matrix = np.dot(marg_prob.reshape(-1, 1), marg_prob.reshape(1, -1))

        marg_prob_n = marg_prob * (1 - marg_prob)
        denom = np.sqrt(np.dot(marg_prob_n.reshape(-1, 1), marg_prob_n.reshape(1, -1)))

        tcs = []

        for tau in range(1, self.max_tau + 1):
            if len(marg_prob) == 1:
                tcs.append(0)
            else:
                # estimated joint probabilities
                joint_prob = self._joint_prob(tau=tau, drop_zero=True)

                # Correlation matrix
                corr = (joint_prob - marg_prob_matrix) / denom
                total_corr = np.mean(corr**2)
                tcs.append(total_corr)

        return np.array(tcs)
