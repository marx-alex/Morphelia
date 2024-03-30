from typing import Union, Optional, Tuple, List

import numpy as np
import pandas as pd
from numpy.random import SeedSequence, BitGenerator, Generator
from tqdm import tqdm
from hmmlearn.hmm import CategoricalHMM


class MotionSimulation:
    """
    Motion simulation using motion models.
    The following models are implemented:
    - Random Walk
    - Biased Random Walk
    - Lévy flight
    - Fractal Brownian motion
    """

    def __init__(
        self,
        dim: int = 2,
        seed: Optional[
            Union[int, np.ndarray, SeedSequence, BitGenerator, Generator]
        ] = None,
        origin: Optional[Union[int, float, Tuple, List, np.ndarray]] = 0,
        origin_min: int = 0,
        origin_max: int = 2048,
    ):
        """
        Parameters
        ----------
        dim : int
            Number of dimensions
        seed : None, int, array_like, SeedSequence, BitGenerator, Generator
            Seed for random processes
        origin : int, float, tuple, list, array_like, optional
            Origin of the trajectories.
            If None, the origin is sampled.
            If integer or float, the number is repeated to fit the dimensions.
        origin_min : int
            If the origin is sampled, this minimum is used
        origin_max : int
            If the origin is sampled, this maximum is used
        """
        assert dim > 0, f"dim must be > 0, instead got {dim}"
        self.dim = dim
        self.rng = np.random.default_rng(seed=seed)
        self.models = {
            "rw": self.random_walk,
            "brw": self.biased_random_walk,
            "levy": self.levy_flight,
            "fbm": self.fractal_brownian_motion,
        }
        if origin is None:
            origin = self.rng.integers(low=origin_min, high=origin_max, size=(1, dim))
        else:
            if isinstance(origin, (float, int)):
                origin = np.full(shape=(1, dim), fill_value=origin)
            else:
                origin = np.array(origin).reshape(1, -1)
                assert (
                    origin.shape[-1] == dim
                ), f"origin must have {dim} dimensions, got {len(origin)}"
        self.origin = origin

    def __str__(self):
        message = """
        Motion Simulation Models:
        #########################
        Random Walk             (rw)
        Biased Random Walk      (brw)
        Lévy Flight             (levy)
        Fractal Brownian Motion (fbm)
        """
        return message

    def _draw_turn_angle(self, n: int) -> np.ndarray:
        """
        Draws d - 1 random turn angles with d dimensions for n cells.
        If there is only one dimensions turn angles are either positive
        or negative displacements.

        Parameters
        ----------
        n : int
            Number of angles

        Returns
        -------
        phi : int
            Turn angles
        """
        if self.dim == 1:
            phi = self.rng.choice([-1, 1], size=(n, 1))
        else:
            phi = self.rng.uniform(low=0, high=2 * np.pi, size=(n, self.dim - 1))
        return phi

    def _to_cartesian(self, rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Converts displacements and turn angles into positions on the cell trajectory
        in the Cartesian space.

        Parameters
        ----------
        rho : np.ndarray
            Displacements
        phi : np.ndarray
            Angles

        Returns
        -------
        positions : np.ndarray
            Trajectory positions
        """
        if self.dim > 1:
            # Angular coordinates to cartesian coordinates with d dimensions and n cells
            t = np.pad(
                phi, pad_width=((0, 0), (0, 1)), mode="constant", constant_values=0
            ).reshape(
                -1, 1, self.dim
            )  # [n x 1 x d]
            t = np.repeat(t, self.dim, axis=1)  # [n x d x d]
            t[np.triu(np.ones_like(t), k=1).astype(bool)] = 1

            # Create a mask to access the diagonals
            diag_mask = np.zeros((self.dim, self.dim)).astype(bool)
            np.fill_diagonal(diag_mask, val=1)
            diag_mask = diag_mask.reshape(1, self.dim, self.dim)
            diag_mask = np.repeat(diag_mask, phi.shape[0], axis=0)
            # Cosines of the diagonals
            t[diag_mask] = np.cos(t[diag_mask])
            # Create a mask to access the lower triangle
            tril_mask = np.tril(np.ones_like(t), k=-1).astype(bool)
            # Sinus of the lower triangle
            t[tril_mask] = np.sin(t[tril_mask])
            phi = np.prod(t, axis=2)  # [n x d]

        # Cartesian coordinates
        steps = rho.reshape(-1, 1) * phi

        # Add origin
        steps = np.vstack((self.origin, steps))
        # Calculate positions
        positions = np.cumsum(steps, axis=0)
        return positions

    def random_walk(
        self,
        length: int,
        loc: Union[float, int] = 5,
        scale: Union[float, int] = None,
        **kwargs,
    ):
        """
        Unbiased random walk model.

        The displacements between trajectory cell positions are sampled from a normal distribution.
        For each step, the turn angle is randomly sampled from a uniform distribution over the
        interval [0, 2*pi).

        Parameters
        ----------
        length : int
            length of trajectory
        loc : int
            Mean of the normal distribution
        scale : std, optional
            Standard deviation of the normal distribution.
            The default scale is loc/5.

        Returns
        -------
        positions : np.ndarray
            Trajectory positions
        """
        if scale is None:
            scale = loc / 5

        # Draw a random displacement from a gaussian distribution
        displacement = self.rng.normal(loc=loc, scale=scale, size=length - 1)
        # Draw a random turn angle
        phi = self._draw_turn_angle(n=length - 1)

        return self._to_cartesian(displacement, phi)

    def biased_random_walk(
        self,
        length: int,
        loc: Union[float, int] = 5,
        scale: Union[float, int] = None,
        bias_prob: Union[float, int] = 0.75,
        **kwargs,
    ) -> np.ndarray:
        """
        Biased random walk model.

        The displacements between trajectory cell positions are sampled from a normal distribution.
        For each step, the turn angle is randomly sampled from a uniform distribution over the
        interval [0, 2*pi). 'bias_prob'% of the cells are replaced by a random biased turn angle.

        Parameters
        ----------
        length : int
            length of trajectory
        loc : int
            Mean of the normal distribution
        scale : std, optional
            Standard deviation of the normal distribution.
            The default scale is loc/5.
        bias_prob : float
            Probability that the cell moves in one particular direction.

        Returns
        -------
        positions : np.ndarray
            Trajectory positions
        """
        assert (
            0 <= bias_prob <= 1
        ), f"bias_prob must be 0 <= bias_prob <= 1, instead got {bias_prob}"

        if scale is None:
            scale = loc / 5

        # Step in this direction bias_param% of the time
        bias_phi = self._draw_turn_angle(n=1)

        # Draw a random displacement from a gaussian distribution
        displacement = self.rng.normal(loc, scale, size=length - 1)
        # Draw a random turn angle
        phi = self._draw_turn_angle(n=length - 1)
        # Change turn angle to bias_phi bias_param% of the time
        random_bias = self.rng.random(size=length - 1)
        phi[random_bias <= bias_prob, ...] = bias_phi

        return self._to_cartesian(displacement, phi)

    def levy_flight(
        self,
        length: int,
        loc: Union[float, int] = 5,
        alpha: Union[float, int] = 2,
        **kwargs,
    ) -> np.ndarray:
        r"""
        Lévy flight model.

        Lévy flights are characterized by a distribution function:

        .. math::
            P(x) \sim x^{-\alpha}

        The displacements between trajectory cell positions are sampled from the
        power-law distribution.
        For each step, the turn angle is randomly sampled from a uniform distribution over the
        interval [0, 2*pi).

        Parameters
        ----------
        length : int
            length of trajectory
        loc : int
            Median of the distribution
        alpha : float, int
            Alpha value of the power-law distribution with 1 < alpha <= 3.

        Returns
        -------
        positions : np.ndarray
            Trajectory positions
        """
        assert 1 < alpha <= 3, f"alpha must be 1 < alpha <= 3, instead got {alpha}"

        # Draw a random displacement from a power law distribution
        x_min = loc / (2 ** (1 / (alpha - 1)))
        displacement = x_min / (1 - self.rng.random(size=length - 1)) ** (1 / alpha)
        # Draw a random turn angle
        phi = self._draw_turn_angle(n=length - 1)

        return self._to_cartesian(displacement, phi)

    def _create_fbm(
        self,
        n: int,
        hurst: float = 0.9,
        loc: Union[float, int] = 5,
        scale: Union[float, int] = 1,
    ) -> np.ndarray:
        # Create covariance matrix for fBm
        k = np.abs(np.arange(n).reshape(-1, 1) - np.arange(n).reshape(1, -1))
        gamma = 0.5 * (
            np.power(k + 1, 2 * hurst)
            + np.power(np.abs(k - 1), 2 * hurst)
            - 2 * np.power(k, 2 * hurst)
        )
        # Cholesky decomposition
        sigma = np.linalg.cholesky(gamma)
        # Generate a random field
        random_vector = self.rng.normal(size=n)
        # Apply Cholesky factorization
        fractal_noise = sigma @ random_vector
        fractal_noise = (fractal_noise * scale) + loc
        return fractal_noise

    def fractal_brownian_motion(
        self,
        length: int,
        loc: Union[float, int] = 5,
        scale: Union[float, int] = None,
        hurst: float = 0.9,
        **kwargs,
    ) -> np.ndarray:
        """
        Fractal Brownian motion model.

        The displacements between trajectory cell positions are sampled from a normal distribution
        with a covariance matrix defined by the Hurst parameter H.
        For each step, the turn angle is randomly sampled from a uniform distribution over the
        interval [0, 2*pi).

        H is a measure of autocorrelation with:

        H = 1/2 : Brownian motion
        H > 1/2 : positive autocorrelation
        H < 1/2 : negative autocorrelation

        Parameters
        ----------
        length : int
            length of trajectory
        loc : int
            Mean of the distribution
        scale : std, optional
            Standard deviation of the distribution.
            The default scale is loc/5.
        hurst : float
            Hurst coefficient. Must be 0 < hurst < 1.

        Returns
        -------
        positions : np.ndarray
            Trajectory positions
        """
        assert (
            0 < hurst < 1
        ), f"Hurst coefficient H must be 0 < H < 1, instead got {hurst}"

        if scale is None:
            scale = loc / 5

        # Draw a random displacement from a fractal gaussian distribution
        displacement = self._create_fbm(n=length - 1, hurst=hurst, loc=loc, scale=scale)

        # Draw a random turn angle
        phi = self._draw_turn_angle(n=length - 1)

        return self._to_cartesian(displacement, phi)

    def simulate_motions(
        self, n: int, model: str = "rw", length: int = 50, fpu=1, **kwargs
    ) -> pd.DataFrame:
        """
        Simulate the trajectory of n cells using a motion model.

        Parameters
        ----------
        n : int
            Number of cells
        model : str
            Motion model. Must be 'rw', 'brw', 'levy' or 'fbm'.
        length : int
            Length of single cell trajectories
        fpu : int
            Frames per unit
        **kwargs
            Keyword arguments passed to the motion model function

        Returns
        -------
        pd.DataFrame
            Result
        """
        assert (
            model in self.models.keys()
        ), f"model must be one of {self.models.keys()}, got {model}"

        m = self.models[model]
        tracks = []

        for ix in tqdm(
            range(1, n + 1),
            desc=f"Simulation (N: {n}, Model: {model}, Length: {length}, Dim: {self.dim})",
        ):
            track = m(length=length, **kwargs)
            track_id = np.full(shape=(length, 1), fill_value=ix)
            time = np.linspace(0, (length / fpu) - (1 / fpu), length).reshape(-1, 1)
            track = np.hstack((track_id, time, track))
            tracks.append(track)

        header = ["Track", "Time"] + [f"X{i+1}" for i in range(self.dim)]
        tracks = np.vstack(tracks)
        tracks = pd.DataFrame(tracks, columns=header)

        return tracks


class MarkovChainSimulation:
    """
    Simulation of categorical time series with Markov Chains
    """

    def __init__(
        self,
        n_states: int,
        t: Optional[np.ndarray] = None,
        start_state: Optional[int] = None,
        seed: Optional[
            Union[int, np.ndarray, SeedSequence, BitGenerator, Generator]
        ] = None,
    ):
        """
        Parameters
        ----------
        n_states : int
            Number of states
        t : np.ndarray, optional
            Transition matrix. If None, a transition matrix with p_ij = 1 / n_states is initialized.
        start_state : int, optional
            State to begin with. If None, the process starts with a random state.
        seed : int, array_like, SeedSequence, BitGenerator, Generator, optional
            Seed for reproducibility
        """
        self.n_states = n_states
        self.rng = np.random.default_rng(seed=seed)
        self.states = np.arange(n_states)

        # Assign start state
        if start_state is not None:
            assert (
                start_state < n_states
            ), f"start_state must be < n_states ({n_states}), got {start_state}"
        self.start_state = start_state

        # Assign transition matrix
        if t is None:
            t = np.ones(shape=(n_states, n_states))
            t = t / t.sum(axis=1).reshape(-1, 1)
        self._t = t

    def markov_chain(self, length: int) -> np.ndarray:
        """
        Simulation of a categorical time series with a given transitions matrix.

        Parameters
        ----------
        length : int
            Length of time series

        Returns
        -------
        np.ndarray
            Categorical time series
        """
        if self.start_state is None:
            current_state = self.rng.choice(self.states)
        else:
            current_state = self.start_state

        path = [current_state]

        for _ in range(length - 1):
            current_state = self.rng.choice(self.states, p=self._t[current_state, :])
            path.append(current_state)

        return np.array(path)

    def simulate_motions(self, n: int, length: int = 50, fpu: int = 1) -> pd.DataFrame:
        """
        Simulate categorical time series of n cells using markov chains.

        Parameters
        ----------
        n : int
            Number of cells
        length : int
            Length of single cell trajectories
        fpu : int
            Frames per unit

        Returns
        -------
        pd.DataFrame
            Result
        """
        tracks = []

        for ix in tqdm(
            range(1, n + 1), desc=f"MC simulation (N: {n}, Length: {length})"
        ):
            track = self.markov_chain(length=length)
            track_id = np.full(shape=(length, 1), fill_value=ix)
            time = np.linspace(0, (length / fpu) - (1 / fpu), length).reshape(-1, 1)
            track = np.hstack((track_id, time, track.reshape(-1, 1)))
            tracks.append(track)

        header = ["Track", "Time", "State"]
        tracks = np.vstack(tracks)
        tracks = pd.DataFrame(tracks, columns=header)

        return tracks


class NDARMASimulation:
    r"""
    New Discrete ARMA process.

    As with ARMA models, DARMA models incorporate both noise (moving average term)
    and past values of the time series (autoregressive term).

    .. math::
        X_t = \sum_{i=1}^p a_t^{(i)} X_{t-i} + \sum_{j=0}^q b_t^{(j)} \epsilon_{t-j}

    where the parameters are i.i.d. multinomial random vectors:

    .. math::
        P_t := [a_t^{1}, ..., a_t^{p}, b_t^{0}, ..., b_t^{q}] \sim Mult(1; \phi_1, ..., \phi_p, \varphi_0, ..., \varphi_q)

    and

    .. math::
        P(\epsilon_t = i) = \pi_i

    where Pi is the marginal distribution of X.

    """

    def __init__(
        self,
        n_states: int,
        coeffs: Optional[np.ndarray] = None,
        pi: Optional[np.ndarray] = None,
        ar_order: int = 2,
        ma_order: int = 0,
        seed: Optional[
            Union[int, np.ndarray, SeedSequence, BitGenerator, Generator]
        ] = None,
    ):
        """
        Parameters
        ----------
        n_states : int
            Number of states
        coeffs : array_like, optional
            Probabilities of the multinomial distribution. Must sum up to 1.
            Number of coeffs must be equal to `ar_order` + `ma_order` + 1.
        pi : array_like, optional
            Marginal state distribution
        ar_order : int
            Order of the autoregressive term
        ma_order : int
            Order of the moving average term
        seed : int, array_like, SeedSequence, BitGenerator, Generator, optional
            Seed for reproducibility
        """
        self.n_states = n_states
        self.rng = np.random.default_rng(seed=seed)
        self.states = np.arange(n_states)

        # Assign coeffs and pi
        if pi is not None:
            assert (
                len(pi) == self.n_states
            ), "Length of `pi` must be equal to number of states"
        else:
            pi = np.ones(shape=n_states)
            pi = pi / pi.sum()

        if coeffs is not None:
            assert (ar_order + ma_order + 1) == len(
                coeffs
            ), "`ar_order` + `ma_order` + 1 must be equal to number of coeffs"
        else:
            coeffs = np.ones(shape=ar_order + ma_order + 1)
            coeffs = coeffs / 1

        self.pi = pi
        self.ar_order = ar_order
        self.ma_order = ma_order
        self.coeffs = coeffs

    def ndarma(self, length: int) -> np.ndarray:
        """
        Simulation of a categorical time series a NDARMA model.

        Parameters
        ----------
        length : int
            Length of time series

        Returns
        -------
        np.ndarray
            Categorical time series
        """
        # Prepend path with n = ar_order states
        path = self.rng.choice(self.states, p=self.pi, size=self.ar_order)

        # For every state in path
        for _ in range(length):
            # All parameters follow a multinomial distribution ~ Mult(1;P)
            params = self.rng.multinomial(1, self.coeffs)
            a = params[: self.ar_order]
            b = params[-(self.ma_order + 1) :]
            epsilon = self.rng.choice(self.states, p=self.pi, size=self.ma_order + 1)

            ar = np.sum(a * path[-self.ar_order :][::-1])  # Autoregression
            ma = np.sum(b * epsilon)  # Moving average
            path = np.concatenate((path, [ar + ma]))

        path = path[self.ar_order :]

        return np.array(path)

    def simulate_motions(self, n: int, length: int = 50, fpu: int = 1) -> pd.DataFrame:
        """
        Simulate categorical time series of n cells using NDARMA model.

        Parameters
        ----------
        n : int
            Number of cells
        length : int
            Length of single cell trajectories
        fpu : int
            Frames per unit

        Returns
        -------
        pd.DataFrame
            Result
        """
        tracks = []

        for ix in tqdm(
            range(1, n + 1), desc=f"NDARMA simulation (N: {n}, Length: {length})"
        ):
            track = self.ndarma(length=length)
            track_id = np.full(shape=(length, 1), fill_value=ix)
            time = np.linspace(0, (length / fpu) - (1 / fpu), length).reshape(-1, 1)
            track = np.hstack((track_id, time, track.reshape(-1, 1)))
            tracks.append(track)

        header = ["Track", "Time", "State"]
        tracks = np.vstack(tracks)
        tracks = pd.DataFrame(tracks, columns=header)

        return tracks


class HMMSimulation:
    """
    Simulation of categorical time series with Categorical Hidden Markov Models.
    This is a wrapper for the hmmlearn.hmm.CategoricalHMM Class.
    """

    def __init__(
        self,
        n_states: int,
        t: Optional[np.ndarray] = None,
        e: Optional[np.ndarray] = None,
        start_prob: Optional[np.ndarray] = None,
        seed: Optional[
            Union[int, np.ndarray, SeedSequence, BitGenerator, Generator]
        ] = None,
    ):
        """
        Parameters
        ----------
        n_states : int
            Number of states
        t : np.ndarray, optional
            Transition matrix. If None, a transition matrix with p_ij = 1 / n_states is initialized.
        e : np.ndarray, optional
            Emission matrix. If None, a transition matrix with p_ij = 1 / n_states is initialized.
        start_prob : np.ndarray, optional
            Prior distribution. If None, a vector with p_i = 1 / n_states is initialized.
        seed : int, array_like, SeedSequence, BitGenerator, Generator, optional
            Seed for reproducibility
        """
        self.n_states = n_states
        self.states = np.arange(n_states)

        # Assign transition matrix
        if t is None:
            t = np.ones(shape=(n_states, n_states))
            t = t / t.sum(axis=1).reshape(-1, 1)
        self._t = t

        # Assign emission matrix
        if e is None:
            e = np.ones(shape=(n_states, n_states))
            e = e / e.sum(axis=1).reshape(-1, 1)
        self._e = e

        # Assign start_prob
        if start_prob is None:
            start_prob = np.ones(shape=n_states)
            start_prob = start_prob / start_prob.sum()
        self._start_prob = start_prob

        self.rng = np.random.default_rng(seed=seed)
        self.model = CategoricalHMM(n_components=n_states)
        self.model.startprob_ = self._start_prob
        self.model.transmat_ = self._t
        self.model.emissionprob_ = self._e

    def hmm(self, length: int) -> np.ndarray:
        """
        Simulation of a categorical time series with a Categorical Hidden Markov Model.

        Parameters
        ----------
        length : int
            Length of time series

        Returns
        -------
        np.ndarray
            Categorical time series
        """
        rs = self.rng.integers(low=0, high=2**32)
        path, _ = self.model.sample(length, random_state=rs)
        return path.flatten()

    def simulate_motions(self, n: int, length: int = 50, fpu: int = 1) -> pd.DataFrame:
        """
        Simulate categorical time series of n cells using categorical hidden markov models.

        Parameters
        ----------
        n : int
            Number of cells
        length : int
            Length of single cell trajectories
        fpu : int
            Frames per unit

        Returns
        -------
        pd.DataFrame
            Result
        """
        tracks = []

        for ix in tqdm(
            range(1, n + 1), desc=f"HMM simulation (N: {n}, Length: {length})"
        ):
            track = self.hmm(length=length)
            track_id = np.full(shape=(length, 1), fill_value=ix)
            time = np.linspace(0, (length / fpu) - (1 / fpu), length).reshape(-1, 1)
            track = np.hstack((track_id, time, track.reshape(-1, 1)))
            tracks.append(track)

        header = ["Track", "Time", "State"]
        tracks = np.vstack(tracks)
        tracks = pd.DataFrame(tracks, columns=header)

        return tracks
