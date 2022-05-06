import numpy as np
from hmmlearn import hmm

import warnings
from typing import Callable, Tuple, List


def bic(X: np.ndarray, k: int, likelihood_func: Callable) -> float:
    """Bayesian information criterion.

    Parameters
    ----------
    X : numpy.ndarray
        Data to fit on.
    k : int
        Free parameters
    likelihood_func : callable
        Likelihood function that takes X as input

    Returns
    -------
    float
        Bayesian information criterion
    """
    return k * np.log(len(X)) - 2 * likelihood_func(X)


def aic(X: np.ndarray, k: int, likelihood_func: Callable) -> float:
    """Akaike information criterion.

    Parameters
    ----------
    X : numpy.ndarray
        Data to fit on
    k : int
        Free parameters
    likelihood_func : callable
        Log likelihood function that takes X as input

    Returns
    -------
    float
        Akaike information criterion
    """
    return 2 * k - 2 * likelihood_func(X)


class HMMSimilarity:
    """Implementation of the distance metric for hidden Markov models.

    The algorithm works through the following steps:

        1. Find the hidden markov models lambda_i with lowest bayesian information criterion for timeseries t_i
        2. Compute stationary distributions pi_i for lambda_i
        3. Compute distance D(b_i||b_i') for every pair of states v_i (lambda_i) and v_i' (lambda_i') using
            Jensen-Shannon divergence
        4. Calculate similarity Se(v_i, v_i')
        5. Estimate the state correspondence matrix Q
        6. Compute the HMM similarity measure S(lambda||lambda') based on the normalized gini index as a measure
            for sparsity

    Parameters
    ----------
    state_range : tuple of int
        Calculate a range of states and choose best state by using a criterion
    criterion : str
        Use Bayesian (`bic`) or Akaike (`aic`) information criterion
    seed : int
        Seed for reproduction

    References
    ----------
    .. [1] S. M. E. Sahraeian and B. Yoon, "A Novel Low-Complexity HMM Similarity Measure"
       in IEEE Signal Processing Letters, vol. 18, no. 2, pp. 87-90, Feb. 2011, doi: 10.1109/LSP.2010.2096417
    """

    def __init__(
        self,
        state_range: Tuple[int, int] = (2, 10),
        criterion: str = "bic",
        seed: int = 42,
    ):
        np.random.seed(seed)
        self.state_range = state_range
        self.hmm_models = []
        self.s = []

        avail_model_select = ["aic", "bic"]
        criterion = criterion.lower()
        assert criterion in avail_model_select, (
            f"model selection criterion must be one of: {avail_model_select}, "
            f"instead got: {criterion}"
        )
        self.criterion = criterion

    def fit(self, s: List[np.ndarray]) -> None:
        """Fit Markov Models.

        Model each time series as a hidden Markov model and
        select number of components based on a criterion.

        Parameters
        ----------
        s : list of numpy.ndarray
            List of time series data, each of shape `[observations, features]`
        """
        hmm_models = []
        for ts in s:
            best_model = self.criterion_select(ts)
            hmm_models.append(best_model)

        self.hmm_models = hmm_models
        self.s = s
        return None

    def criterion_select(self, X: np.ndarray) -> hmm.GaussianHMM:
        """Estimate number of components based on criterion.

        Parameters
        ----------
        X : numpy.ndarray
            Data to fit

        Returns
        -------
        hmm.GaussianHMM
            Gaussian Hidden Markov Model
        """
        criterion_ceil = np.inf
        best_model = None
        hmm_model = None

        for n_components in range(self.state_range[0], self.state_range[1]):
            hmm_model = hmm.GaussianHMM(
                n_components=n_components, covariance_type="diag"
            )
            hmm_model.fit(X)

            transmat = hmm_model.transmat_
            transmat_rows = np.sum(transmat, axis=1)

            if not any(val == 0 for val in transmat_rows):

                n_features = hmm_model.n_features
                free_parameters = (
                    2 * (n_components * n_features)
                    + n_components * (n_components - 1)
                    + (n_components - 1)
                )

                if self.criterion == "bic":
                    criterion_score = bic(X, free_parameters, hmm_model.score)
                elif self.criterion == "aic":
                    criterion_score = aic(X, free_parameters, hmm_model.score)
                else:
                    raise ValueError(f"criterion method unknown: {self.criterion}")

                if criterion_score < criterion_ceil:
                    best_model = hmm_model
                    criterion_ceil = criterion_score

        if best_model is not None:
            return best_model
        else:
            warnings.warn("No HMM with criterion < inf")
            # take last model instead
            return hmm_model

    def similarity(self) -> np.ndarray:
        """Calculate similarity between all HMMs.

        Returns
        -------
        numpy.ndarray
            Similarity matrix between all models
        """
        n = len(self.hmm_models)
        sim_matrix = np.zeros((n, n))
        n_ix, m_ix = np.triu_indices(n)

        for i, j in zip(n_ix, m_ix):
            sim_matrix[i, j] = HMM_similarity(self.hmm_models[i], self.hmm_models[j])

        # make symmetric
        lower_ix = np.tril_indices(n, -1)
        sim_matrix[lower_ix] = sim_matrix.T[lower_ix]

        return sim_matrix


def fisher_rao(
    params_1: Tuple[np.ndarray, np.ndarray], params_2: Tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Univariate Fisher-Rao distance.

    Parameters
    ----------
    params_1 : tuple of numpy.ndarray
        mean and standard deviation of first distribution
    params_2 : tuple of numpy.ndarray
        mean and standard deviation of second distribution

    Returns
    -------
    numpy.ndarray
        Fisher-Rao distance
    """
    mu_1, sig_1 = params_1
    mu_2, sig_2 = params_2
    l1 = (mu_1 - mu_2) ** 2 + 2 * (sig_1 - sig_2) ** 2
    l2 = (mu_1 + mu_2) ** 2 + 2 * (sig_1 + sig_2) ** 2
    f = np.sqrt(l1 * l2)
    dist = np.sqrt(2) * (
        np.log(f + (mu_1 - mu_2) ** 2 + 2 * (sig_1 ** 2 + sig_2 ** 2))
        - np.log(4 * sig_1 * sig_2)
    )
    return dist


def multivariate_fisher_rao(
    params_1: Tuple[np.ndarray, np.ndarray], params_2: Tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Multivariate Fisher-Rao distance with given diagonal covariance matrix and means.

    We consider the parameters :math:`\\theta = (\\mu_1, \\sigma_1, \\mu_2, \\sigma_2, ... \\mu_n, \\sigma_n)`.
    The distance is given by:

    .. math::

        d(\\theta_1, \\theta_2) = \\sqrt{ \\sum{d((\\mu_1, \\sigma_1), (\\mu_2, \\sigma_2))^2}}

    where d is the Fisher-Rao distance in the univariate case.

    Parameters
    ----------
    params_1 : tuple of numpy.ndarray
        mean and standard deviation of first distribution
    params_2 : tuple of numpy.ndarray
        mean and standard deviation of second distribution

    Returns
    -------
    numpy.ndarray
        Fisher-Rao distance
    """
    mu_1, sig_1 = params_1
    mu_2, sig_2 = params_2
    m, n = len(mu_1), len(mu_2)
    assert m == n, f"two distributions with different dimensions: {m} != {n}"
    sig_1, sig_2 = sig_1.diagonal(), sig_2.diagonal()
    dist = [fisher_rao((mu_1[i], sig_1[i]), (mu_2[i], sig_2[i])) ** 2 for i in range(m)]
    return np.sqrt(np.sum(dist))


def kl_multi_div(
    params_1: Tuple[np.ndarray, np.ndarray], params_2: Tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Kullback-Leibler divergence for multivariate diagonal-covariance gaussians.

    The divergence is defined as:

    .. math::

        D( (\\mu_1, \\sigma_1) || (\\mu_2, \\sigma_2) )
        = .5 * ( Spur(\\sigma_2^{-1} * \\sigma_1) + (\\mu_2 - \\mu_1).T * \\sigma_2^{-1}
            * (\\mu_2 - \\mu_1) - k + ln(det(\\sigma_2) / det(\\sigma_1)))

    with:
    :math:`\\mu_1, \\mu_2` as means,
    :math:`\\sigma_1, \\sigma_2` as diagonal covariance matrices and
    :math:`k` as number of dimensions

    Parameters
    ----------
    params_1 : tuple of numpy.ndarray
        mean and standard deviation of first distribution
    params_2 : tuple of numpy.ndarray
        mean and standard deviation of second distribution

    Returns
    -------
    numpy.ndarray
        Kullback-Leibler divergence
    """
    mu_1, sig_1 = params_1
    mu_2, sig_2 = params_2
    mu_diff = mu_2 - mu_1
    sig_2_inv = np.linalg.inv(sig_2)
    k = mu_1.shape[0]

    tr_term = np.trace(sig_2_inv @ sig_1)
    det_term = np.log(np.linalg.det(sig_2) / np.linalg.det(sig_1))
    quad_term = mu_diff.T @ sig_2_inv @ mu_diff

    kl = 0.5 * (tr_term + det_term + quad_term - k)
    return kl


def js_multi_div(
    params_1: Tuple[np.ndarray, np.ndarray], params_2: Tuple[np.ndarray, np.ndarray]
) -> float:
    """Jensen-Shannon divergence for multivariate gaussians.

    The divergence is defined as:

    .. math::

        D = .5 * Dkl(P || M) + .5 * Dkl(Q || M)

    where P and Q are two probability distributions and M is the average of the two distributions:

    .. math::

        M = .5 * (P + Q)

    Parameters
    ----------
    params_1 : tuple of numpy.ndarray
        mean and standard deviation of first distribution
    params_2 : tuple of numpy.ndarray
        mean and standard deviation of second distribution

    Returns
    -------
    float
        Jensen-Shannon divergence
    """
    mu_1, sig_1 = params_1
    mu_2, sig_2 = params_2
    M_mu = 0.5 * (mu_1 + mu_2)
    M_sig = 0.5 * (sig_1 + sig_2)
    return (0.5 * kl_multi_div(params_1, (M_mu, M_sig))) + (
        0.5 * kl_multi_div(params_2, (M_mu, M_sig))
    )


def gini(x: np.ndarray, epsilon: float = 1e-8) -> float:
    """Gini coefficient.

    Parameters
    ----------
    x : numpy.ndarray
        Array
    epsilon : float
        Avoid zero-division

    Returns
    -------
    float
        Gini coefficient
    """
    # mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # relative mean absolute difference
    rmad = mad / (np.mean(x) + epsilon)
    # gini coefficient
    g = 0.5 * rmad
    return g


def HMM_similarity(
    hmm_1: hmm.GaussianHMM,
    hmm_2: hmm.GaussianHMM,
    method: str = "js",
    epsilon: float = 0.01,
) -> float:
    """Simmilarity between Hidden Markov Models.

    Parameters
    ----------
    hmm_1 : hmm.GaussianHMM
        First HMM
    hmm_2 : hmm.GaussianHMM
        Second HMM
    method : str
        Calculate distance with Jensen-Shannon (`js`) or Fisher-Rao (`fisher`) distance
    epsilon : float
        Avoid zero-division

    Returns
    -------
    float
        Similarity measure between two Hidden Markov Models
    """
    n = hmm_1.n_components
    m = hmm_2.n_components
    stationary_1 = hmm_1.get_stationary_distribution()
    stationary_2 = hmm_2.get_stationary_distribution()

    avail_methods = ["js", "fisher"]
    method = method.lower()
    assert method in avail_methods, f"method not in available methods: {avail_methods}"
    if method == "js":
        dist_func = js_multi_div
    elif method == "fisher":
        dist_func = multivariate_fisher_rao
    else:
        raise ValueError(f"distance function unknown: {method}")

    Se_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            params_1 = hmm_1.means_[i], hmm_1.covars_[i]
            params_2 = hmm_2.means_[j], hmm_2.covars_[j]
            Se_matrix[i, j] = 1 / (dist_func(params_1, params_2) + epsilon)

    Q = np.outer(stationary_1, stationary_2) * Se_matrix
    Q = Q / np.sum(Q)

    gini_rows = np.mean(np.apply_along_axis(lambda x: n * gini(x) / max(n, 1), 0, Q))
    gini_cols = np.mean(np.apply_along_axis(lambda x: m * gini(x) / max(m, 1), 1, Q))
    similarity = 0.5 * (gini_rows + gini_cols)

    return similarity
