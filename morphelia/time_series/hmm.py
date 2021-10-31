import numpy as np
from hmmlearn import hmm
from scipy.spatial.distance import jensenshannon as js
import warnings


def bic(X, k, likelihood_func):
    """Bayesian information criterion.

    Args:
        X (np.ndarray): Data to fit on.
        k (int): Free parameters.
        likelihood_func (function): Likelihood function that takes X as input.
    """
    return k * np.log(len(X)) - 2 * likelihood_func(X)


class HMMSimilarity:
    """Implementation of the distance metric for hidden Markov models by Sahraeian et al., 2011:
    S. M. E. Sahraeian and B. Yoon, "A Novel Low-Complexity HMM Similarity Measure,"
    in IEEE Signal Processing Letters, vol. 18, no. 2, pp. 87-90, Feb. 2011, doi: 10.1109/LSP.2010.2096417

    The algorithm applies the following steps.

        1. Find the hidden markov models lambda_i with lowest bayesian information criterion for timeseries t_i
        2. Compute stationary distributions pi_i for lambda_i
        3. Compute distance D(b_i||b_i') for every pair of states v_i (lambda_i) and v_i' (lambda_i') using
            Jensen-Shannon divergence
        4. Calculate similarity Se(v_i, v_i')
        5. Estimate the state correspondence matrix Q
        6. Compute the HMM similarity measure S(lambda||lambda') based on the normalized gini index as a measure
            for sparsity
    """
    def __init__(self, comp_range=(1, 10)):
        """
        Args:
            comp_range (tuple): Range with number of components to fit model on.
        """
        np.random.seed(42)
        if isinstance(comp_range, list):
            comp_range = tuple(comp_range)
        assert isinstance(comp_range, tuple), f"tuple expected for comp_range, instead got {type(comp_range)}"
        self.comp_range = comp_range
        self.hmm_models = []
        self.s = []

    def fit(self, s):
        """
        Model each time series as a hidden Markov model with bayesian model selection.
        Args:
            s (list of np.ndarray): List of time series data with observation x features.
        """
        hmm_models = []
        for ts in s:
            best_model, _ = self.bic_hmmlearn(ts)
            hmm_models.append(best_model)

        self.hmm_models = hmm_models
        self.s = s
        return None

    def bic_hmmlearn(self, X):
        """Bayesian information criterion from hmmlearn model.

        Args:
            X (np.ndarray): Data to fit on.
        """
        comp_range = self.comp_range
        bic_low = np.inf
        bics = []
        best_model = None
        hmm_model = None

        for n_components in range(comp_range[0], comp_range[1]):
            hmm_model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag')
            hmm_model.fit(X)

            transmat = hmm_model.transmat_
            transmat_rows = np.sum(transmat, axis=1)
            if all(val == 1 for val in transmat_rows):

                n_features = hmm_model.n_features
                free_parameters = 2 * (n_components * n_features) + n_components * (n_components - 1) + (
                            n_components - 1)

                bic_score = bic(X, free_parameters, hmm_model.score)
                bics.append(bic_score)

                if bic_score < bic_low:
                    bic_low = bic_score
                    best_model = hmm_model

        if best_model is not None:
            return best_model, bics
        else:
            warnings.warn("No HMM with BIC < inf")
            # take last model instead
            return hmm_model, bics

    def similarity(self):
        """Similarity between all models"""
        n = len(self.hmm_models)
        sim_matrix = np.zeros((n, n))
        n_ix, m_ix = np.triu_indices(n)

        for i, j in zip(n_ix, m_ix):
            sim_matrix[i, j] = HMM_similarity(self.hmm_models[i], self.hmm_models[j],
                                              self.s[i], self.s[j])

        # make symmetric
        lower_ix = np.tril_indices(n, -1)
        sim_matrix[lower_ix] = sim_matrix.T[lower_ix]

        return sim_matrix


def kl_multi_div(params_1, params_2, epsilon=1e-5):
    """Kullback-Leibler divergence for multivariate gaussians.

    The divergence is defined as:

    D( (mu_1, eps_1) || (mu_2, eps_2) )
        = .5 * ( Spur(eps_2**(-1) * eps_1) + (mu_2 - mu_1).T * eps_2**(-1)
            * (mu_2 - mu_1) - k + ln(det(eps_2) / det(eps_1)))

    with:
        mu_1, mu_2 as means
        eps_1, eps_2 as covariance matrices
        k as number of dimensions

    Args:
        params_1 (np.ndarray, np.ndarray): mu_1, eps_1
        params_2 (np.ndarray, np.ndarray): mu_2, eps_2
        epsilon (float): Avoid division by zero

    Returns:
        float: KL divergence
    """
    mu_1, e_1 = params_1
    mu_2, e_2 = params_2
    mu_diff = mu_2 - mu_1
    e_2_inv = np.linalg.inv(e_2)
    k = mu_1.shape[0]

    tr_term = np.trace(e_2_inv @ e_1)
    det_term = np.log(np.linalg.det(e_2) / np.linalg.det(e_1))
    quad_term = mu_diff.T @ e_2_inv @ mu_diff

    kl = .5 * (tr_term + det_term + quad_term - k)
    return kl + epsilon


def js_multi_div(params_1, params_2):
    """Jensen-Shannon divergence for multivariate gaussians.

    The divergence is defined as:

    D = .5 * Dkl(P || M) + .5 * Dkl(Q || M)
    where P and Q are two probability distributions and M is the average of the two distributions:
    M = .5 * (P + Q)

    Returns:
        float: JS divergence
    """
    mu_1, e_1 = params_1
    mu_2, e_2 = params_2
    M_mu = .5 * (mu_1 + mu_2)
    M_e = .5 * (e_1 + e_2)
    return (.5 * kl_multi_div(params_1, (M_mu, M_e))) + (.5 * kl_multi_div(params_2, (M_mu, M_e)))


def gini_norm(x):
    """Normalized Gini index.

    The normalized gini index is defined as:

    H(u) = (N / (N-1)) - 2 * sum((u(k) / ||u||l * ((N - k + .5) / N)

    where ||u||l is the l1 norm of u and u(k) is the kth smallest element of u.

    Returns:
        float: Gini index
    """
    N = len(x)
    l1 = np.linalg.norm(x, ord=1)
    k = np.array(range(1, N+1))
    x = np.sort(x)
    div = max((N - 1), 1)
    end_term = ((N - k + .5) / div)
    if l1 == 0:
        return 0.

    mid_term = x / l1
    gini = (N / div) - 2 * np.sum(mid_term * end_term)
    return gini


def gini(x, epsilon=1e-5):
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad / (np.mean(x) + epsilon)
    # Gini coefficient
    g = 0.5 * rmad
    return g


def HMM_similarity(hmm_1, hmm_2, X_1, X_2, epsilon=1e-5):
    """HMM similarity"""
    n = hmm_1.n_components
    m = hmm_2.n_components
    stationary_1 = hmm_1.get_stationary_distribution()
    stationary_2 = hmm_2.get_stationary_distribution()
    logprob_1 = hmm_1._compute_log_likelihood(X_1)
    logprob_2 = hmm_2._compute_log_likelihood(X_2)

    Se_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            # cov_1, mu_1 = hmm_1.covars_[i], hmm_1.means_[i]
            # cov_2, mu_2 = hmm_2.covars_[j], hmm_2.means_[j]
            Se_matrix[i, j] = 1 / (js(logprob_1[:, i], logprob_2[:, j]) + epsilon)

    Q = np.outer(stationary_1, stationary_2) * Se_matrix
    Q = Q / np.sum(Q)

    gini_rows = np.mean(np.apply_along_axis(lambda x: gini_norm(x), 0, Q))
    gini_cols = np.mean(np.apply_along_axis(lambda x: gini_norm(x), 1, Q))
    similarity = .5 * (gini_rows + gini_cols)
    # print(np.round(Se_matrix), 2)
    # print(np.round(Q, 2))
    # print(f"Sim: {similarity}")
    return similarity


