import numpy as np
from hmmlearn import hmm
import warnings


def bic(X, k, likelihood_func):
    """Bayesian information criterion.

    Args:
        X (np.ndarray): Data to fit on.
        k (int): Free parameters.
        likelihood_func (function): Likelihood function that takes X as input.
    """
    return k * np.log(len(X)) - 2 * likelihood_func(X)


def aic(X, k, likelihood_func):
    """Akaike information criterion.

    Args:
        X (np.ndarray): Data to fit on.
        k (int): Free parameters.
        likelihood_func (function): Log likelihood function that takes X as input.
    """
    return 2 * k - 2 * likelihood_func(X)


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
    def __init__(self, state_range=(2, 10), criterion='bic', seed=42):
        """
        Args:
            state_range (tuple): Calculate states in specified range.
            criterion (str): AIC or BIC
            seed (int)
        """
        np.random.seed(seed)
        self.state_range = state_range
        self.hmm_models = []
        self.s = []

        avail_model_select = ['aic', 'bic']
        criterion = criterion.lower()
        assert criterion in avail_model_select, (
            f"model selection criterion must be one of: {avail_model_select}, "
            f"instead got: {criterion}"
        )
        self.criterion = criterion

    def fit(self, s):
        """
        Model each time series as a hidden Markov model with bayesian model selection.

        Args:
            s (list of np.ndarray): List of time series data with observation x features.
        """
        hmm_models = []
        for ts in s:
            best_model = self.criterion_select(ts)
            hmm_models.append(best_model)

        self.hmm_models = hmm_models
        self.s = s
        return None

    def criterion_select(self, X):
        """Select best hmmlearn model.

        Args:
            X (np.ndarray): Data to fit on.
        """
        criterion_ceil = np.inf
        best_model = None
        hmm_model = None

        for n_components in range(self.state_range[0], self.state_range[1]):
            hmm_model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag')
            hmm_model.fit(X)

            transmat = hmm_model.transmat_
            transmat_rows = np.sum(transmat, axis=1)

            if not any(val == 0 for val in transmat_rows):

                n_features = hmm_model.n_features
                free_parameters = 2 * (n_components * n_features) + n_components * (n_components - 1) + (
                            n_components - 1)

                if self.criterion == 'bic':
                    criterion_score = bic(X, free_parameters, hmm_model.score)
                elif self.criterion == 'aic':
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

    def similarity(self):
        """Similarity between all models"""
        n = len(self.hmm_models)
        sim_matrix = np.zeros((n, n))
        n_ix, m_ix = np.triu_indices(n)

        for i, j in zip(n_ix, m_ix):
            sim_matrix[i, j] = HMM_similarity(self.hmm_models[i], self.hmm_models[j])

        # make symmetric
        lower_ix = np.tril_indices(n, -1)
        sim_matrix[lower_ix] = sim_matrix.T[lower_ix]

        return sim_matrix


def fisher_rao(params_1, params_2):
    """Univariate Fisher-Rao distance.
    """
    mu_1, sig_1 = params_1
    mu_2, sig_2 = params_2
    l1 = (mu_1 - mu_2) ** 2 + 2 * (sig_1 - sig_2) ** 2
    l2 = (mu_1 + mu_2) ** 2 + 2 * (sig_1 + sig_2) ** 2
    f = np.sqrt(l1 * l2)
    dist = np.sqrt(2) * (np.log(f + (mu_1 - mu_2) ** 2 + 2 * (sig_1 ** 2 + sig_2 ** 2)) - np.log(4 * sig_1 * sig_2))
    return dist


def multivariate_fisher_rao(params_1, params_2):
    """Multivariate Fisher-Rao distance with given diagonal covariance matrix and means.

    We consider the parameters theta = (mu_1, sig_1, mu_2, sig_2, ... mu_n, sig_n).
    The distance is given by:
        d(theta_1, theta_2) = sqrt(sum(d((mu_1, sig_1), (mu_2, sig_2) ** 2)
        where d is the Fisher-Rao distance in the univariate case.
    """
    mu_1, sig_1 = params_1
    mu_2, sig_2 = params_2
    m, n = len(mu_1), len(mu_2)
    assert m == n, f"two distributions with different dimensions: {m} != {n}"
    sig_1, sig_2 = sig_1.diagonal(), sig_2.diagonal()
    dist = [fisher_rao((mu_1[i], sig_1[i]), (mu_2[i], sig_2[i])) ** 2 for i in range(m)]
    return np.sqrt(np.sum(dist))


def kl_multi_div(params_1, params_2):
    """Kullback-Leibler divergence for multivariate diagonal-covariance gaussians.

    The divergence is defined as:

    D( (mu_1, sig_1) || (mu_2, sig_2) )
        = .5 * ( Spur(sig_2**(-1) * sig_1) + (mu_2 - mu_1).T * sig_2**(-1)
            * (mu_2 - mu_1) - k + ln(det(sig_2) / det(sig_1)))

    with:
        mu_1, mu_2 as means
        sig_1, sig_2 as diagonal covariance matrices
        k as number of dimensions

    Args:
        params_1 (np.ndarray, np.ndarray): mu_1, sig_1
        params_2 (np.ndarray, np.ndarray): mu_2, sig_2

    Returns:
        float: KL divergence
    """
    mu_1, sig_1 = params_1
    mu_2, sig_2 = params_2
    mu_diff = mu_2 - mu_1
    sig_2_inv = np.linalg.inv(sig_2)
    k = mu_1.shape[0]

    tr_term = np.trace(sig_2_inv @ sig_1)
    det_term = np.log(np.linalg.det(sig_2) / np.linalg.det(sig_1))
    quad_term = mu_diff.T @ sig_2_inv @ mu_diff

    kl = .5 * (tr_term + det_term + quad_term - k)
    return kl


def js_multi_div(params_1, params_2):
    """Jensen-Shannon divergence for multivariate gaussians.

    The divergence is defined as:

    D = .5 * Dkl(P || M) + .5 * Dkl(Q || M)
    where P and Q are two probability distributions and M is the average of the two distributions:
    M = .5 * (P + Q)

    Returns:
        float: JS divergence
    """
    mu_1, sig_1 = params_1
    mu_2, sig_2 = params_2
    M_mu = .5 * (mu_1 + mu_2)
    M_sig = .5 * (sig_1 + sig_2)
    return (.5 * kl_multi_div(params_1, (M_mu, M_sig))) + (.5 * kl_multi_div(params_2, (M_mu, M_sig)))


def gini(x, epsilon=1e-8):
    # mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # relative mean absolute difference
    rmad = mad / (np.mean(x) + epsilon)
    # gini coefficient
    g = 0.5 * rmad
    return g


def HMM_similarity(hmm_1, hmm_2, method='js', epsilon=0.01):
    """HMM similarity"""
    n = hmm_1.n_components
    m = hmm_2.n_components
    stationary_1 = hmm_1.get_stationary_distribution()
    stationary_2 = hmm_2.get_stationary_distribution()

    avail_methods = ['js', 'fisher']
    method = method.lower()
    assert method in avail_methods, f'method not in available methods: {avail_methods}'
    if method == 'js':
        dist_func = js_multi_div
    elif method == 'fisher':
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
    similarity = .5 * (gini_rows + gini_cols)

    # if hmm_1 == hmm_2:
    #     print(np.round(Se_matrix, 2))
    #     print(stationary_1, stationary_2)
    #     # print(np.round(np.outer(stationary_1, stationary_2), 2))
    #     print(Q)
    #     print(gini_rows, gini_cols)
    #     print(f"Sim: {similarity}")
    return similarity


