import anndata as ad
import numpy as np
import scipy as sc
import pandas as pd
from tqdm import tqdm
from hmmlearn import hmm
from sklearn.utils import check_random_state
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
from typing import Union, Optional, Tuple
import logging
from collections import OrderedDict

from morphelia.tools import choose_representation

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def _fit_hmm(
    X: np.ndarray,
    n_components: int,
    lengths: Optional[np.ndarray] = None,
    init_startprob: Optional[np.ndarray] = None,
    init_transmat: Optional[np.ndarray] = None,
    init_means: Optional[np.ndarray] = None,
    init_covars: Optional[np.ndarray] = None,
    n_init: int = 10,
    emissions: str = 'g',
    **kwargs,
):
    avail_emissions = ['g', 'gm']
    assert emissions in avail_emissions, f"emissions must be one of {avail_emissions}, instead got {emissions}"
    rs = check_random_state(546)

    kwargs.setdefault("covariance_type", "full")
    kwargs.setdefault("random_state", rs)

    best_ll = None
    best_model = None

    for i in range(n_init):
        if emissions == 'g':
            model = hmm.GaussianHMM(n_components=n_components, **kwargs)
        elif emissions == 'gm':
            model = hmm.GMMHMM(n_components=n_components, **kwargs)
        else:
            raise NotImplementedError(f"Emission {emissions} not implemented")

        if init_startprob is not None:
            model.startprob_ = init_startprob
        if init_transmat is not None:
            model.transmat_ = init_transmat
        if init_means is not None:
            model.means_ = init_means
        if init_covars is not None:
            model.covars_ = init_covars

        model.fit(X, lengths=lengths)
        ll = model.score(X, lengths=lengths)

        if best_model is None or ll > best_ll:
            best_ll = ll
            best_model = model

    aic = best_model.aic(X, lengths=lengths)
    bic = best_model.bic(X, lengths=lengths)
    silhouette = silhouette_score(X, best_model.predict(X, lengths=lengths))

    return best_model, best_ll, aic, bic, silhouette


def _get_X_and_lengths(
    adata: ad.AnnData,
    track_key: str,
    time_key: str,
    rep: Optional[str] = None,
    n_pcs: Optional[int] = None,
    return_index: bool = False,
    min_track_len: Optional[int] = None
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, pd.Index]]:
    adata.strings_to_categoricals()
    # sort adata by tracks and time
    adata = adata[adata.obs.sort_values([track_key, time_key]).index, :]

    # get length and X
    tracks, lengths = np.unique(adata.obs[track_key], return_counts=True)
    if min_track_len is not None:
        tracks = tracks[lengths >= min_track_len]
        lengths = lengths[lengths >= min_track_len]
        X = choose_representation(adata[adata.obs[track_key].isin(tracks), :], rep=rep, n_pcs=n_pcs)
    else:
        X = choose_representation(adata, rep=rep, n_pcs=n_pcs)

    if return_index:
        return X, lengths, adata.obs.index
    return X, lengths


def _get_track_subsample(
    adata: ad.AnnData, track_key: str, sample_size: int
) -> ad.AnnData:
    adata.strings_to_categoricals()
    # get subsample of all tracks
    root_choice = np.random.choice(
        adata.obs[track_key].unique(), size=sample_size, replace=False
    )
    sdata = adata[adata.obs[track_key].isin(root_choice), :]

    return sdata


def _get_best_model(
        models: list,
        model_selection_params: dict,
        select_model_by: str = 'bic',
        verbose: bool = False
):
    avail_methods = ['bic', 'aic', 'silhouette']
    assert (
        select_model_by in avail_methods
    ), f'select_model_by must be one of {avail_methods}, instead got {select_model_by}'

    if select_model_by == 'bic' or select_model_by == 'aic':
        select_ix = np.argmin(model_selection_params[select_model_by])
        select_criterion = np.min(model_selection_params[select_model_by])
    else:
        select_ix = np.argmax(model_selection_params[select_model_by])
        select_criterion = np.max(model_selection_params[select_model_by])

    selected_component = model_selection_params['tested_components'][select_ix]

    if select_ix == 0:
        logger.warning(f"Lower bound of components seems to be too high, selected component {selected_component}")
    elif select_ix == (len(model_selection_params['bic']) - 1):
        logger.warning(f"Upper bound of components seems to be too low, selected component {selected_component}")

    if verbose:
        logger.info(f"Choose {selected_component} components with {select_model_by}: {select_criterion}")

    return models[select_ix]


def fit_hmm(
    adata: ad.AnnData,
    n_components: Union[int, list],
    track_key: str,
    time_key: str,
    hidden_state_key: Optional[str] = "HMMState",
    rep: Optional[str] = "X_umap",
    n_pcs: Optional[str] = None,
    select_model_by: str = 'bic',
    emissions: str = 'g',
    min_track_len: Optional[int] = None,
    sample_size: int = 1000,
    n_init: int = 10,
    seed: int = 0,
    verbose: bool = False,
    **kwargs,
) -> ad.AnnData:
    """Find Hidden Morphological States with Hidden Markov Models.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    n_components : int, list
        Number of Components for different HMMs.
        The HMM with the best BIC is selected.
    track_key : str
        Name of observation with track identifiers
    time_key : str
        Name of observation with time
    hidden_state_key : str, optional
        Key which should be used to store hidden states in `.obs`
    rep : str, optional
        Representation in `.obsm`
    n_pcs : int, optional
        Number of principal components if `rep` is `X_pca`
    select_model_by : str
        Measure to use for model selection.
        Can be bic, aic, or silhouette.
    emissions : str
        'g' for gaussian emissions.
        'gm' for gaussian mixture emissions.
    min_track_len : int, optional
        Minimum track length for fit
    sample_size : int
        Number of tracks to sample for the fitting step
    n_init : int
        Number of initializations per component
    seed : int
    verbose : bool
    **kwargs
        Keyword arguments are passed to hmmlearn.hmm.GaussianHMM

    Returns
    -------
    adata : anndata.AnnData
        AnnData object with HMM parameters in `.uns` and hidden states
        in `.obs[hidden_state_key]`
    """
    np.random.seed(seed)
    # get subsample of all tracks
    sdata = _get_track_subsample(adata, track_key, sample_size)

    X, lengths = _get_X_and_lengths(
        sdata, track_key=track_key, time_key=time_key, rep=rep, n_pcs=n_pcs, min_track_len=min_track_len
    )

    if isinstance(n_components, int):
        n_components = [n_components]

    aics = []
    bics = []
    silhouettes = []
    lls = []

    models = []

    for n in (pbar := tqdm(n_components, unit="Components")):
        pbar.set_description(f"{n} Components")

        model, ll, aic, bic, silhouette = _fit_hmm(
            X, lengths=lengths, n_components=n, n_init=n_init, emissions=emissions, **kwargs
        )
        aics.append(aic)
        bics.append(bic)
        silhouettes.append(silhouette)
        lls.append(ll)

        models.append(model)

    model_selection_params = dict(
        tested_components=n_components,
        aic=aics,
        bic=bics,
        silhouette=silhouettes,
        ll=lls
    )
    best_model = _get_best_model(
        models=models,
        model_selection_params=model_selection_params,
        select_model_by=select_model_by,
        verbose=verbose
    )

    # predict hidden states for all cells
    X, lengths, index = _get_X_and_lengths(
        adata, track_key=track_key, time_key=time_key, rep=rep, n_pcs=n_pcs, return_index=True
    )
    states = best_model.predict(X, lengths=lengths)

    if hidden_state_key in adata.obs.columns:
        adata.obs = adata.obs.drop(columns=[hidden_state_key])

    adata.obs.loc[index, hidden_state_key] = states
    adata.obs[hidden_state_key] = (
        adata.obs[hidden_state_key].astype("int").astype("category")
    )

    params = dict(
        stationary=best_model.get_stationary_distribution(),
        transmat=best_model.transmat_,
        n_components=best_model.n_components,
        startprob=best_model.startprob_,
        covars=best_model.covars_,
        means=best_model.means_,
    )

    adata.uns["hmm"] = {}
    adata.uns["hmm"]["params"] = params
    adata.uns["hmm"]["model_selection_params"] = model_selection_params

    return adata


def fit_hmm_by_key(
    adata: ad.AnnData,
    key: str,
    track_key: str,
    time_key: str,
    rep: Optional[str] = "X_umap",
    n_pcs: Optional[str] = None,
    emissions: str = 'g',
    min_track_len: Optional[int] = None,
    sample_size: int = 100,
    n_init: int = 10,
    seed: int = 0,
    **kwargs,
) -> ad.AnnData:
    """Fit Hidden Markov Models on different instances.

    The parameters for model initialization must be estimated beforehand
    using `fit_hmm`. Because all instances in `key` are initialized with
    the same estimated parameters hidden states are comparable afterwards.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    track_key : str
        Name of observation with track identifiers
    time_key : str
        Name of observation with time
    rep : str, optional
        Representation in `.obsm`
    n_pcs : int, optional
        Number of principal components if `rep` is `X_pca`
    emissions : str
        'g' for gaussian emissions.
        'gm' for gaussian mixture emissions.
    min_track_len : int, optional
        Minimum track length for fit
    sample_size : int
        Number of tracks to sample for the fitting step
    n_init : int
        Number of initializations per component
    seed : int
    **kwargs
        Keyword arguments are passed to hmmlearn.hmm.GaussianHMM

    Raises
    -------
    AssertionError
        If 'hmm' is not in '.uns'

    Returns
    -------
    adata : anndata.AnnData
        AnnData object with HMM parameters in `.uns[f"{key}_hmm"]` and hidden states
        in `.obs[`key`_HMMState]`.
        A summary of the stationary distributions is stored in `.uns[f"{key}_stationary_dist"]`.
    """
    np.random.seed(seed)

    assert "hmm" in adata.uns.keys(), "``hmm` is not in `.uns`, use `fit_hmm` before"
    hidden_state_key = f"{key}_HMMState"
    adata.obs[hidden_state_key] = -1

    adata.strings_to_categoricals()
    unique_keys = adata.obs[key].cat.categories

    key_summary = OrderedDict()

    kwargs.setdefault("init_params", "")

    for k in (pbar := tqdm(unique_keys)):
        pbar.set_description(f"Key: {k}")

        kdata = adata[adata.obs[key] == k, :]
        # get subsample
        sdata = _get_track_subsample(kdata, track_key, sample_size)
        X, lengths = _get_X_and_lengths(
            sdata, track_key=track_key, time_key=time_key, rep=rep, n_pcs=n_pcs, min_track_len=min_track_len
        )

        model, ll, aic, bic, silhouette = _fit_hmm(
            X,
            lengths=lengths,
            n_components=adata.uns["hmm"]["params"]["n_components"],
            init_startprob=adata.uns["hmm"]["params"]["startprob"],
            init_transmat=adata.uns["hmm"]["params"]["transmat"],
            init_covars=adata.uns["hmm"]["params"]["covars"],
            init_means=adata.uns["hmm"]["params"]["means"],
            n_init=n_init,
            emissions=emissions,
            **kwargs,
        )

        params = dict(
            aic=aic,
            bic=bic,
            ll=ll,
            key=key,
            stationary=model.get_stationary_distribution(),
            transmat=model.transmat_,
            n_components=model.n_components,
            startprob=model.startprob_,
            covars=model.covars_,
            means=model.means_,
        )
        key_summary[k] = params

        # predict hidden states
        X, lengths, index = _get_X_and_lengths(
            kdata, track_key=track_key, time_key=time_key, rep=rep, n_pcs=n_pcs, return_index=True
        )
        states = model.predict(X, lengths=lengths)
        adata.obs.loc[index, hidden_state_key] = states

    if hidden_state_key in adata.obs.columns:
        adata.obs = adata.obs.drop(columns=[hidden_state_key])

    adata.obs[hidden_state_key] = (
        adata.obs[hidden_state_key].astype(int).astype("category")
    )

    adata.uns[f"hmm_by_key"] = key_summary

    return adata


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon Distance between two probability distributions."""
    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (sc.stats.entropy(p, m) + sc.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


def hmm_distance(
    adata: ad.AnnData,
    key: str,
    include_startprob: bool = False,
    make_plot: bool = False,
    save: Optional[str] = None,
    show: bool = False,
    return_array: bool = False,
) -> Union[pd.DataFrame, np.ndarray, Tuple[pd.DataFrame, plt.Figure, plt.Axes]]:
    """Distance between different Hidden Markov Models.

    This function computes the distance between two HMMs based on their
    stationary distribution. Because the stationary distribution is a
    probability distribution that sums up to 1, the Jensen-Shannon distance
    can be used.
    Use `.fit_hmm_by_key` beforehand.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    key : str
        Key in `.obs` with precomputed HMMs
    include_startprob : bool
        Include start probability in distance matrix
    make_plot : bool
        Plot the distance matrix
    save : bool
        Path where to save the figure
    show : bool
        Show plot and return figure and axis
    return_array : bool
        Return distance matrix as numpy array

    Raises
    ------
    AssertionError
        If `key_hmm` is not in `.uns`
    OSError
        If figure can not be saved at specified location

    Returns
    -------
    pd.DataFrame
        Distance Matrix for all HMMs
    """
    uns_key = f"{key}_hmm"
    assert (
        uns_key in adata.uns
    ), f"{uns_key} not in `.uns`, use `fit_hmm_by_key` beforehand"

    adata.strings_to_categoricals()
    unique_keys = list(adata.obs[key].cat.categories)
    stat_dict = {k: adata.uns[uns_key][k]["stationary"] for k in unique_keys}

    if include_startprob:
        assert "hmm" in adata.uns, "`hmm` not in `.uns`, use `fit_hmm` beforehand"
        startprob = adata.uns["hmm"]["startprob"]
        unique_keys = unique_keys + ["startprob"]
        stat_dict["startprob"] = startprob

    D = pd.DataFrame(columns=unique_keys, index=unique_keys, dtype=float)
    for i in unique_keys:
        for j in unique_keys:
            D.loc[i, j] = jensen_shannon_distance(stat_dict[i], stat_dict[j])

    if make_plot:
        sns.set_theme()
        cmap = mpl.cm.plasma

        fig = plt.figure(figsize=(7, 5))
        ax = sns.heatmap(D, cmap=cmap)
        plt.suptitle("HMM similarity", fontsize=16)

        if save:
            try:
                plt.savefig(os.path.join(save, "feature_correlation.png"))
            except OSError:
                print(f"Can not save figure to {save}.")

        if show:
            plt.show()
            return D, fig, ax

    if return_array:
        return D.values
    return D
