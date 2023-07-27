import numpy as np
import anndata as ad
from scipy.spatial import cKDTree
from tqdm import tqdm
import btrack
from btrack.constants import BayesianUpdates
import pandas as pd

import logging
from typing import Union, Tuple, List, Optional

logger = logging.getLogger("worker_process")
logger.setLevel(level=logging.WARN)

btrack_config = {
    "MotionModel": {
        "name": "cell_motion",
        "dt": 1.0,
        "measurements": 3,
        "states": 6,
        "accuracy": 7.5,
        "prob_not_assign": 0.001,
        "max_lost": 5,
        "A": {
            "matrix": [
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ]
        },
        "H": {"matrix": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]},
        "P": {
            "sigma": 150.0,
            "matrix": [
                0.1,
                0,
                0,
                0,
                0,
                0,
                0,
                0.1,
                0,
                0,
                0,
                0,
                0,
                0,
                0.1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
        },
        "G": {"sigma": 15.0, "matrix": [0.5, 0.5, 0.5, 1, 1, 1]},
        "R": {"sigma": 5.0, "matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
    },
    "ObjectModel": {},
    "HypothesisModel": {
        "name": "cell_hypothesis",
        "hypotheses": ["P_FP", "P_init", "P_term", "P_link", "P_branch", "P_dead"],
        "lambda_time": 5.0,
        "lambda_dist": 3.0,
        "lambda_link": 10.0,
        "lambda_branch": 50.0,
        "eta": 1e-10,
        "theta_dist": 20.0,
        "theta_time": 5.0,
        "dist_thresh": 40,
        "time_thresh": 2,
        "apop_thresh": 5,
        "segmentation_miss_rate": 0.1,
        "apoptosis_rate": 0.001,
        "relax": True,
    },
}

fate_dict = {
    "FALSE_POSITIVE": "false_positive",
    "DIVIDE": "divide",
    "APOPTOSIS": "apoptosis",
    "TERMINATE_BACK": "terminate_back",
    "TERMINATE_LAZY": "terminate_lazy",
}


def track(
    adata: ad.AnnData,
    time_var: str = "Metadata_Time",
    group_vars: Union[Tuple[str], List[str], str] = ("Metadata_Well", "Metadata_Field"),
    x_loc: str = "Cells_Location_Center_X",
    y_loc: str = "Cells_Location_Center_Y",
    parent_id: str = "Metadata_Track_Parent",
    track_id: str = "Metadata_Track",
    fate_id: str = "Metadata_Fate",
    root_id: str = "Metadata_Track_Root",
    gen_id: str = "Metadata_Gen",
    use_features: Optional[Union[str, List[str]]] = None,
    filter_dummies: bool = False,
    filter_delayed_roots: bool = False,
    drop_untracked: bool = True,
    max_search_radius: int = 100,
    field_size: Tuple[int, int] = (2048, 2048),
    disable_btrack_logger: bool = False,
    approx: bool = False,
    verbose: bool = False,
) -> ad.AnnData:
    """Wrapper for Bayesian Tracker.

    Tracks objects per field of view defined by group_vars.
    Objects can be filtered for trajectory length and dummy objects.

    Parameters
    ----------
    adata : anndata.AnnData
        Morphological data from cells with different time stamps
    time_var : str
        Variable name for time point in annotations
    group_vars : str, list of str, tuple of str
        Variables in annotations.
        Should point to specific wells/ or fields on plates that can be compared over time
    x_loc : str
        Identifier for x location in annotations
    y_loc : str
        Identifier for y location in annotations
    parent_id : str
        Variable name used to store index of parent track
    track_id : str
        Variable name used to store unique track id
    fate_id : str
        Variable name used to store fate of a track
    root_id : str
        Variable name used to store root of a track
    gen_id : str
        Variable name to store generation of a track
    use_features : list
        Use these features for tracking
    filter_dummies : bool
        Filter trees with dummies
    filter_delayed_roots : bool
        Filter roots that initiate after t=0
    drop_untracked : bool
        Drop all cell that have not been tracked
    max_search_radius : int
        Local spatial search radius (pixels)
    field_size : tuple of ints
        Height and width in of field
    disable_btrack_logger : bool
        Disable the btrack logger
    approx : bool
        Speed up processing on very large datasets
    verbose : bool

    Returns
    -------
    anndata.AnnData
        AnnData object with track information

    Raises
    -------
    AssertionError
        If `group_vars`, `time_var`, `x_loc` or `y_loc` not in `.obs`
    AssertionError
        If `x_loc` equals `y_loc`

    References
    ----------
    .. [1] Automated deep lineage tree analysis using a Bayesian single cell tracking approach
       Ulicna K, Vallardi G, Charras G and Lowe AR. bioRxiv (2020)
    """
    if disable_btrack_logger:
        btrack_logger = logging.getLogger("btrack")
        btrack_logger.setLevel(logging.WARNING)
    # check variables
    if isinstance(group_vars, str):
        group_vars = [group_vars]
    elif isinstance(group_vars, tuple):
        group_vars = list(group_vars)

    if isinstance(group_vars, list):
        assert all(
            gv in adata.obs.columns for gv in group_vars
        ), f"One or all group_vars not in .obs.columns: {group_vars}"
    else:
        raise KeyError(
            f"Expected type(list) or type(str) for group_vars, "
            f"instead got {type(group_vars)}"
        )

    if isinstance(use_features, str):
        use_features = [use_features]
    if use_features is not None:
        tracking_updates = ["motion", "visual"]
    else:
        tracking_updates = ["motion"]

    assert x_loc != y_loc, (
        f"x_loc expected to be different from y_loc, " f"x_loc: {x_loc}, y_loc: {y_loc}"
    )

    field_height, field_width = field_size

    # create new column to store index of parent object
    adata.obs[parent_id] = -1
    # create new column and store id for every trace tree
    adata.obs[track_id] = -1
    # store fate, root and generation
    adata.obs[fate_id] = np.nan
    adata.obs[root_id] = -1
    adata.obs[gen_id] = -1

    # load config settings
    t_config = {
        "motion_model": btrack.utils.read_motion_model(btrack_config),
        "object_model": btrack.utils.read_object_model(btrack_config),
        "hypothesis_model": btrack.utils.read_hypothesis_model(btrack_config),
    }

    # id of first track
    track_start_id = 0

    # iterate over every field and get field at different times
    total_its = np.prod([len(adata.obs[gv].unique()) for gv in group_vars])
    for ix, (groups, field_df) in tqdm(
        enumerate(adata.obs.groupby(list(group_vars))),
        desc="Tracking cells in every field",
        total=int(total_its),
    ):
        # create objects for btracker
        z = np.zeros((len(field_df),))
        _id = field_df.index

        objects_raw = {
            "x": field_df[x_loc].to_numpy(),
            "y": field_df[y_loc].to_numpy(),
            "z": z,
            "t": field_df[time_var].to_numpy(),
            "id": _id,
        }
        if use_features is not None:
            for feat in use_features:
                objects_raw[feat] = adata[_id, feat].X.flatten()

        objects = btrack.io.objects_from_dict(objects_raw)

        # start tracker
        with btrack.BayesianTracker(verbose=False) as tracker:
            tracker.configure(t_config)

            tracker.max_search_radius = max_search_radius

            if approx:
                tracker.update_method = BayesianUpdates.APPROXIMATE

            if use_features is not None:
                tracker.features = use_features

            tracker.append(objects)

            # set the volume (Z axis volume is set very large for 2D data)
            tracker.volume = ((0, field_height), (0, field_width), (-1e5, 1e5))

            # track them
            tracker.track(tracking_updates=tracking_updates)

            # generate hypotheses and run the global optimizer
            tracker.optimize()

            tracks = tracker.tracks

        t_header = ["ID", "t"] + ["z", "y", "x"][-2:]
        p_header = ["t", "state", "generation", "root", "parent", "dummy"]

        header = t_header + p_header
        dict_tracks = _tracks_as_dict(tracks, header, add_fate=True)

        cat_tracks = _cat_track_dicts(dict_tracks)
        output = pd.DataFrame(cat_tracks)

        # filter delayed roots
        if filter_delayed_roots:
            delayed_roots = output.loc[output["ID"] == output["parent"], :]
            delayed_roots = (
                delayed_roots.groupby("ID")
                .filter(lambda x: x["t"].min() != 0)["root"]
                .unique()
            )
            output = output.loc[~output["root"].isin(delayed_roots), :]
        # filter roots that contain dummies
        if filter_dummies:
            dummy_roots = output.loc[output["dummy"], "root"].unique()
            output = output.loc[~output["root"].isin(dummy_roots), :]
        # add absolute ids to output
        if len(output) > 0:
            output, end_id = _add_absolute_ids(
                output, start_id=track_start_id, return_end_id=True
            )
            # delete dummies before updating adata
            output = output[output["id"] != "nan"]

            track_start_id = end_id

            # update adata
            if len(output) > 0:
                adata.obs.loc[
                    output["id"], [track_id, parent_id, fate_id, root_id, gen_id]
                ] = output[["track_id", "parent_id", "fate", "root_id", "gen"]].to_numpy()

    if drop_untracked:
        len_before = len(adata)
        # drop untracked cells
        adata = adata[adata.obs[track_id] != -1, :]
        adata = adata[adata.obs[parent_id] != -1, :]
        adata = adata[adata.obs[root_id] != -1, :]
        adata = adata[adata.obs[fate_id].notna(), :]
        if verbose:
            logging.info(
                f"Dropped {len_before - len(adata)} cells with false positive tracks."
            )

    return adata.copy()


def _add_absolute_ids(
    output: Union[pd.DataFrame, dict],
    start_id: Optional[int] = None,
    return_end_id: bool = False,
):
    """Change track ids to absolute ids."""
    track_ids = output["ID"].unique()
    n_tracks = len(track_ids)

    if start_id is None:
        start_id = 0
    end_id = start_id + n_tracks

    ids = list(range(start_id, end_id))
    mapping = {track_id: new_id for track_id, new_id in zip(track_ids, ids)}
    output["track_id"] = output["ID"].map(mapping)
    output["parent_id"] = output["parent"].map(mapping)
    output["root_id"] = output["root"].map(mapping)
    output["gen"] = output["generation"]

    if return_end_id:
        return output, end_id

    return output


def _tracks_as_dict(tracks: list, properties: list, add_fate: bool = False):
    """Convert tracks to dictionaries."""
    # ensure lexicographic ordering of tracks
    tracks = sorted(list(tracks), key=lambda t: t.ID)
    dict_tracks = []

    for ix, tr in enumerate(tracks):
        trk = tr.to_dict(properties)

        if add_fate:
            fate = tr.fate.name
            if fate in fate_dict:
                fate = fate_dict[fate]
            else:
                fate = "false_positive"
            trk["fate"] = fate

        data = {}

        for key in trk.keys():
            prop = trk[key]
            if not isinstance(prop, (list, np.ndarray)):
                prop = [prop] * len(tr)

            assert len(prop) == len(tr)
            data[key] = prop

        dict_tracks.append(data)

    return dict_tracks


def _cat_track_dicts(tracks: list):
    """Concatenate tracks."""
    data = {}

    for ix, trk in enumerate(tracks):

        if not data:
            data = {k: [] for k in trk.keys()}

        for key in data.keys():
            prop = trk[key]
            assert isinstance(
                prop, (list, np.ndarray)
            ), "all track properties must be lists"

            data[key].append(prop)

    for key in data.keys():
        data[key] = np.concatenate(data[key])

    return data


def track_nn(
    adata: ad.AnnData,
    time_var: str = "Metadata_Time",
    group_vars: Union[str, List[str], Tuple[str]] = ("Metadata_Well", "Metadata_Field"),
    x_loc: str = "Cells_Location_Center_X",
    y_loc: str = "Cells_Location_Center_Y",
    parent_id: str = "Metadata_Trace_Parent",
    tree_id: str = "Metadata_Trace_Tree",
    start_tp: int = 0,
) -> ad.AnnData:
    """Neares neighbor tracking.

    Iterates over every well or field that can be traced over time.
    Fore every object and time point calculate the nearest neighbor from
    the previous time point.

    Parameters
    ----------
    adata : anndata.AnnData
        Morphological data from cells with different time stamps
    time_var : str
        Variable name for time point in annotations
    group_vars : str or list of str or tuple of str
        Variables in annotations.
        Should point to specific wells/ or fields on plates that can be compared over time.
    x_loc : str
        Identifier for x location in annotations
    y_loc : str
        Identifier for y location in annotations
    parent_id : str
        Variable name used to store index of parent cell
    tree_id : str
        Variable name used to store unique branch number for a certain field
    start_tp : int
        Start time point

    Raises
    -------
    AssertionError
        If `group_vars`, `time_var`, `x_loc` or `y_loc` is not in `.obs`
    AssertionError
        If `x_loc` equals `y_loc`
    ValueError
        If AnnData has no data with a given `start_tp`

    Returns
    -------
    anndata.AnnData
        AnnData object with track information
    """

    # check variables
    if isinstance(group_vars, str):
        group_vars = [group_vars]
    elif isinstance(group_vars, tuple):
        group_vars = list(group_vars)

    if isinstance(group_vars, list):
        assert all(
            gv in adata.obs.columns for gv in group_vars
        ), f"One or all group_vars not in .obs.columns: {group_vars}"
    else:
        raise KeyError(
            f"Expected type(list) or type(str) for group_vars, "
            f"instead got {type(group_vars)}"
        )

    assert time_var in adata.obs.columns, f"time_var not in .obs.columns: {time_var}"
    assert x_loc in adata.obs.columns, f"x_loc not in .obs.columns: {x_loc}"
    assert y_loc in adata.obs.columns, f"y_loc not in .obs.columns: {y_loc}"
    assert x_loc != y_loc, (
        f"x_loc expected to be different from y_loc, " f"x_loc: {x_loc}, y_loc: {y_loc}"
    )

    # create new column to store index of parent object
    adata.obs[parent_id] = np.nan
    # create new column and store id for every trace tree
    adata.obs[tree_id] = np.nan
    n_start_tp = len(adata[adata.obs[time_var] == start_tp])
    if n_start_tp > 1:
        adata.obs.loc[adata.obs[time_var] == start_tp, tree_id] = range(n_start_tp)
    else:
        raise ValueError(
            f"No observation with time_var {time_var} and start_tp {start_tp}"
        )

    # iterate over every field and get field at different times
    total_its = np.prod([len(adata.obs[gv].unique()) for gv in group_vars])
    for ix, (groups, field_df) in tqdm(
        enumerate(adata.obs.groupby(list(group_vars))),
        desc="Tracking cells in every field",
        total=int(total_its),
    ):

        # cache lagged values
        lagged = None

        # iterate over timepoints
        for t, t_df in field_df.groupby(time_var):

            # find closest object in lagged objects
            if lagged is not None:
                # get locations of objects and lagged objects
                t_loc = t_df[[x_loc, y_loc]].to_numpy()
                t_loc_lagged = lagged[[x_loc, y_loc]].to_numpy()

                # get nearest object in lagged objects for every object
                tree = cKDTree(t_loc_lagged)
                _, parent_ix = tree.query(t_loc, k=1)

                # assign lagged trace ids to objects
                adata.obs.loc[t_df.index, tree_id] = lagged.iloc[parent_ix][
                    tree_id
                ].tolist()
                # assign trace parents to objects
                adata.obs.loc[t_df.index, parent_id] = lagged.iloc[parent_ix].index

            # cache field_df
            lagged = adata.obs.loc[t_df.index, [x_loc, y_loc, tree_id]]

    return adata
