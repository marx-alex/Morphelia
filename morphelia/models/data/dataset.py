from collections import defaultdict
from typing import Optional, Union, List

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import anndata as ad

from ._utils import data_converter


class LineageTreeDataset(Dataset):
    """Dataset handler for tracked lineage trees.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    target_key : str
        Target variable
    root_key : str
        Variable with lineage roots
    track_key : str
        Variable with lineage tracks
    parent_key :str
        Variable with lineage parents
    time_key : str
        Time variable
    condition_key : str, optional
        Contition variable
    seq_len : str or int
        Length of sequences. If `seq_len` is `max`, the maximum sequence length is used.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        target_key: str = "Metadata_Treatment",
        root_key: str = "Metadata_Track_Root",
        track_key: str = "Metadata_Track",
        parent_key: str = "Metadata_Track_Parent",
        time_key: str = "Metadata_Time",
        condition_key: Optional[str] = None,
        seq_len: Union[str, int] = "max",
    ) -> None:
        self.root = adata.obs[root_key].to_numpy()
        self.track = adata.obs[track_key].to_numpy()
        self.parent = adata.obs[parent_key].to_numpy()
        self.unique_trees = np.unique(self.root)
        self.pathways = self._find_all_paths()

        # get data
        self.data = data_converter(adata.X)

        # convert string obs to categories
        adata.strings_to_categoricals()
        # get target
        self.target = data_converter(adata.obs[target_key]).long()

        # get condition
        self.condition_key = condition_key
        if self.condition_key is not None:
            self.conditions = data_converter(
                adata.obs[condition_key].astype("category")
            ).long()
            self.n_conditions = len(self.conditions.unique())

        # get time
        self.time = data_converter(adata.obs[time_key])
        # unique time points
        unique_tps, _ = self.time.unique().sort()
        self.n_tps = len(unique_tps)
        self.time_idx = dict(zip(unique_tps.tolist(), list(range(self.n_tps))))
        self.seq_len = seq_len
        if seq_len == "max":
            self.seq_len = self.n_tps

        # get absolute indices
        self.ids = torch.tensor(range(len(adata)))

    def __getitem__(self, index):
        # generate output
        output = dict()

        # get absolute indices from root indices
        tracks = self.pathways[index]
        data_mask = torch.from_numpy(np.in1d(self.track, tracks))

        # get time of samples
        time = self.time[data_mask]
        assert len(time) == len(
            torch.unique(time)
        ), "Time points are not unique paths, check pathways for uniqueness of time points."
        time_idx = time.apply_(lambda x: self.time_idx[x]).long()

        # get data
        x = torch.zeros(self.n_tps, self.data.shape[-1])
        x[time_idx, :] = self.data[data_mask, :]
        padding_mask = torch.ones(self.n_tps, dtype=torch.bool)
        padding_mask[time_idx] = 0

        # get ids
        ids = torch.full(size=(1, self.n_tps), fill_value=-1).flatten()
        ids[time_idx] = self.ids[data_mask]

        # get target
        target = self.target[data_mask]
        assert (
            len(target.unique()) == 1
        ), "Targets are not unique for paths, check pathways for uniqueness of targets."
        target = target[0]

        # get condition
        if self.condition_key is not None:
            condition = self.conditions[data_mask]
            assert (
                len(condition.unique()) == 1
            ), "Conditions are not unique for paths, check pathways for uniqueness of conditions."
            condition = condition[0]
            output["c"] = condition

        # introduce padding
        if self.n_tps > self.seq_len:
            x = x[: self.seq_len, :]
            padding_mask = padding_mask[: self.seq_len]
            ids = ids[: self.seq_len]
        elif self.n_tps < self.seq_len:
            padding_2d = (0, 0, 0, self.seq_len - self.n_tps)
            padding_1d = (0, self.seq_len - self.n_tps)
            x = F.pad(x, padding_2d, "constant", 0)
            padding_mask = F.pad(padding_mask, padding_1d, "constant", 1)
            ids = F.pad(ids, padding_1d, "constant", -1)

        output["x"] = x
        output["target"] = target
        output["padding_mask"] = padding_mask
        output["ids"] = ids

        return output

    def __len__(self):
        return len(self.pathways)

    def _find_all_paths(self):
        """
        Find all paths in all lineages.
        """
        paths = []
        for root in self.unique_trees:
            root_mask = self.root == root
            tree_edges = list(set(zip(self.parent[root_mask], self.track[root_mask])))
            tree_dict = defaultdict(list)
            for i, j in tree_edges:
                tree_dict[i].append(j)

            paths.extend(self._recursive_paths(tree_dict, root))

        return paths

    def _recursive_paths(self, edges, node, paths=None, current_path=None):
        if paths is None:
            paths = []
        if current_path is None:
            current_path = []

        # delete self-directed edges
        if node in edges[node]:
            edges[node].remove(node)

        current_path.append(node)

        if len(edges[node]) == 0:
            paths.append(current_path)

        else:
            for child in edges[node]:
                self._recursive_paths(edges, child, paths, list(current_path))

        return paths


class TSDataset(Dataset):
    """Dataset handler for sequential data.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    target_key : str
        Target variable
    track_key : str
        Variable with lineage tracks
    time_key : str
        Time variable
    condition_key : str, optional
        Contition variable
    seq_len : str or int
        Length of sequences. If `seq_len` is `max`, the maximum sequence length is used.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        target_key: str = "Metadata_Treatment",
        track_key: str = "Metadata_Track",
        time_key: str = "Metadata_Time",
        condition_key: Optional[str] = None,
        seq_len: Union[str, int] = "max",
    ) -> None:
        self.tracks = adata.obs[track_key].to_numpy()
        self.unique_tracks = np.unique(self.tracks)
        self.n_tracks = len(self.unique_tracks)

        # get data
        self.data = data_converter(adata.X)

        # convert string obs to categories
        adata.strings_to_categoricals()
        # get target
        self.target = data_converter(adata.obs[target_key]).long()

        # get condition
        self.condition_key = condition_key
        if self.condition_key is not None:
            self.conditions = data_converter(
                adata.obs[condition_key].astype("category")
            ).long()
            self.n_conditions = len(self.conditions.unique())

        # get time
        self.time = data_converter(adata.obs[time_key])
        # unique time points
        unique_tps, _ = self.time.unique().sort()
        self.n_tps = len(unique_tps)
        self.time_idx = dict(zip(unique_tps.tolist(), list(range(self.n_tps))))
        self.seq_len = seq_len
        if seq_len == "max":
            self.seq_len = self.n_tps

        # get absolute indices
        self.ids = torch.tensor(range(len(adata)))

    def __getitem__(self, index):
        # generate output
        output = dict()

        # get absolute indices from root indices
        track = self.unique_tracks[index]
        data_mask = torch.from_numpy(self.tracks == track)

        # get time of samples
        time = self.time[data_mask]
        assert len(time) == len(
            torch.unique(time)
        ), "Time points are not unique paths, check pathways for uniqueness of time points."
        time_idx = time.apply_(lambda x: self.time_idx[x]).long()

        # get data
        x = torch.zeros(self.n_tps, self.data.shape[-1])
        x[time_idx, :] = self.data[data_mask, :]
        padding_mask = torch.ones(self.n_tps, dtype=torch.bool)
        padding_mask[time_idx] = 0

        # get ids
        ids = torch.full(size=(1, self.n_tps), fill_value=-1).flatten()
        ids[time_idx] = self.ids[data_mask]

        # get target
        target = self.target[data_mask]
        assert len(target.unique()) == 1, "Targets are not unique within tracks."
        target = target[0]

        # get condition
        if self.condition_key is not None:
            condition = self.conditions[data_mask]
            assert (
                len(condition.unique()) == 1
            ), "Conditions are not unique within tracks."
            condition = condition[0]
            output["c"] = condition

        # introduce padding
        if self.n_tps > self.seq_len:
            x = x[: self.seq_len, :]
            padding_mask = padding_mask[: self.seq_len]
            ids = ids[: self.seq_len]
        elif self.n_tps < self.seq_len:
            padding_2d = (0, 0, 0, self.seq_len - self.n_tps)
            padding_1d = (0, self.seq_len - self.n_tps)
            x = F.pad(x, padding_2d, "constant", 0)
            padding_mask = F.pad(padding_mask, padding_1d, "constant", 1)
            ids = F.pad(ids, padding_1d, "constant", -1)

        output["x"] = x
        output["target"] = target
        output["padding_mask"] = padding_mask
        output["ids"] = ids

        return output

    def __len__(self):
        return self.n_tracks


class SequenceDataset(Dataset):
    """Dataset handler for non sequential data, that can be load as sequences.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    target_key : str
        Target variable
    condition_key : str, optional
        Contition variable
    seq_len : str or int
        Length of sequences. If `seq_len` is `max`, the maximum sequence length is used.
    seed : int
        Seed partition assignment
    """

    def __init__(
        self,
        adata: ad.AnnData,
        target_key: str = "Metadata_Treatment",
        group_key: Union[List[str], str] = "Metadata_Treatment",
        condition_key: Optional[str] = None,
        seq_len: int = 5,
        seed: int = 0,
    ) -> None:
        # get data
        np.random.seed(seed)
        self.data = data_converter(adata.X)
        self.group_key = group_key
        self.seq_len = seq_len
        # get partitions
        self.partitions = self._partition_data(adata)
        self.n_partitions = len(self.partitions)

        # convert string obs to categories
        adata.strings_to_categoricals()
        # get target
        self.target = data_converter(adata.obs[target_key]).long()

        # get condition
        self.condition_key = condition_key
        if self.condition_key is not None:
            self.conditions = data_converter(
                adata.obs[condition_key].astype("category")
            ).long()
            self.n_conditions = len(self.conditions.unique())

    def __getitem__(self, index):
        # generate output
        output = dict()

        ids = self.partitions[index]

        # get data
        x = self.data[ids, :]

        # get target
        target = self.target[ids]
        assert len(target.unique()) == 1, "Targets are not unique within sequences."
        target = target[0]

        # get condition
        if self.condition_key is not None:
            condition = self.conditions[ids]
            assert (
                len(condition.unique()) == 1
            ), "Conditions are not unique within sequences."
            condition = condition[0]
            output["c"] = condition

        output["x"] = x
        output["target"] = target
        output["ids"] = ids

        return output

    def __len__(self):
        return self.n_partitions

    def _partition_data(self, adata: ad.AnnData):
        """
        Partition indices by groups.
        """
        m = adata.obs[self.group_key].value_counts().min() // self.seq_len
        ids = np.arange(len(adata))

        assert (
            m > 0
        ), f"Sequence length {self.seq_len=} is larger than number of samples: {m=}"

        all_partitions = []
        for _, d in adata.obs.groupby(self.group_key):
            group_mask = np.isin(adata.obs.index, d.index)
            group_idxs = ids[group_mask]
            n_partitions = len(d) // self.seq_len
            partitions = np.array_split(np.random.permutation(group_idxs), n_partitions)
            partitions = [
                torch.from_numpy(p).long() for p in partitions if len(p) == self.seq_len
            ]
            all_partitions.extend(partitions)

        return all_partitions
