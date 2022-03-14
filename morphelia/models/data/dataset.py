from collections import defaultdict
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import anndata as ad

from ._utils import data_converter


class LineageTreeDataset(Dataset):
    """
    Dataset handler for tracked lineage trees.

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
            self.conditions = data_converter(adata.obs[condition_key]).long()
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
