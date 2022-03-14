from typing import List, Union

import anndata as ad
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import LineageTreeDataset
from .collate import collate_concat
from morphelia.tools import group_shuffle_split

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class TSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        adata: Union[str, ad.AnnData],
        target_key: str = "Metadata_Treatment",
        root_key: str = "Metadata_Track_Root",
        track_key: str = "Metadata_Track",
        parent_key: str = "Metadata_Track_Parent",
        time_key: str = "Metadata_Time",
        condition_key: str = None,
        batch_size: int = 32,
        num_workers: int = 0,
        test_size: float = 0.2,
        val_size: float = 0.2,
        train_dataloader_opts: dict = None,
        valid_dataloader_opts: dict = None,
        seq_len: Union[str, int] = "max",
    ):
        """
        Data Loading of time series data with sampling from lineage trees.

        Args:
            adata: Path to anndata object or anndata object.
            target_key: Key in .obs with targets.
            root_key: Key in .obs with roots.
            track_key: Key in .obs with tracks.
            parent_key: Key in .obs with parents.
            time_key: Key in .obs with time.
            condition_key: Key in .obs with batch.
            batch_size: Size of mini batches.
            num_workers: Number of subprocesses.
            test_size: Size for test set.
            val_size: Size of validation set.
            train_dataloader_opts: Additional arguments for training dataloader.
            valid_dataloader_opts: Additional arguments for validation dataloader.
            seq_len: Lengths of sequences.
                If a sequence is smaller than ts_len, the sequence is padded with zeros.
                If a sequence is longer than ts_len, the sequence is shortened.
        """
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_len = seq_len

        # defining options for data loaders
        self.train_dataloader_opts = {
            "batch_size": self.batch_size,
            "num_workers": num_workers,
            "shuffle": True,
            "collate_fn": collate_concat,
        }

        self.valid_dataloader_opts = {
            "batch_size": self.batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_concat,
        }

        if train_dataloader_opts is not None:
            self.train_dataloader_opts.update(train_dataloader_opts)

        if valid_dataloader_opts is not None:
            self.valid_dataloader_opts.update(valid_dataloader_opts)

        if isinstance(adata, str):
            adata = ad.read_h5ad(adata)

        # convert string obs to categories
        adata.strings_to_categoricals()

        self.class_labels = adata.obs[target_key].unique()
        self.n_classes = len(self.class_labels)
        self.n_features = adata.n_vars
        if condition_key is not None:
            self.n_conditions = len(adata.obs[condition_key].unique())
        else:
            self.n_conditions = 0
        if seq_len == "max":
            self.seq_len = len(adata.obs[time_key].unique())
        else:
            self.seq_len = seq_len

        train_val, self.test = group_shuffle_split(adata, root_key, test_size=test_size)
        self.train, self.valid = group_shuffle_split(
            train_val, root_key, test_size=val_size
        )

        self.root_key = root_key
        self.track_key = track_key
        self.parent_key = parent_key
        self.condition_key = condition_key
        self.time_key = time_key
        self.target_key = target_key

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            LineageTreeDataset(
                self.train.copy(),
                target_key=self.target_key,
                root_key=self.root_key,
                track_key=self.track_key,
                parent_key=self.parent_key,
                time_key=self.time_key,
                condition_key=self.condition_key,
                seq_len=self.seq_len,
            ),
            **self.train_dataloader_opts,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            LineageTreeDataset(
                self.valid.copy(),
                target_key=self.target_key,
                root_key=self.root_key,
                track_key=self.track_key,
                parent_key=self.parent_key,
                time_key=self.time_key,
                condition_key=self.condition_key,
                seq_len=self.seq_len,
            ),
            **self.valid_dataloader_opts,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            LineageTreeDataset(
                self.test.copy(),
                target_key=self.target_key,
                root_key=self.root_key,
                track_key=self.track_key,
                parent_key=self.parent_key,
                time_key=self.time_key,
                condition_key=self.condition_key,
                seq_len=self.seq_len,
            ),
            **self.valid_dataloader_opts,
        )
