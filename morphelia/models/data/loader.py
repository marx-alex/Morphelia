from typing import List, Union

import anndata as ad
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import LineageTreeDataset
from .collate import collate_concat
from morphelia.tools import group_shuffle_split


class TSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        target_key: str = "Metadata_Treatment",
        root_key: str = "Metadata_Track_Root",
        track_key: str = "Metadata_Track",
        parent_key: str = "Metadata_Track_Parent",
        time_key: str = "Metadata_Time",
        batch_key: str = None,
        batch_size: int = 32,
        num_workers: int = 0,
        test_size: float = 0.2,
        val_size: float = 0.2,
        train_dataloader_opts: dict = None,
        valid_dataloader_opts: dict = None,
        conc_filter: dict = None,
        seq_len="max",
        mask=False,
        masking_len=3,
        masking_ratio=0.15,
    ):
        """
        Data Loading of time series data with sampling from lineage trees.

        Args:
            data_path (str): Path to anndata object.
            target_key (str): Key in .obs with targets.
            root_key (str): Key in .obs with roots.
            track_key (str): Key in .obs with tracks.
            parent_key (str): Key in .obs with parents.
            time_key (str): Key in .obs with time.
            batch_key (str): Key in .obs with batch.
            batch_size (int): Size of mini batches.
            num_workers (int): Number of subprocesses.
            test_size (float): Size for test set.
            val_size (float): Size of validation set.
            train_dataloader_opts (dict): Additional arguments for training dataloader.
            valid_dataloader_opts (dict): Additional arguments for validation dataloader.
            conc_filter (dict): Key in .obs with value to use to filter dataset.
            seq_len (str, int): Lengths of sequences.
                If a sequence is smaller than ts_len, the sequence is padded with zeros.
                If a sequence is longer than ts_len, the sequence is shortened.
            mask (bool): Mask data by geometric masking.
            masking_len (int): Average length of masks.
            masking_ratio (float): Average masking ratio for a feature.
        """
        super(TSDataModule).__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mask = mask
        self.masking_len = masking_len
        self.masking_ratio = masking_ratio
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

        adata = ad.read_h5ad(data_path)

        # filter concentration
        if conc_filter is not None:
            assert (
                len(conc_filter) == 1
            ), f"one key expected for conc_filter, got {len(conc_filter)}"
            conc_key = list(conc_filter.keys())[0]
            conc_val = conc_filter[conc_key]
            adata = adata[adata.obs[conc_key] == conc_val, :]

        train_val, self.test = group_shuffle_split(adata, root_key, test_size=test_size)
        self.train, self.valid = group_shuffle_split(
            train_val, root_key, test_size=val_size
        )

        self.root_key = root_key
        self.track_key = track_key
        self.parent_key = parent_key
        self.batch_key = batch_key
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
                batch_key=self.batch_key,
                seq_len=self.seq_len,
                mask=self.mask,
                masking_len=self.masking_len,
                masking_ratio=self.masking_ratio,
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
                batch_key=self.batch_key,
                seq_len=self.seq_len,
                mask=self.mask,
                masking_len=self.masking_len,
                masking_ratio=self.masking_ratio,
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
                batch_key=self.batch_key,
                seq_len=self.seq_len,
                mask=self.mask,
                masking_len=self.masking_len,
                masking_ratio=self.masking_ratio,
            ),
            **self.valid_dataloader_opts,
        )
