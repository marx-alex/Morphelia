from typing import List, Union, Optional

import anndata as ad
from anndata.experimental import AnnLoader
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import LineageTreeDataset, TSDataset, SequenceDataset
from .collate import collate_concat, AdataCollator
from morphelia.tools import group_shuffle_split, train_test_split

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        adata: Union[str, ad.AnnData],
        batch_size: int = 32,
        num_workers: int = 0,
        train_dataloader_opts: Optional[dict] = None,
        valid_dataloader_opts: Optional[dict] = None,
    ):
        """
        Base class for data loading.

        Args:
            adata: Path to anndata object or anndata object.
            batch_size: Size of mini batches.
            num_workers: Number of subprocesses.
            train_dataloader_opts: Additional arguments for training dataloader.
            valid_dataloader_opts: Additional arguments for validation dataloader.
        """
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

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
            "shuffle": False,
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

        self.n_features = adata.n_vars


class LineageTreeDataModule(BaseDataModule):
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
        super().__init__(
            adata=adata,
            batch_size=batch_size,
            num_workers=num_workers,
            train_dataloader_opts=train_dataloader_opts,
            valid_dataloader_opts=valid_dataloader_opts,
        )

        self.seq_len = seq_len

        self.class_labels = adata.obs[target_key].unique()
        self.n_classes = len(self.class_labels)

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


class TSDataModule(BaseDataModule):
    def __init__(
        self,
        adata: Union[str, ad.AnnData],
        target_key: str = "Metadata_Treatment",
        track_key: str = "Metadata_Track",
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
        Data Loading of time series data with sampling from single tracks.

        Args:
            adata: Path to anndata object or anndata object.
            target_key: Key in .obs with targets.
            track_key: Key in .obs with tracks.
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
        super().__init__(
            adata=adata,
            batch_size=batch_size,
            num_workers=num_workers,
            train_dataloader_opts=train_dataloader_opts,
            valid_dataloader_opts=valid_dataloader_opts,
        )

        self.seq_len = seq_len

        self.class_labels = adata.obs[target_key].unique()
        self.n_classes = len(self.class_labels)

        if condition_key is not None:
            self.n_conditions = len(adata.obs[condition_key].unique())
        else:
            self.n_conditions = 0
        if seq_len == "max":
            self.seq_len = len(adata.obs[time_key].unique())
        else:
            self.seq_len = seq_len

        train_val, self.test = group_shuffle_split(
            adata, group=track_key, test_size=test_size
        )
        self.train, self.valid = group_shuffle_split(
            train_val, group=track_key, test_size=val_size
        )

        self.track_key = track_key
        self.condition_key = condition_key
        self.time_key = time_key
        self.target_key = target_key

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            TSDataset(
                self.train.copy(),
                target_key=self.target_key,
                track_key=self.track_key,
                time_key=self.time_key,
                condition_key=self.condition_key,
                seq_len=self.seq_len,
            ),
            **self.train_dataloader_opts,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            TSDataset(
                self.valid.copy(),
                target_key=self.target_key,
                track_key=self.track_key,
                time_key=self.time_key,
                condition_key=self.condition_key,
                seq_len=self.seq_len,
            ),
            **self.valid_dataloader_opts,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            TSDataset(
                self.test.copy(),
                target_key=self.target_key,
                track_key=self.track_key,
                time_key=self.time_key,
                condition_key=self.condition_key,
                seq_len=self.seq_len,
            ),
            **self.valid_dataloader_opts,
        )


class SequenceDataModule(BaseDataModule):
    def __init__(
        self,
        adata: Union[str, ad.AnnData],
        target_key: str = "Metadata_Treatment",
        group_key: str = "Metadata_Treatment",
        condition_key: str = None,
        batch_size: int = 32,
        num_workers: int = 0,
        test_size: float = 0.2,
        val_size: float = 0.2,
        train_dataloader_opts: dict = None,
        valid_dataloader_opts: dict = None,
        seq_len: int = 5,
    ):
        """
        Data Loading as sequences.

        Args:
            adata: Path to anndata object or anndata object.
            target_key: Key in .obs with targets.
            group_key: Key in .obs with groups to sample from.
            condition_key: Key in .obs with batch.
            batch_size: Size of mini batches.
            num_workers: Number of subprocesses.
            test_size: Size for test set.
            val_size: Size of validation set.
            train_dataloader_opts: Additional arguments for training dataloader.
            valid_dataloader_opts: Additional arguments for validation dataloader.
            seq_len: Lengths of sequences.
        """
        super().__init__(
            adata=adata,
            batch_size=batch_size,
            num_workers=num_workers,
            train_dataloader_opts=train_dataloader_opts,
            valid_dataloader_opts=valid_dataloader_opts,
        )

        self.seq_len = seq_len
        self.class_labels = adata.obs[target_key].unique()
        self.n_classes = len(self.class_labels)

        if condition_key is not None:
            self.n_conditions = len(adata.obs[condition_key].unique())
        else:
            self.n_conditions = 0

        train_val, self.test = train_test_split(adata, test_size=test_size)
        self.train, self.valid = train_test_split(train_val, test_size=val_size)

        self.group_key = group_key
        self.condition_key = condition_key
        self.target_key = target_key

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            SequenceDataset(
                self.train.copy(),
                target_key=self.target_key,
                group_key=self.group_key,
                condition_key=self.condition_key,
                seq_len=self.seq_len,
            ),
            **self.train_dataloader_opts,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            SequenceDataset(
                self.valid.copy(),
                target_key=self.target_key,
                group_key=self.group_key,
                condition_key=self.condition_key,
                seq_len=self.seq_len,
            ),
            **self.valid_dataloader_opts,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            SequenceDataset(
                self.test.copy(),
                target_key=self.target_key,
                group_key=self.group_key,
                condition_key=self.condition_key,
                seq_len=self.seq_len,
            ),
            **self.valid_dataloader_opts,
        )


class AnnDataModule(BaseDataModule):
    def __init__(
        self,
        adata: Union[str, ad.AnnData],
        target_key: str = "Metadata_Treatment",
        condition_key: Optional[str] = None,
        time_key: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        test_size: float = 0.2,
        val_size: float = 0.2,
        split_by: Optional[str] = None,
        split_by_groups: Optional[dict] = None,
        train_dataloader_opts: dict = None,
        valid_dataloader_opts: dict = None,
    ):
        """
        Data Loading of AnnDataViews.

        Args:
            adata: Path to anndata object or anndata object.
            target_key: Key in .obs with targets.
            condition_key: Key in .obs with batch.
            batch_size: Size of mini batches.
            num_workers: Number of subprocesses.
            test_size: Size for test set.
            val_size: Size of validation set.
            split_by: A variable in .obs that should be used to split data
                into train, validation and test set.
            split_by_groups: If `split_by` is not None, a dictionary with `train`,
            `valid` and `test` as keys should indicate the labels per group.
            train_dataloader_opts: Additional arguments for training dataloader.
            valid_dataloader_opts: Additional arguments for validation dataloader.
        """
        collator = AdataCollator(
            target_key=target_key, condition_key=condition_key, time_key=time_key
        )
        if train_dataloader_opts is None:
            train_dataloader_opts = dict(collate_fn=collator)
        elif "collate_fn" not in train_dataloader_opts:
            train_dataloader_opts.update(dict(collate_fn=collator))
        if valid_dataloader_opts is None:
            valid_dataloader_opts = dict(collate_fn=collator)
        elif "collate_fn" not in valid_dataloader_opts:
            valid_dataloader_opts.update(dict(collate_fn=collator))

        super().__init__(
            adata=adata,
            batch_size=batch_size,
            num_workers=num_workers,
            train_dataloader_opts=train_dataloader_opts,
            valid_dataloader_opts=valid_dataloader_opts,
        )

        self.class_labels = adata.obs[target_key].unique()
        self.n_classes = len(self.class_labels)

        if condition_key is not None:
            self.n_conditions = len(adata.obs[condition_key].unique())
        else:
            self.n_conditions = 0

        if time_key is not None:
            self.t_max = adata.obs[time_key].max()
        else:
            self.t_max = None

        if split_by is None:
            train_val, self.test = train_test_split(adata, test_size=test_size)
            self.train, self.valid = train_test_split(train_val, test_size=val_size)
        else:
            assert split_by in adata.obs, f"split_by is not in .obs: {split_by}"
            assert isinstance(
                split_by_groups, dict
            ), f"split_by_groups should be of type dict, instead got {type(split_by_groups)}"
            self.train, self.valid, self.test = None, None, None
            if "train" in split_by_groups:
                self.train = adata[
                    adata.obs[split_by].isin(split_by_groups["train"]), :
                ].copy()
            if "valid" in split_by_groups:
                self.valid = adata[
                    adata.obs[split_by].isin(split_by_groups["valid"]), :
                ].copy()
            if "test" in split_by_groups:
                self.test = adata[
                    adata.obs[split_by].isin(split_by_groups["test"]), :
                ].copy()

        self.condition_key = condition_key
        self.target_key = target_key

    def train_dataloader(self) -> AnnLoader:
        return AnnLoader(self.train, **self.train_dataloader_opts)

    def val_dataloader(self) -> AnnLoader:
        return AnnLoader(self.valid, **self.valid_dataloader_opts)

    def test_dataloader(self) -> AnnLoader:
        return AnnLoader(self.test, **self.valid_dataloader_opts)
