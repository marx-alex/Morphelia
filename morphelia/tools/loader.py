import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl


class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data, return_index=True):
        self.return_index = return_index
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        if self.return_index:
            return self.X_data[index], self.y_data[index], index
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 X_train: np.ndarray = None, y_train: np.ndarray = None,
                 X_test: np.ndarray = None, y_test: np.ndarray = None,
                 X_val: np.ndarray = None, y_val: np.ndarray = None,
                 weighted=False,
                 batch_size: int = 32,
                 num_workers=0):
        """
        Data Module.

        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :param X_val:
        :param y_val:
        :param batch_size:
        :param num_workers:
        """
        super().__init__()

        self.train_dataset = None
        if X_train is not None:
            self.train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(),
                                                   torch.from_numpy(y_train).long())

        self.test_dataset = None
        if X_test is not None:
            self.test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(),
                                                  torch.from_numpy(y_test).long())

        self.val_dataset = None
        if X_val is not None:
            self.val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(),
                                                 torch.from_numpy(y_val).long())
        self.weighted = weighted
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_shape = X_train.shape[-1]
        self.n_classes = len(np.unique(y_train))

    @staticmethod
    def weighted_sampler(y):
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        values, counts = np.unique(y, return_counts=True)
        weight = 1 / counts
        samples_weight = np.array([weight[v] for v in y])

        samples_weight = torch.from_numpy(samples_weight).double()
        return WeightedRandomSampler(samples_weight, len(samples_weight))

    def train_dataloader(self):
        sampler = None
        shuffle = True
        if self.weighted:
            sampler = self.weighted_sampler(self.train_dataset.y_data)
            shuffle = False
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=shuffle,
                          sampler=sampler)

    def val_dataloader(self):
        sampler = None
        if self.weighted:
            sampler = self.weighted_sampler(self.val_dataset.y_data)
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=sampler,
                          shuffle=False)

    def test_dataloader(self):
        sampler = None
        if self.weighted:
            sampler = self.weighted_sampler(self.test_dataset.y_data)
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=sampler,
                          shuffle=False)

    def predict_dataloader(self):
        sampler = None
        if self.weighted:
            sampler = self.weighted_sampler(self.test_dataset.y_data)
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=sampler,
                          shuffle=False)