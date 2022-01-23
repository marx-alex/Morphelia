import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray = None, y_test: np.ndarray = None,
                 X_val: np.ndarray = None, y_val: np.ndarray = None,
                 batch_size: int = 32,
                 num_workers=0):
        super().__init__()

        self.train_dataset = ClassifierDataset(torch.from_numpy(X_train).double(),
                                               torch.from_numpy(y_train).long())

        self.test_dataset = None
        if X_test is not None:
            self.test_dataset = ClassifierDataset(torch.from_numpy(X_test).double(),
                                                  torch.from_numpy(y_test).long())

        self.val_dataset = None
        if X_val is not None:
            self.val_dataset = ClassifierDataset(torch.from_numpy(X_val).double(),
                                                 torch.from_numpy(y_val).long())
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_shape = X_train.shape[-1]
        self.n_classes = len(np.unique(y_train))

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)