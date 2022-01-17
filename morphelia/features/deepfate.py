import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import pytorch_lightning as pl

from typing import Optional
import logging
import numpy as np

from morphelia.tools import choose_representation

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ClusterAssignment(nn.Module):
    def __init__(
        self,
        n_classes: int,
        enc_shape: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used to measure similarity between feature vector and each
        cluster centroid.
        :param n_classes: number of clusters
        :param enc_shape: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super().__init__()
        self.enc_shape = enc_shape
        self.n_classes = n_classes
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_classes, self.enc_shape, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.
        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class ClusterDistance(nn.Module):
    def __init__(
        self,
        n_classes: int,
        enc_shape: int,
        cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        """

        :param n_classes: number of clusters
        :param enc_shape: embedding dimension of feature vectors
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super().__init__()
        self.enc_shape = enc_shape
        self.n_classes = n_classes
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_classes, self.enc_shape, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: FloatTensor of [batch size, embedding dimension]
        :param y: FloatTensor of [batch size,]
        :return: FloatTensor [batch size, number of clusters]
        """

        return torch.cdist(x, self.cluster_centers)


class CombinedAutoencoder(pl.LightningModule):
    """Autoencoder.
    Augmentation of DeepMap as described in Ren et al., 2021, bioRxiv
    by an decoding layer to better represent original feature space.

    Args:
    in_shape (int): input shape
    enc_shape (int): desired encoded shape
    """

    def __init__(self, in_shape, enc_shape, n_classes, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.enc_shape = enc_shape
        self.n_classes = n_classes

        self.encode = nn.Sequential(
            nn.Linear(in_shape, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Linear(16, self.enc_shape),
        )

        self.decode = nn.Sequential(
            nn.Linear(enc_shape, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, in_shape)
        )

        self.cluster = nn.Sequential(
            ClusterDistance(self.n_classes, self.enc_shape),
            nn.Tanhshrink(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        z = self.encode(x)
        out = self.cluster(z)
        x_hat = self.decode(z)

        return x_hat, out

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = self._prepare_batch(batch)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _prepare_batch(self, batch):
        x, y = batch
        return x.view(x.size(0), -1), y

    def _common_step(self, batch, batch_idx, stage: str):
        x, y = self._prepare_batch(batch)
        x_hat, out = self(x)
        loss1 = F.cross_entropy(out, y)
        loss2 = F.mse_loss(x_hat, x)
        loss = loss1 + loss2

        self.log(f"{stage}_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss


class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class DataModule(pl.LightningDataModule):
    def __init__(self, X_data: np.ndarray, y_data: np.ndarray,
                 groups=None, batch_size: int = 32,
                 num_workers=0, test_size=0.33, val_size=0.2):
        super().__init__()

        ix_data = np.arange(X_data.shape[0])

        if groups is None:
            X_trainval, X_test, y_trainval, y_test, ix_trainval, ix_test = train_test_split(X_data, y_data,
                                                                                            ix_data,
                                                                                            test_size=test_size,
                                                                                            stratify=y_data,
                                                                                            random_state=0)

            X_train, X_val, y_train, y_val, ix_train, ix_val = train_test_split(X_trainval, y_trainval,
                                                                                ix_trainval,
                                                                                test_size=val_size, stratify=y_trainval,
                                                                                random_state=0)

        else:
            gss_test = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=0)
            ix_trainval, ix_test = [], []
            for ix_tv, ix_t in gss_test.split(X_data, y_data, groups):
                ix_trainval.append(ix_tv)
                ix_test.append(ix_t)
            ix_trainval = ix_trainval[0]
            ix_test = ix_test[0]
            X_trainval = X_data[ix_trainval, :]
            y_trainval = y_data[ix_trainval]
            groups_trainval = groups[ix_trainval]
            X_test = X_data[ix_test, :]
            y_test = y_data[ix_trainval]

            gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
            ix_train, ix_val = [], []
            for ix_t, ix_v in gss_val.split(X_trainval, y_trainval, groups_trainval):
                ix_train.append(ix_t)
                ix_val.append(ix_v)
            ix_train = ix_train[0]
            ix_val = ix_val[0]
            X_train = X_trainval[ix_train, :]
            y_train = y_trainval[ix_train]
            X_val = X_trainval[ix_val, :]
            y_val = y_trainval[ix_val]

        self.ix_data = ix_data
        self.ix_trainval = ix_trainval
        self.ix_test = ix_test
        self.ix_train = ix_train
        self.ix_val = ix_val
        self.test_dataset = ClassifierDataset(torch.from_numpy(X_test).double(),
                                              torch.from_numpy(y_test).long())
        self.train_dataset = ClassifierDataset(torch.from_numpy(X_train).double(),
                                               torch.from_numpy(y_train).long())
        self.val_dataset = ClassifierDataset(torch.from_numpy(X_val).double(),
                                             torch.from_numpy(y_val).long())
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class DeepFate:
    """Dimensionality reduction with deep embedding.
    """

    def __init__(self,
                 n_epochs=500,
                 enc_dims=2,
                 lr=1e-3,
                 num_workers=0,
                 batch_size=32):
        self.n_epochs = n_epochs
        self.enc_dims = enc_dims
        self.lr = lr
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.model = CombinedAutoencoder
        self.data = None

    def fit(self,
            adata,
            y_label='Metadata_Treatment_Enc',
            groups=None,
            use_rep=None,
            n_pcs=50,
            test_size=0.33,
            val_size=0.2):
        """
        Fit model.

        Args:
            adata (anndata.AnnData): Multidimensional morphological data.
            y_label (str): Variable name for labels in .obs.
            groups (str): Variable for groups that should not be split during train-test-split.
            use_rep (bool): Make representation of data 3d
            n_pcs (int): Number of PCs to use if use_rep is "X_pca"
            test_size (float): Size of test data.
            val_size (float): Size of validation data.
        """
        # get representation of data
        if use_rep is None:
            use_rep = 'X'
        X = choose_representation(adata,
                                  rep=use_rep,
                                  n_pcs=n_pcs)

        in_shape = X.shape[-1]
        assert y_label in adata.obs.columns, f"y_label not in .obs: {y_label}"
        y = adata.obs[y_label].to_numpy().flatten()
        n_classes = len(np.unique(y))

        if groups is not None:
            assert groups in adata.obs.columns, f"groups is not in .obs:, {groups}"
            groups = adata.obs[groups].to_numpy().flatten()

        self.data = DataModule(X_data=X, y_data=y,
                               num_workers=self.num_workers, batch_size=self.batch_size,
                               groups=groups,
                               test_size=test_size, val_size=val_size)
        train_loader = self.data.train_dataloader()
        valid_loader = self.data.val_dataloader()
        self.model = CombinedAutoencoder(in_shape=in_shape, enc_shape=self.enc_dims,
                                         n_classes=n_classes, lr=self.lr).double()
        trainer = pl.Trainer(max_epochs=self.n_epochs)
        trainer.fit(self.model, train_loader, valid_loader)

    def predict(self,
                adata=None):
        """
        Predict embedding on stored test data.
        Return new adata object if give.
        """
        X_test = self.data.test_dataset.X_data
        ix_test = self.data.ix_test

        with torch.no_grad():
            encoded = self.model.encode(X_test)
            enc = encoded.cpu().detach().numpy()

        if adata is not None:
            adata = adata[ix_test, :].copy()
            adata.obsm['X_nne'] = enc

            return adata

        else:
            return enc

    def save_model(self, path="./"):
        torch.save(self.model, path)

    def load_model(self, path):
        self.model = torch.load(path)

