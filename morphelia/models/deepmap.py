import anndata
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Optional
import logging

from morphelia.tools import choose_representation, DataModule

logger = logging.getLogger()
logger.setLevel(logging.INFO)


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


class DeepMap(pl.LightningModule):
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

        # self.decode = nn.Sequential(
        #     nn.Linear(enc_shape, 16),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(True),
        #     nn.Linear(16, 32),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(True),
        #     nn.Linear(32, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(True),
        #     nn.Linear(128, in_shape)
        # )

        self.cluster = nn.Sequential(
            ClusterDistance(self.n_classes, self.enc_shape),
            nn.Tanhshrink(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        z = self.encode(x)
        out = self.cluster(z)
        # x_hat = self.decode(z)

        return out

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
        out = self(x)
        loss = F.cross_entropy(out, y)
        # loss2 = F.mse_loss(x_hat, x)
        # loss = loss1 + loss2

        self.log(f"{stage}_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss


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
        self.use_rep = None
        self.n_pcs = None

        self.model = DeepMap

    def fit(self,
            adata_train,
            adata_val=None,
            y_label='Metadata_Treatment_Enc',
            use_rep='X',
            n_pcs=50):
        """
        Fit model.

        Args:
            adata_train (anndata.AnnData): Multidimensional morphological data.
            adata_val (anndata.AnnData): Multidimensional morphological data.
            y_label (str): Variable name for labels in .obs.
            use_rep (str): Make representation of data 3d
            n_pcs (int): Number of PCs to use if use_rep is "X_pca"
        """
        # store parameters
        self.use_rep = use_rep
        self.n_pcs = n_pcs

        assert y_label in adata_train.obs.columns, f"y_label not in .obs: {y_label}"
        y_train = adata_train.obs[y_label].to_numpy().flatten().copy()
        X_train = choose_representation(adata_train,
                                        rep=self.use_rep,
                                        n_pcs=self.n_pcs)

        X_val = None
        y_val = None
        if adata_val is not None:
            y_val = adata_val.obs[y_label].to_numpy().flatten().copy()
            X_val = choose_representation(adata_val,
                                          rep=self.use_rep,
                                          n_pcs=self.n_pcs)

        data = DataModule(X_train=X_train, y_train=y_train,
                          X_val=X_val, y_val=y_val,
                          num_workers=self.num_workers,
                          batch_size=self.batch_size)

        in_shape = data.in_shape
        n_classes = data.n_classes

        train_loader = data.train_dataloader()
        valid_loader = data.val_dataloader()
        self.model = DeepMap(in_shape=in_shape, enc_shape=self.enc_dims,
                             n_classes=n_classes, lr=self.lr).double()
        trainer = pl.Trainer(max_epochs=self.n_epochs, auto_lr_find=True)
        trainer.fit(self.model, train_loader, valid_loader)

    def embed(self,
              adata,
              model=None):
        """
        Predict embedding on stored test data.
        Return new adata object if give.

        Args:
            adata (anndata.AnnData): Multidimensional morphological data. Test set.
            model (pytorch.nn.Module): Trained module to use.
        """
        X_test = choose_representation(adata,
                                       rep=self.use_rep,
                                       n_pcs=self.n_pcs)

        X_test = torch.from_numpy(X_test).double()

        if model is not None:
            self.model = model

        with torch.no_grad():
            encoded = self.model.encode(X_test)
            enc = encoded.cpu().detach().numpy()

        adata.obsm['X_nne'] = enc

        return adata

    def save_model(self, path="./"):
        torch.save(self.model, path)

    def load_model(self, path):
        self.model = torch.load(path)

