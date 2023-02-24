from typing import Optional, Union

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
import numpy as np
import anndata as ad

from ..modules import Encoder
from .deepmap_modules import DEEPMAP
from ..base import BaseModel

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class DeepMap(BaseModel):
    """Morphelia class for the DeepMap model.

    This is a convenience class for model initialization,
    data loading, training and predicting morphological data.
    The DeepMap model models image-based profiles from
    single cells using a combined clustering and classification layer.
    The clustering layer initializes cluster centers that are learned during
    training and classification works by distances to cluster centers.

    Parameters
    ----------
    data : pytorch_lightning.LightningDataModule
        This data module should have a train, validation and
        test loader that yield batches of sequential data
    in_features : int, optional
        Number of input dimensions
    n_classes : int, optional
        Number of classes
    cdist_act : str, optional
        Optional activation function after calculating cluster
        center distances. Can be `tanhshrink`, `tanh`, `relu`, `gelu`, `sigmoid`.
    layer_dims : list, optional
        Dimensions of hidden layers. The length of
        the sequences it equal to the number of hidden layers.
    dropout : float
        Dropout rate
    batch_norm : bool
        Include batch normalization after every encoder layer
    layer_norm : bool
        Include layer normalization after every encoder layer
    latent_dim : int
        Dimensions of the latent space
    learning_rate : float
        Learning rate during training
    optimizer : str
        Optimizer for the training process.
        Can be `Adam` or `AdamW`.

    References
    ----------
    .. [1] Ren et al., 2021, bioRxiv
    """

    def __init__(
        self,
        data: pl.LightningDataModule,
        in_features: Optional[int] = None,
        n_classes: Optional[int] = None,
        cdist_act: Optional[str] = None,
        layer_dims: Optional[list] = None,
        dropout: float = 0.1,
        batch_norm: bool = False,
        layer_norm: bool = True,
        latent_dim: int = 2,
        learning_rate: float = 1e-4,
        optimizer: str = "Adam",
    ) -> None:
        self.in_features = in_features
        if in_features is None:
            self.in_features = data.n_features
        self.n_classes = n_classes
        if n_classes is None:
            self.n_classes = data.n_classes
        self.data = data

        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        if layer_dims is None:
            layer_dims = [128, 64, 32]
        layer_dims = [self.in_features] + layer_dims
        self.layer_dims = layer_dims

        self.encoder = Encoder(
            layer_dims=layer_dims,
            latent_dim=latent_dim,
            sequential=False,
            dropout=dropout,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
        )

        self.model = DEEPMAP(
            encoder=self.encoder,
            n_classes=self.n_classes,
            cdist_act=cdist_act,
            learning_rate=learning_rate,
            optimizer=optimizer,
        )

        self.default_wandb_kwargs = dict(
            project="Deep-Map",
            config=dict(
                n_features=self.in_features,
                layer_dims=layer_dims,
                latent_dim=latent_dim,
                n_classes=self.n_classes,
                dropout=dropout,
                batch_norm=batch_norm,
                layer_norm=layer_norm,
                cdist_act=cdist_act,
                learning_rate=learning_rate,
                optimizer=optimizer,
            ),
            save_code=True,
            save_dir="logs",
            log_model=False,
        )

        self.default_trainer_kwargs = dict(max_epochs=1000, log_every_n_steps=1)

        self.callbacks = [
            EarlyStopping(monitor="valid/F1", min_delta=0.005, patience=50, mode="max"),
            ModelCheckpoint(
                filename="{epoch}_best_f1",
                monitor="valid/F1",
                save_top_k=3,
                mode="max",
            ),
            ModelCheckpoint(
                filename="{epoch}_best_loss",
                monitor="valid/loss",
                save_top_k=3,
                mode="min",
            ),
        ]

    def train(
        self, wandb_log: bool = True, wandb_kwargs: Optional[dict] = None, **kwargs
    ) -> None:
        """Training method.

        Parameters
        ----------
        wandb_log : bool
            Log training to Weights and Biases
        wandb_kwargs : dict, optional
            Keyword arguments for the logger
        **kwargs
            Keyword arguments passed to pytorch_lightning.Trainer
        """
        if wandb_log:
            default_wandb_kwargs = self.default_wandb_kwargs.copy()
            if wandb_kwargs is not None:
                default_wandb_kwargs.update(wandb_kwargs)

            logger = WandbLogger(**default_wandb_kwargs)
        else:
            logger = False

        default_trainer_kwargs = self.default_trainer_kwargs.copy()
        default_trainer_kwargs.update(dict(logger=logger, callbacks=self.callbacks))
        default_trainer_kwargs.update(kwargs)

        trainer = pl.Trainer(**default_trainer_kwargs)
        trainer.fit(self.model, self.data)

        if wandb_log:
            logger.experiment.finish()

    def load_from_checkpoints(self, ckpt: str) -> None:
        """Load model from checkpoints.

        Parameters
        ----------
        ckpt : str
            Path to checkpoint file
        """
        self.model = self.model.load_from_checkpoint(ckpt)

    def get_latent(self, loader: str = "test") -> Union[ad.AnnData, dict]:
        """Load latent representation.

        Latent features and predictions can be loaded
        from the train, validation or test loader.
        The latent embedding is returned as new embedding in
        the AnnData object if DataModule as an AnnData object stored.
        Otherwise a dictionary is returned.

        Parameters
        ----------
        loader : str
            Load latent representation of `train`,
            `valid` or `test` loader

        Returns
        -------
        anndata.AnnData or dict
            AnnData object with embedded features and predictions (in `.obs`)
            if an AnnData object is stored in the DataModule

        Raises
        -------
        NotImplementedError
            If loader in neither `train`, `valid` nor `test`
        """
        if loader == "test":
            loader = self.data.test_dataloader()
            adata = self.data.test
        elif loader == "valid":
            loader = self.data.val_dataloader()
            adata = self.data.valid
        elif loader == "train":
            loader = self.data.train_dataloader()
            adata = self.data.train
        else:
            raise NotImplementedError(
                f"loader must be 'test', 'train' or 'valid', instead got {loader}"
            )
        features = []
        pred = []
        classes = []
        ids = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Load samples from test set."):
                x = batch["x"]

                features.append(self.model.forward_features(x).cpu().detach().numpy())

                pred.append(self.model.forward_pred(x).cpu().detach().numpy())

                classes.append(batch["target"].numpy())
                ids.append(batch["ids"].numpy())
        self.model.train()

        features = np.concatenate(features, axis=0)
        pred = np.concatenate(pred, axis=0)
        classes = np.concatenate(classes, axis=0).flatten()
        ids = np.concatenate(ids, axis=0).flatten()

        if adata is not None:
            adata = adata[ids, :].copy()
            adata.obsm["X_deepmap"] = features
            adata.obs["prediction"] = np.argmax(pred, axis=1)
            for cl in range(pred.shape[1]):
                adata.obs[f"class_prob_{cl}"] = pred[:, cl]
            return adata

        else:
            return dict(ids=ids, classes=classes, pred=pred, features=features)
