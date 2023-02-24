from typing import Optional, Union

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
import numpy as np
import anndata as ad

from ..modules import VAEEncoder, MMDDecoder
from .trvae_modules import TRVAE
from ..base import BaseModel

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class trVAE(BaseModel):
    """Morphelia class for the trVAE model.

    This is a convenience class for model initialization,
    data loading, training and predicting morphological data.
    The trVAE model models image-based profiles from
    single cells using a conditional variaitonal autoencoder.

    Parameters
    ----------
    data : pytorch_lightning.LightningDataModule
        This data module should have a train, validation and
        test loader that yield batches of sequential data
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
    .. [1] Lotfollahi et al., 2020, Bioinformatics
    """

    def __init__(
        self,
        data: pl.LightningDataModule,
        layer_dims: list = None,
        latent_dim: int = 10,
        dropout: float = 0.1,
        batch_norm: bool = True,
        layer_norm: bool = False,
        learning_rate: float = 1e-4,
        optimizer: str = "Adam",
    ):

        self.in_features = data.n_features
        self.n_conditions = data.n_conditions
        self.data = data

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        if layer_dims is None:
            layer_dims = [128, 64, 32]

        self.layer_dims = layer_dims
        self.latent_dim = latent_dim
        encoder_layer_dims = self.layer_dims.copy()
        encoder_layer_dims.insert(0, self.in_features)
        decoder_layer_dims = encoder_layer_dims.copy()
        decoder_layer_dims.reverse()

        self.encoder = VAEEncoder(
            layer_dims=layer_dims,
            latent_dim=latent_dim,
            n_conditions=self.n_conditions,
            sequential=False,
            dropout=dropout,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
        )

        self.decoder = MMDDecoder(
            layer_dims=decoder_layer_dims,
            latent_dim=self.latent_dim,
            n_conditions=self.n_conditions,
            sequential=True,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            dropout=dropout,
        )

        self.model = TRVAE(
            encoder=self.encoder,
            decoder=self.decoder,
            learning_rate=learning_rate,
            optimizer=optimizer,
        )

        self.default_wandb_kwargs = dict(
            project="trVAE",
            config=dict(
                n_features=self.in_features,
                layer_dims=layer_dims,
                latent_dim=latent_dim,
                dropout=dropout,
                batch_norm=batch_norm,
                layer_norm=layer_norm,
                learning_rate=learning_rate,
                optimizer=optimizer,
            ),
            save_code=True,
            save_dir="logs",
            log_model=False,
        )

        self.default_trainer_kwargs = dict(max_epochs=1000, log_every_n_steps=1)

        self.callbacks = [
            EarlyStopping(
                monitor="valid/loss", min_delta=0.005, patience=50, mode="max"
            ),
            ModelCheckpoint(
                filename="{epoch}_best_loss",
                monitor="valid/loss",
                save_top_k=3,
                mode="min",
            ),
        ]

    def train(
        self,
        kld_beta: float = 1.0,
        mmd_beta: float = 1.0,
        mse_beta: float = 1.0,
        wandb_log: bool = True,
        wandb_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        # update model parameters
        self.model.kld_beta = kld_beta
        self.model.mmd_beta = mmd_beta
        self.model.mse_beta = mse_beta

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
        self.model = self.model.load_from_checkpoint(ckpt)

    def get_latent(self, loader: str = "test") -> Union[ad.AnnData, dict]:
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
        ids = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Load samples from test set."):
                x = batch["x"]

                features.append(self.model.forward_features(x).cpu().detach().numpy())

                ids.append(batch["ids"].numpy())
        self.model.train()

        features = np.concatenate(features, axis=0)
        ids = np.concatenate(ids, axis=0).flatten()

        if adata is not None:
            adata = adata[ids, :].copy()
            adata.obsm["X_trvae"] = features
            return adata

        else:
            return dict(ids=ids, features=features)
