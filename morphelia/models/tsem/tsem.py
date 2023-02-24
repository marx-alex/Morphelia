from typing import Optional, Union

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
import numpy as np
import anndata as ad

from ..modules import Encoder
from .tsem_modules import TSEMModule
from ..base import BaseModel

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class tsEM(BaseModel):
    """Morphelia class for the time series evolutionary model."""

    def __init__(
        self,
        data: pl.LightningDataModule,
        in_features: Optional[int] = None,
        n_classes: Optional[int] = None,
        layer_dims: list = None,
        dropout: float = 0.1,
        batch_norm: bool = False,
        layer_norm: bool = True,
        latent_dim: int = 2,
        learning_rate: float = 1e-4,
        optimizer: str = "Adam",
    ):
        self.in_features = in_features
        if in_features is None:
            self.in_features = data.n_features
        self.n_classes = n_classes
        if n_classes is None:
            self.n_classes = data.n_classes
        self.n_conditions = data.n_conditions
        self.t_max = data.t_max
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
            n_conditions=self.n_conditions,
            dropout=dropout,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
        )

        self.model = TSEMModule(
            encoder=self.encoder,
            n_classes=self.n_classes,
            t_max=self.t_max,
            learning_rate=learning_rate,
            optimizer=optimizer,
        )

        self.default_wandb_kwargs = dict(
            project="tsEM",
            config=dict(
                n_features=self.in_features,
                layer_dims=layer_dims,
                latent_dim=latent_dim,
                n_classes=self.n_classes,
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
        self,
        ce_beta: float = 1.0,
        mse_beta: float = 1.0,
        wandb_log: bool = True,
        wandb_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        self.model.ce_beta = ce_beta
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
        pred = []
        classes = []
        ids = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Load samples from test set."):
                x = batch["x"]
                c = None
                if "c" in batch:
                    c = batch["c"]

                features.append(
                    self.model.forward_features(x, c=c).cpu().detach().numpy()
                )

                pred.append(self.model.forward_pred(x, c=c).cpu().detach().numpy())

                classes.append(batch["target"].numpy())
                ids.append(batch["ids"].numpy())
        self.model.train()

        features = np.concatenate(features, axis=0)
        pred = np.concatenate(pred, axis=0)
        classes = np.concatenate(classes, axis=0).flatten()
        ids = np.concatenate(ids, axis=0).flatten()

        if adata is not None:
            adata = adata[ids, :].copy()
            adata.obsm["X_tsem"] = features
            adata.obs["prediction"] = np.argmax(pred, axis=1)
            for cl in range(pred.shape[1]):
                adata.obs[f"class_prob_{cl}"] = pred[:, cl]
            return adata

        else:
            return dict(ids=ids, classes=classes, pred=pred, features=features)
