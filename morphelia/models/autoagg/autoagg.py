from typing import Optional, Union

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
import numpy as np
import anndata as ad

from ..modules import (
    Encoder,
    TransformerClassifier,
    TransformerTokenClassifier,
    MeanClassifier,
    MedianClassifier,
    MajorityClassifier,
)
from .autoagg_modules import AUTOAGG
from ..base import BaseModel

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class Autoagg(BaseModel):
    """Morphelia class for Autoaggregation."""

    def __init__(
        self,
        data: pl.LightningDataModule,
        encoder_layer_dims: list = None,
        latent_dim: int = 10,
        encoder_dropout: float = 0.1,
        classification_method: str = "transformer",
        nhead: int = 1,
        dim_feedforward: Optional[int] = None,
        transformer_dropout: float = 0.1,
        pos_dropout: float = 0.1,
        transformer_norm: str = "layer",
        pos_encoding: Optional[str] = "learnable",
        n_transformer_layers: int = 1,
        learning_rate: float = 1e-4,
        optimizer: str = "Adam",
    ):
        self.n_conditions = data.n_conditions
        self.in_features = data.n_features
        self.seq_len = data.seq_len
        self.n_classes = data.n_classes
        self.class_labels = data.class_labels
        self.batch_size = data.batch_size
        self.data = data

        self.learning_rate = learning_rate
        self.optimizer = optimizer

        if encoder_layer_dims is None:
            encoder_layer_dims = [128, 64, 32]
        encoder_layer_dims = [self.in_features] + encoder_layer_dims
        self.encoder_layer_dims = encoder_layer_dims

        self.encoder = Encoder(
            layer_dims=encoder_layer_dims,
            latent_dim=latent_dim,
            n_conditions=self.n_conditions,
            sequential=True,
            dropout=encoder_dropout,
            batch_norm=True,
            layer_norm=False,
        )

        classification_method = classification_method.lower()
        avail_methods = [
            "transformer",
            "transformer_token",
            "mean",
            "median",
            "majority",
        ]
        assert (
            classification_method in avail_methods
        ), f"'classification_method' must be one of {avail_methods}, instead got {classification_method}"

        if classification_method == "transformer":
            classifier = TransformerClassifier(
                input_dim=latent_dim,
                seq_len=self.seq_len,
                n_classes=self.n_classes,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=transformer_dropout,
                pos_dropout=pos_dropout,
                norm=transformer_norm,
                pos_encoding=pos_encoding,
                num_layers=n_transformer_layers,
            )
        elif classification_method == "transformer_token":
            classifier = TransformerTokenClassifier(
                input_dim=latent_dim,
                seq_len=self.seq_len,
                n_classes=self.n_classes,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=transformer_dropout,
                pos_dropout=pos_dropout,
                norm=transformer_norm,
                pos_encoding=pos_encoding,
                num_layers=n_transformer_layers,
            )
        elif classification_method == "mean":
            classifier = MeanClassifier(input_dim=latent_dim, n_classes=self.n_classes)
        elif classification_method == "median":
            classifier = MedianClassifier(
                input_dim=latent_dim, n_classes=self.n_classes
            )
        elif classification_method == "majority":
            classifier = MajorityClassifier(
                input_dim=latent_dim, n_classes=self.n_classes
            )
        else:
            raise NotImplementedError
        self.classifier = classifier

        self.model = AUTOAGG(
            encoder=self.encoder,
            classifier=self.classifier,
            n_classes=self.n_classes,
            learning_rate=learning_rate,
            optimizer=optimizer,
        )

        self.default_wandb_kwargs = dict(
            project="autoagg",
            config=dict(
                n_features=self.in_features,
                latent_dim=latent_dim,
                encoder_layer_dims=encoder_layer_dims,
                batch_size=self.batch_size,
                n_conditions=self.n_conditions,
                seq_len=self.seq_len,
                n_classes=self.n_classes,
                nhead=nhead,
                transformer_dropout=transformer_dropout,
                encoder_dropout=encoder_dropout,
                dim_feedforward=dim_feedforward,
                transformer_norm=transformer_norm,
                pos_encoding=pos_encoding,
                n_transformer_layers=n_transformer_layers,
                learning_rate=learning_rate,
                optimizer=optimizer,
            ),
            save_code=True,
            save_dir="logs",
            log_model=False,
        )

        self.default_trainer_kwargs = dict(
            max_epochs=1000,
            sync_batchnorm=True,
        )

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
        mask: bool = False,
        masking_len: int = 3,
        masking_ratio: float = 0.15,
        wandb_log: bool = True,
        wandb_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        # update model parameters
        self.model.mask = mask
        self.model.masking_len = masking_len
        self.model.masking_ratio = masking_ratio

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

    def get_latent(
        self,
    ) -> Union[ad.AnnData, dict]:
        loader = self.data.test_dataloader()
        adata = self.data.test
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
                padding_mask = None
                if "padding_mask" in batch:
                    padding_mask = batch["padding_mask"]

                features.append(
                    self.model.forward_features(x, c=c).cpu().detach().numpy()
                )

                pred.append(
                    np.tile(
                        self.model.forward_pred(x, c=c, padding_mask=padding_mask)
                        .cpu()
                        .detach()
                        .numpy(),
                        (self.seq_len, 1),
                    )
                )
                classes.append(np.tile(batch["target"].numpy(), self.seq_len))
                ids.append(batch["ids"].numpy())
        self.model.train()

        features = np.concatenate(features, axis=0)
        n_features = features.shape[-1]
        features = features.reshape(-1, n_features)
        pred = np.concatenate(pred, axis=0)
        classes = np.concatenate(classes, axis=0).flatten()
        ids = np.concatenate(ids, axis=0).flatten()

        padding_mask = ids != -1
        ids, features, pred, classes = (
            ids[padding_mask],
            features[padding_mask, :],
            pred[padding_mask, :],
            classes[padding_mask],
        )

        if adata is not None:
            adata = adata[ids, :].copy()
            adata.obsm["X_autoagg"] = features
            adata.obs["prediction"] = np.argmax(pred, axis=1)
            for cl in range(pred.shape[1]):
                adata.obs[f"class_prob_{cl}"] = pred[:, cl]
            return adata

        else:
            return dict(ids=ids, classes=classes, pred=pred, features=features)
