from typing import Optional, Union

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
import numpy as np
import anndata as ad

from ..modules import VAEEncoder, TransformerEncoder, MMDDecoder
from . import cVAET_us, cVAET_s
from ..base import BaseModel

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class cVAET(BaseModel):
    """Morphelia class for the Conditional-VAE plus Transformer model."""

    def __init__(
        self,
        data: pl.LightningDataModule,
        layer_dims: list = [256, 64],
        latent_dim: int = 10,
        nhead: int = 1,
        dim_feedforward: Optional[int] = None,
        transformer_dropout: float = 0.1,
        pos_dropout: float = 0.1,
        vae_dropout: float = None,
        transformer_norm: str = "batch",
        pos_encoding: Optional[str] = "learnable",
        n_transformer_layers: int = 1,
        vae_batch_norm: bool = False,
        vae_layer_norm: bool = True,
    ):
        self.n_conditions = data.n_conditions
        self.in_features = data.n_features
        self.seq_len = data.seq_len
        self.n_classes = data.n_classes
        self.class_labels = data.class_labels
        self.batch_size = data.batch_size
        self.data = data

        self.layer_dims = layer_dims
        self.latent_dim = latent_dim
        encoder_layer_dims = self.layer_dims.copy()
        encoder_layer_dims.insert(0, self.in_features)
        decoder_layer_dims = encoder_layer_dims.copy()
        decoder_layer_dims.reverse()

        self.encoder = VAEEncoder(
            layer_dims=encoder_layer_dims,
            latent_dim=self.latent_dim,
            n_conditions=self.n_conditions,
            sequential=True,
            batch_norm=vae_batch_norm,
            layer_norm=vae_layer_norm,
            dropout=vae_dropout,
        )

        self.decoder = MMDDecoder(
            layer_dims=decoder_layer_dims,
            latent_dim=self.latent_dim,
            n_conditions=self.n_conditions,
            sequential=True,
            batch_norm=vae_batch_norm,
            layer_norm=vae_layer_norm,
            dropout=vae_dropout,
        )

        self.transformer = TransformerEncoder(
            input_dim=self.latent_dim,
            seq_len=self.seq_len,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            pos_dropout=pos_dropout,
            norm=transformer_norm,
            pos_encoding=pos_encoding,
            num_layers=n_transformer_layers,
        )

        self.default_wandb_kwargs = dict(
            project="cvaet",
            config=dict(
                n_features=self.in_features,
                batch_size=self.batch_size,
                n_conditions=self.n_conditions,
                seq_len=self.seq_len,
                n_classes=self.n_classes,
            ),
            save_code=True,
            save_dir="logs",
            log_model=False,
        )

        self.default_trainer_kwargs = dict(
            max_epochs=1000,
            accumulate_grad_batches=4,
            sync_batchnorm=True,
            log_every_n_steps=1,
            flush_logs_every_n_steps=1,
        )

        self.pretrained_model = None
        self.model = None

    def pretrain(
        self,
        kld_alpha: float = 1.0,
        mmd_alpha: float = 1.0,
        mse_alpha: float = 1.0,
        mask: bool = True,
        masking_len=3,
        masking_ratio=0.15,
        learning_rate: float = 1e-4,
        optimizer: str = "Adam",
        wandb_log: bool = True,
        wandb_kwargs: Optional[dict] = None,
        **kwargs
    ):
        if wandb_log:
            default_wandb_kwargs = self.default_wandb_kwargs.copy()
            default_wandb_kwargs.update(dict(group="unsupervised pretraining"))
            if wandb_kwargs is not None:
                default_wandb_kwargs.update(wandb_kwargs)

            logger = WandbLogger(**default_wandb_kwargs)
        else:
            logger = False

        model = cVAET_us(
            encoder=self.encoder,
            transformer=self.transformer,
            decoder=self.decoder,
            kld_beta=kld_alpha,
            mmd_beta=mmd_alpha,
            mse_beta=mse_alpha,
            mask=mask,
            masking_len=masking_len,
            masking_ratio=masking_ratio,
            learning_rate=learning_rate,
            optimizer=optimizer,
        )

        callbacks = [
            EarlyStopping(
                monitor="valid/loss", min_delta=0.005, patience=30, mode="max"
            ),
            ModelCheckpoint(
                filename="{epoch}_best_loss",
                monitor="valid/loss",
                save_top_k=3,
                mode="min",
            ),
        ]

        default_trainer_kwargs = self.default_trainer_kwargs.copy()
        default_trainer_kwargs.update(dict(logger=logger, callbacks=callbacks))
        default_trainer_kwargs.update(kwargs)

        trainer = pl.Trainer(**default_trainer_kwargs)
        trainer.fit(model, self.data)
        self.pretrained_model = model

        if wandb_log:
            logger.experiment.finish()

    def train(
        self,
        learning_rate: float = 1e-4,
        optimizer: str = "Adam",
        wandb_log: bool = True,
        wandb_kwargs: Optional[dict] = None,
        **kwargs
    ):
        assert self.pretrained_model is not None, "No pretrained model."

        if wandb_log:
            default_wandb_kwargs = self.default_wandb_kwargs.copy()
            default_wandb_kwargs.update(dict(group="supervised training"))
            if wandb_kwargs is not None:
                default_wandb_kwargs.update(wandb_kwargs)

            logger = WandbLogger(**default_wandb_kwargs)
        else:
            logger = False

        model = cVAET_s(
            self.pretrained_model,
            n_classes=self.n_classes,
            learning_rate=learning_rate,
            optimizer=optimizer,
        )

        callbacks = [
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

        default_trainer_kwargs = self.default_trainer_kwargs.copy()
        default_trainer_kwargs.update(dict(logger=logger, callbacks=callbacks))
        default_trainer_kwargs.update(kwargs)

        trainer = pl.Trainer(**default_trainer_kwargs)
        trainer.fit(model, self.data)
        self.model = model

        if wandb_log:
            logger.experiment.finish()

    def load_from_checkpoints(
        self, ckpt: Optional[str] = None, pretrained_ckpt: Optional[str] = None
    ) -> None:
        if ckpt is not None:
            self.model = cVAET_s.load_from_checkpoint(ckpt)
        if pretrained_ckpt is not None:
            self.pretrained_model = cVAET_us.load_from_checkpoint(pretrained_ckpt)

    def get_latent(
        self,
    ) -> Union[ad.AnnData, dict]:
        loader = self.data.test_dataloader()
        adata = self.data.test
        features = []
        logits = []
        classes = []
        ids = []

        self.model.eval()
        with torch.no_grad():
            for sample in tqdm(loader, desc="Load samples from test set."):
                x = sample["x"]
                c = None
                if "c" in sample:
                    c = sample["c"]
                padding_mask = None
                if "padding_mask" in sample:
                    padding_mask = sample["padding_mask"]

                features.append(
                    self.model.forward_features(x, c=c, key_padding_mask=padding_mask)
                    .cpu()
                    .detach()
                    .numpy()
                )
                logits.append(
                    np.tile(
                        self.model(x, c=c, padding_mask=padding_mask)
                        .cpu()
                        .detach()
                        .numpy(),
                        (self.seq_len, 1),
                    )
                )
                classes.append(np.tile(sample["target"].numpy(), self.seq_len))
                ids.append(sample["ids"].numpy())
        self.model.train()

        features = np.concatenate(features, axis=0)
        n_features = features.shape[-1]
        features = features.reshape(-1, n_features)
        logits = np.concatenate(logits, axis=0)
        classes = np.concatenate(classes, axis=0).flatten()
        ids = np.concatenate(ids, axis=0).flatten()

        padding_mask = ids != -1
        ids, features, logits, classes = (
            ids[padding_mask],
            features[padding_mask, :],
            logits[padding_mask, :],
            classes[padding_mask],
        )

        if adata is not None:
            adata = adata[ids, :].copy()
            adata.obsm["X_cvaet"] = features
            adata.obsm["X_logits"] = logits
            return adata

        else:
            return dict(ids=ids, classes=classes, logits=logits, features=features)
