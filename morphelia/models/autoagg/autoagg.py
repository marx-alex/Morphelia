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

from typing import Optional, Union, Sequence
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class Autoagg(BaseModel):
    """Morphelia class for Autoaggregation.

    This is a convenience class for model initialization,
    data loading, training and predicting morphological data.
    The Autoaggregation model models image-based profiles from
    single cells as sequences. A sequence contains samples
    from same conditions (i.g. treatment groups).
    The model contains two main parts: An encoder for
    dimensionality reduction and a classification part.
    Classification can be done by a Transformer classifier,
    a mean oder median classifier of by majority vote.

    Parameters
    ----------
    data : pytorch_lightning.LightningDataModule
        This data module should have a train, validation and
        test loader that yield batches of sequential data
    encoder_layer_dims : sequence of int, optional
        Dimensions of hidden layers. The length of
        the sequences it equal to the number of hidden layers.
    latent_dim : int
        Dimensions of the latent space
    encoder_dropout : float
        Dropout rate for the encoder part
    classification_method : str
        Choose one of the available classifiers: `transformer`,
        `transformer_token`, `mean`, `median`, `majority`
    nhead : int
        Number of heads for the Transformer classifier
    dim_feedforward : int, optional
        Number of dimensions for the feed-forward layer of the
        transformer classifier. The default dimensions is
        2 * input dimensions.
    transformer_dropout : float
        Dropout rate for the Transformer
    pos_dropout : float
        Dropout rate for the positional encoding
    transformer_norm : str
        Normalization method for the Transformer classifier.
        Either batch normalization (`batch`) or layer
        normalization (`layer`).
    pos_encoding : str
        Kind of the positional encoding for the Transformer classifier.
        Either `learnable` of `fixed`.
    n_transformer_layers : int
        Number of layers for the Transformer classifier
    learning_rate : float
        Learning rate during training
    optimizer : str
        Optimizer for the training process.
        Can be `Adam` or `AdamW`.

    Raises
    ------
    AssertionError
        If `classification_method` is unknown
    """

    def __init__(
        self,
        data: pl.LightningDataModule,
        encoder_layer_dims: Optional[Sequence[int]] = None,
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
    ) -> None:
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
    ) -> None:
        """Training method.

        Parameters
        ----------
        mask : bool
            Geometrical masks to hide parts of the sequential data to the model
        masking_len : int
            Mean length of masks
        masking_ratio : float
            Absolute fraction of masked values
        wandb_log : bool
            Log training to Weights and Biases
        wandb_kwargs : dict, optional
            Keyword arguments for the logger
        **kwargs
            Keyword arguments passed to pytorch_lightning.Trainer
        """
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
                    np.repeat(
                        self.model.forward_pred(x, c=c, padding_mask=padding_mask)
                        .cpu()
                        .detach()
                        .numpy(),
                        repeats=self.seq_len,
                        axis=0,
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
