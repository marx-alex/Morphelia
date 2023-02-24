from typing import Optional, Tuple

import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics

from morphelia.models.utils import geometric_noise


class AUTOAGG(pl.LightningModule):
    """Implementation of the Autoaggregation model as
    as PyTorch Lighnting module.

    Parameters
    ----------
    encoder : torch.nn.Module
        Encoder module for dimensionality reduction
    classifier : torch.nn.Module
        Classifier module
    n_classes : int
        Number of classes
    mask : bool
        Geometrical masks to hide parts of the sequential data to the model
    masking_len : int
        Mean length of masks
    masking_ratio : float
        Absolute fraction of masked values
    learning_rate : float
        Learning rate during training
    optimizer : str
        Optimizer for the training process.
        Can be `Adam` or `AdamW`.
    """

    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        n_classes: int,
        mask: bool = False,
        masking_len: int = 3,
        masking_ratio: float = 0.15,
        learning_rate: float = 1e-4,
        optimizer: str = "Adam",
    ):
        super().__init__()

        self.save_hyperparameters()
        self.mask = mask
        self.masking_len = masking_len
        self.masking_ratio = masking_ratio
        self.n_classes = n_classes
        self.latent_dim = encoder.latent_dim
        self.seq_len = classifier.seq_len

        self.encoder = encoder
        self.classifier = classifier

        self.cross_entropy = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(-1)

        metrics = torchmetrics.MetricCollection(
            dict(
                ACC=torchmetrics.Accuracy(num_classes=n_classes),
                F1=torchmetrics.F1Score(num_classes=n_classes),
            )
        )

        self.train_metric = metrics.clone(prefix="train/")
        self.valid_metric = metrics.clone(prefix="valid/")
        self.test_metric = metrics.clone(prefix="test/")

    def forward_with_attention(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Passes tensor through encoder and classification
        module and returns predictions with attention weights.

        Parameters
        ----------
        x : torch.Tensor
        c : torch.Tensor, optional
            Conditions
        padding_mask : torch.Tensor, optional
            Padding mask

        Returns
        -------
        torch.Tensor, torch.Tensor
            Predictions and attention weights
        """
        z = self.encoder(x, c=c)
        logits, w = self.classifier(
            z, key_padding_mask=padding_mask, return_attention=True
        )
        pred = self.softmax(logits)

        return pred, w

    def forward_features(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Passes tensor through encoder module.

        Parameters
        ----------
        x : torch.Tensor
        c : torch.Tensor, optional
            Conditions

        Returns
        -------
        torch.Tensor
            Latent representation
        """
        z = self.encoder(x, c=c)
        return z

    def forward_pred(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forwards tensor through model and transforms
        logits with the SoftMax function.

        Parameters
        ----------
        x : torch.Tensor
        c : torch.Tensor, optional
            Conditions
        padding_mask : torch.Tensor, optional
            Padding mask

        Returns
        -------
        torch.Tensor
            Predictions
        """
        logits = self(x, c=c, padding_mask=padding_mask)
        return self.softmax(logits)

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forwards tensor through model.

        Parameters
        ----------
        x : torch.Tensor
        c : torch.Tensor, optional
            Conditions
        padding_mask : torch.Tensor, optional
            Tensor of shape `[batch size, sequence length]`.
            0s mean keep vector at this position, 1s means padding.

        Returns
        -------
        torch.Tensor
            Logits
        """
        z = self.forward_features(x, c=c)
        logits = self.classifier(z, key_padding_mask=padding_mask)  # [N, n_classes]
        return logits

    def training_step(self, batch: dict, batch_idx: torch.Tensor) -> torch.Tensor:
        """Training step.

        Parameters
        ----------
        batch : dict
            Dictionary with `x` and `target` as required keys.
            `c` and `padding_mask` are optional keys.
        batch_idx : torch.Tensor
            Batch index

        Returns
        -------
        torch.Tensor
            Loss
        """
        loss, pred, target = self._common_step(batch, prefix="train")
        self.train_metric.update(pred, target)
        self.log_dict(self.train_metric, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: torch.Tensor) -> torch.Tensor:
        """Validation step.

        Parameters
        ----------
        batch : dict
            Dictionary with `x` and `target` as required keys.
            `c` and `padding_mask` are optional keys.
        batch_idx : torch.Tensor
            Batch index

        Returns
        -------
        torch.Tensor
            Loss
        """
        loss, pred, target = self._common_step(batch, prefix="valid")
        self.valid_metric.update(pred, target)
        self.log_dict(self.valid_metric, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: dict, batch_idx: torch.Tensor) -> torch.Tensor:
        """Test step.

        Parameters
        ----------
        batch : dict
            Dictionary with `x` and `target` as required keys.
            `c` and `padding_mask` are optional keys.
        batch_idx : torch.Tensor
            Batch index

        Returns
        -------
        torch.Tensor
            Loss
        """
        loss, pred, target = self._common_step(batch, prefix="test")
        self.test_metric.update(pred, target)
        self.log_dict(self.test_metric, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer.

        Returns
        -------
        torch.optim.Adam or torch.optim.AdamW
        """
        if self.hparams.optimizer == "Adam":
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "AdamW":
            opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        else:
            assert False, f"{self.hparams.optimizer=} is not Adam or AdamW"

        return opt

    def _masking(self, x: torch.Tensor) -> torch.Tensor:
        x_mask = geometric_noise(x, self.masking_len, self.masking_ratio)
        x = x * x_mask
        return x

    @staticmethod
    def _prepare_batch(batch):
        x = batch["x"]
        c = None
        if "c" in batch:
            c = batch["c"]
        padding_mask = None
        if "padding_mask" in batch:
            padding_mask = batch["padding_mask"]
        target = batch["target"]
        return x, target, c, padding_mask

    def _common_step(self, batch: dict, prefix: str = "train"):
        x, target, c, padding_mask = self._prepare_batch(batch)

        if self.mask:
            x = self._masking(x)

        logits = self(x, c=c, padding_mask=padding_mask)
        pred = self.softmax(logits)

        loss = self.cross_entropy(logits, target)

        self.log(
            f"{prefix}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss, pred, target
