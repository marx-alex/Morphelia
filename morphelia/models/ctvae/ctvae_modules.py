from typing import Optional, Tuple

import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics

from morphelia.models.utils import geometric_noise


class CTVAE(pl.LightningModule):
    """Implementation of the Autoaggregation model as
    a PyTorch Lighnting module.

    Parameters
    ----------
    encoder : torch.nn.Module
        Encoder module for dimensionality reduction
    transformer : torch.nn.Module
        Transformer module
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
        transformer: nn.Module,
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

        self.encoder = encoder
        self.transformer = transformer
        self.classifier = nn.Linear(
            transformer.seq_len * encoder.latent_dim, self.n_classes
        )

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
        module and returns embedding with attention weights.

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
            Embedding and attention weights
        """
        z = self.encoder(x, c=c)
        z, w = self.transformer(z, key_padding_mask=padding_mask, return_attention=True)

        return z, w

    def forward_features(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Passes tensor through encoder module.

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
            Latent representation
        """
        z = self.encoder(x, c=c)
        z = self.transformer(z, key_padding_mask=padding_mask)

        return z

    def classify(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Classification by a single classification layer.

        Parameters
        ----------
        x : torch.Tensor
        padding_mask : torch.Tensor, optional
            Padding mask

        Returns
        -------
        torch.Tensor
            Logits
        """
        if padding_mask is not None:
            y = x * ~padding_mask.unsqueeze(-1)
        else:
            y = x

        y = x.reshape(y.shape[0], -1)  # [N, S x F]
        return self.classifier(y)  # [N, n_classes]

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
        x = self(x, c=c, padding_mask=padding_mask)
        return self.softmax(x)

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
        z = self.forward_features(x, c=c, padding_mask=padding_mask)
        logits = self.classify(z, padding_mask=padding_mask)  # [N, n_classes]

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
