import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl

from typing import Optional, Tuple
import logging

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


class ClusterFate(nn.Module):
    def __init__(
            self,
            n_classes: int,
            enc_shape: int,
            term_states: Optional[torch.Tensor] = None,
            start_state: Optional[torch.Tensor] = None
    ) -> None:
        """

        :param n_classes:
        :param enc_shape:
        :param term_states:
        :param start_state:
        """
        super().__init__()
        self.enc_shape = enc_shape
        self.n_classes = n_classes

        if term_states is None:
            initial_term_states = torch.zeros(
                self.n_classes, self.enc_shape, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_term_states)
        else:
            initial_term_states = term_states
        self.term_states = nn.Parameter(initial_term_states)

        if start_state is None:
            initial_start_state = torch.mean(self.cluster_centers, dim=0)
        else:
            initial_start_state = start_state
        self.start_state = nn.Parameter(initial_start_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """


class Encoder(nn.Module):
    def __init__(self,
                 in_shape,
                 out_shape):
        super(Encoder, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

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
            nn.Linear(16, self.out_shape),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): B x F
        """
        return self.encode(x)  # (B,F) -> (B,enc_shape)


class Decoder(nn.Module):
    def __init__(self,
                 in_shape,
                 out_shape):
        super(Decoder, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

        self.decode = nn.Sequential(
            nn.Linear(self.out_shape, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, self.in_shape)
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): B x F
        """
        return self.decode(x)  # (B,enc_shape) -> (B,F)


class DeepMap(pl.LightningModule):
    """
    DeepMap as described by Ren et al., 2021, bioRxiv

    Args:
        in_shape (int): input shape
        enc_shape (int): desired encoded shape
        n_classes (int): number of classes
        lr (float): learning rate
        betas (list): weights for Cross Entropy, MSE and Cluster Center Loss
    """

    def __init__(self, in_shape, enc_shape, n_classes, lr=1e-3, betas=None):
        super().__init__()
        self.lr = lr
        self.in_shape = in_shape
        self.enc_shape = enc_shape
        self.n_classes = n_classes

        if betas is not None:
            assert len(betas) == 3, f'betas should have lenght 3, instead got lenght {len(betas)}'
        else:
            betas = [1, 1, 1]
        self.betas = betas

        self.encoder = Encoder(self.in_shape,
                               self.enc_shape)

        self.decoder = Decoder(self.in_shape,
                               self.enc_shape)

        self.clustering = ClusterDistance(self.n_classes, self.enc_shape)
        self.tanhshrink = nn.Tanhshrink()
        self.softmax = nn.Softmax(dim=1)

        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

        metrics = torchmetrics.MetricCollection(
            dict(
                ACC=torchmetrics.Accuracy(num_classes=self.n_classes),
                F1=torchmetrics.F1Score(num_classes=self.n_classes),
            )
        )

        self.train_metric = metrics.clone("train/")
        self.valid_metric = metrics.clone("valid/")
        self.test_metric = metrics.clone("test/")

    def cluster(self, x, return_logits=False):
        z = self.encoder(x)
        logits, cluster_loss = self.clustering(z)
        logits = self.tanhshrink(logits)
        pred = self.softmax(logits)
        if return_logits:
            return pred, logits
        return pred

    def forward(self, x):
        z = self.encoder(x)
        logits, cluster_loss = self.clustering(z)
        logits = self.tanhshrink(logits)
        pred = self.softmax(logits)
        x_hat = self.decoder(z)
        return logits, pred, x_hat, cluster_loss

    def training_step(self, batch, batch_idx):
        losses, pred, target = self._common_step(batch)
        loss = sum([w * l for w, l in zip(self.betas, losses)])
        return dict(loss=loss, pred=pred, target=target, losses=losses)

    def training_step_end(self, outs):
        loss = outs['loss']
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        losses = {'CrossEntropy': outs['losses'][0], 'MSE': outs['losses'][1], 'ClusterDistance': outs['losses'][2]}
        self.log('train/losses', losses, on_step=False, on_epoch=True)
        self.train_metric(outs["pred"], outs["target"])
        return dict(
            loss=loss,
            pred=outs["pred"].detach(),
            target=outs["target"].detach(),
        )

    def training_epoch_end(self, outs) -> None:
        self.log_dict(
            self.train_metric.compute(), on_step=False, on_epoch=True
        )
        self.train_metric.reset()

    def validation_step(self, batch, batch_idx):
        losses, pred, target = self._common_step(batch)
        loss = sum([w * l for w, l in zip(self.betas, losses)])
        self.log("valid/loss", loss)
        return dict(loss=loss, pred=pred, target=target, losses=losses)

    def validation_step_end(self, outs):
        loss = outs['loss']
        self.log("valid/loss", loss, on_step=False, on_epoch=True)
        losses = {'CrossEntropy': outs['losses'][0], 'MSE': outs['losses'][1], 'ClusterDistance': outs['losses'][2]}
        self.log('valid/losses', losses, on_step=False, on_epoch=True)
        self.valid_metric(outs["pred"], outs["target"])
        return dict(
            loss=loss,
            pred=outs["pred"],
            target=outs["target"]
        )

    def validation_epoch_end(self, outs) -> None:
        self.log_dict(
            self.valid_metric.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.valid_metric.reset()

    def test_step(self, batch, batch_idx):
        losses, pred, target = self._common_step(batch)
        loss = sum([w * l for w, l in zip(self.betas, losses)])
        self.log("test/loss", loss)
        return dict(loss=loss, pred=pred, target=target)

    def test_step_end(self, outs):
        loss = outs['loss']
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.test_metric(outs["pred"], outs["target"])
        return dict(
            loss=loss,
            pred=outs["pred"],
            target=outs["target"]
        )

    def test_epoch_end(self, outs) -> None:
        self.log_dict(self.test_metric.compute(), on_step=False, on_epoch=True)
        self.test_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def _prepare_batch(batch):
        x, y, _ = batch
        return x, y

    def _common_step(self, batch):
        x, y = self._prepare_batch(batch)
        logits, pred, x_hat, cluster_loss = self(x)
        loss1 = self.cross_entropy(logits, y)
        loss2 = self.mse(x_hat, x)
        return (loss1, loss2, cluster_loss), pred, y