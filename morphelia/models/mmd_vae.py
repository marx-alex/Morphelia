import torch
from torch import nn
import pytorch_lightning as pl
from torch.autograd import Variable


def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
    tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)

    return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / dim * 1.0)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)


class Encoder(nn.Module):
    def __init__(self, in_shape, out_shape):
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
            nn.Linear(32, self.out_shape),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): B x F
        """
        return self.encode(x)  # (B,F) -> (B,enc_shape)


class Decoder(nn.Module):
    def __init__(self, in_shape, out_shape):
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
            nn.Linear(128, self.in_shape),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): B x F
        """
        return self.decode(x)  # (B,enc_shape) -> (B,F)


class MMDVAE(pl.LightningModule):
    """
    MMD-Variational Autoencoder.
    Based on https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder/blob/master/mmd_vae.ipynb

    Args:
        in_shape (int): input shape
        enc_shape (int): desired encoded shape
        n_classes (int): number of classes
        lr (float): learning rate
    """

    def __init__(self, in_shape, enc_shape, n_classes, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.in_shape = in_shape
        self.enc_shape = enc_shape
        self.n_classes = n_classes

        self.encoder = Encoder(self.in_shape, self.enc_shape)

        self.decoder = Decoder(self.in_shape, self.enc_shape)

        if torch.cuda.is_available():
            self.true_samples = self.true_samples.cuda()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        return dict(loss=loss)

    def training_step_end(self, outs):
        loss = outs["loss"]
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("valid/loss", loss.item())
        return dict(loss=loss)

    def validation_step_end(self, outs):
        loss = outs["loss"]
        self.log("valid/loss", loss, on_step=False, on_epoch=True)
        return dict(loss=loss)

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("test/loss", loss.item())
        return dict(loss=loss)

    def test_step_end(self, outs):
        loss = outs["loss"]
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return dict(loss=loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def _prepare_batch(batch):
        x, y, _ = batch
        return x, y

    def _common_step(self, batch):
        x, y = self._prepare_batch(batch)
        z, x_hat = self(x)

        true_samples = Variable(
            torch.randn(len(x), self.enc_shape), requires_grad=False
        )

        mmd = compute_mmd(true_samples, z)
        nll = (x_hat - x).pow(2).mean()
        loss = nll + mmd
        return loss
