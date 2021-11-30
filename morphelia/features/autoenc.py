import torch
from torch import nn, optim
import logging
from tqdm import tqdm
import numpy as np
from morphelia.tools._utils import _choose_representation

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Autoencoder(nn.Module):
    """Simple autoencoder with two encoding and two decoding layers.

    Args:
    in_shape (int): input shape
    enc_shape (int): desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(Autoencoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(in_shape, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, enc_shape),
        )

        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, in_shape)
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


def train(model, error, optimizer, n_epochs, x, verbose=False):
    model.train()

    iter_obj = tqdm(range(1, n_epochs + 1), desc="Training autoencoder...")
    for epoch in iter_obj:
        optimizer.zero_grad()
        output = model(x)
        loss = error(output, x)
        loss.backward()
        optimizer.step()

        if verbose:
            if epoch % int(0.1 * n_epochs) == 0:
                logging.info(f'Epoch: {epoch}, Loss: {loss.item():.4g}')


def autoencode(adata,
               use_rep=None,
               n_pcs=50,
               n_epochs=5000,
               enc_dims=2,
               verbose=False):
    """Dimensionality reduction with an autoencoder.

    Args:
        adata
        use_rep
        n_pcs
        n_epochs
        enc_dims
        verbose (bool)

    Returns:
        adata
    """
    # get representation of data
    if use_rep is None:
        use_rep = 'X'
    X = _choose_representation(adata,
                               rep=use_rep,
                               n_pcs=n_pcs)
    in_shape = X.shape[-1]

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.from_numpy(X).double().to(device)

    encoder = Autoencoder(in_shape=in_shape, enc_shape=enc_dims).double().to(device)

    error = nn.MSELoss()

    optimizer = optim.Adam(encoder.parameters())
    train(encoder, error, optimizer, n_epochs, X, verbose)

    with torch.no_grad():
        encoded = encoder.encode(X)
        decoded = encoder.decode(encoded)
        mse = error(decoded, X).item()
        enc = encoded.cpu().detach().numpy()
        dec = decoded.cpu().detach().numpy()

    if verbose:
        logging.info(f"Root mean squared error: {np.sqrt(mse):.3f}")

    adata.obsm['X_enc'] = enc
    adata.uns['enc'] = {'mse': mse}

    return adata
