import argparse
import os
from pathlib import Path

from morphelia.models import cVAET, TSDataModule


parser = argparse.ArgumentParser(description="cVAET")

parser.add_argument("-b", "--batch", default=16, type=int, help="Batch size")
parser.add_argument("-a", "--acc", default=4, type=int, help="Accumulate gradients.")
parser.add_argument("-f", "--file", type=str, help="AnnData file.")
parser.add_argument("-o", "--out", type=str, help="Output directory.")


args = parser.parse_args()

batch_size = args.batch
acc_grad = args.acc
file = args.file
out = args.out

data = TSDataModule(file, condition_key="PlateNumber")

cvaet = cVAET(data)

cvaet.pretrain(wandb_log=True)
cvaet.train(wandb_log=True)
latent = cvaet.get_latent()

latent.write(Path(os.path.join(out, "cvaet_latent.h5ad")))
