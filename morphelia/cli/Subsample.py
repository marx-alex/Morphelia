import argparse
import os
from pathlib import Path
import logging
import sys

import anndata as ad

import morphelia

logger = logging.getLogger('Subsample')
logging.basicConfig()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def run(inp, out,
        by=("BatchNumber", "PlateNumber", "Metadata_Well"),
        frac=0.1):
    """Subsample data.

    """
    if not os.path.exists(out):
        os.makedirs(out)

    # load data
    logger.info(f"Read AnnData object: {inp}")
    adata = ad.read_h5ad(inp)

    logger.info("Subsample data...")
    adata = morphelia.pp.subsample(adata, perc=frac, by=by)

    # write file
    fname = Path(inp).stem
    new_name = f"{fname}_subsample.h5ad"
    logger.info(f"Write file as {new_name}")
    adata.write(os.path.join(out, new_name))


def main(args=None):
    """Implements the commandline tool to subsample morphological data."""
    # initiate the arguments parser
    parser = argparse.ArgumentParser(description=f'Subsample data.')

    parser.add_argument('-i', '--inp', type=str,
                        help='Input directory to Cellprofiler output for a whole experiment.')
    parser.add_argument('-o', '--out', type=str, default="./",
                        help="Where to store the stitched AnnData object.")
    parser.add_argument('-f', '--frac', type=float, default=0.1,
                        help="Fraction of all cells to subsample.")
    parser.add_argument('-b', '--by', nargs='+', default=["BatchNumber", "PlateNumber", "Metadata_Well"],
                        help='Group data by those values and subsample every group.')

    # parser
    args = parser.parse_args(args)

    # run
    run(inp=args.inp,
        out=args.out,
        frac=args.frac,
        by=args.by)
