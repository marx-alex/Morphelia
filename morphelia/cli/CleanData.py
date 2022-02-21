import argparse
import os
from pathlib import Path
import logging
import sys

import anndata as ad

import morphelia

logger = logging.getLogger("CleanData")
logging.basicConfig()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def run(inp, out, nan_frac=0.5):
    """Basic cleaning routine for morphological data."""
    if not os.path.exists(out):
        os.makedirs(out)

    # load data
    logger.info(f"Read AnnData object: {inp}")
    adata = ad.read_h5ad(inp)

    # nan values
    logger.info(f"Drop features with more than {nan_frac * 100}% Nan values")
    adata = morphelia.pp.drop_nan(adata, axis=0, min_nan_frac=nan_frac, verbose=True)

    logger.info("Drop cells with Nan values")
    adata = morphelia.pp.drop_nan(adata, axis=1, verbose=True)

    # duplicated features / cells
    logger.info("Drop duplicated features")
    adata = morphelia.pp.drop_duplicates(adata, axis=1, verbose=True)

    logger.info("Drop duplicated cells")
    adata = morphelia.pp.drop_duplicates(adata, axis=0, verbose=True)

    # invariant features / cells
    logger.info("Drop invariant features")
    adata = morphelia.pp.drop_invariant(adata, axis=0, verbose=True)

    logger.info("Drop invariant cells")
    adata = morphelia.pp.drop_invariant(adata, axis=1, verbose=True)

    # write file
    fname = Path(inp).stem
    new_name = f"{fname}_clean.h5ad"
    logger.info(f"Write file as {new_name}")
    adata.write(os.path.join(out, new_name))


def main(args=None):
    """Implements the commandline tool to clean morphological data."""
    # initiate the arguments parser
    parser = argparse.ArgumentParser(description="Clean data.")

    parser.add_argument(
        "-i",
        "--inp",
        type=str,
        help="Input directory to Cellprofiler output for a whole experiment.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="./",
        help="Where to store the stitched AnnData object.",
    )
    parser.add_argument(
        "--nan_frac",
        type=float,
        default=0.5,
        help="Minimum fraction of nan values in features before being dropped.",
    )

    # parser
    args = parser.parse_args(args)

    # run
    run(inp=args.inp, out=args.out, nan_frac=args.nan_frac)
