import argparse
import os
from pathlib import Path
import logging
import sys

import anndata as ad

import morphelia

logger = logging.getLogger("Split")
logging.basicConfig()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def run(
    inp,
    out,
    train_sfx="adata_train",
    test_sfx="adata_test",
    stratify=None,
    group=None,
    test_size=0.2,
):
    """Split data."""
    if not os.path.exists(out):
        os.makedirs(out)

    # load data
    logger.info(f"Read AnnData object: {inp}")
    adata = ad.read_h5ad(inp)

    logger.info("Split data...")
    if group is not None:
        adata_train, adata_test = morphelia.tl.group_shuffle_split(
            adata, group="Metadata_Trace_Tree", stratify=stratify, test_size=test_size
        )

    else:
        adata_train, adata_test = morphelia.tl.train_test_split(
            adata, stratify=stratify, test_size=test_size
        )

    # write file
    fname = Path(inp).stem
    logger.info(f"Write files as {fname}_{train_sfx}.h5ad and {fname}_{test_sfx}.h5ad")
    adata_train.write(os.path.join(out, f"{fname}_{train_sfx}.h5ad"))
    adata_train.write(os.path.join(out, f"{fname}_{test_sfx}.h5ad"))


def main(args=None):
    """Implements the commandline tool to split morphological data."""
    # initiate the arguments parser
    parser = argparse.ArgumentParser(description="Split data.")

    parser.add_argument(
        "-i",
        "--inp",
        type=str,
        help="Input AnnData file.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="./",
        help="Output directory.",
    )
    parser.add_argument(
        "--train_sfx",
        type=str,
        default="train",
        help="Suffix for the training set.",
    )
    parser.add_argument(
        "--test_sfx",
        type=str,
        default="test",
        help="Suffix for the testing set.",
    )
    parser.add_argument(
        "--stratify",
        type=str,
        default=None,
        help="Variable for stratification",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Variable for grouping",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Size of the test set.",
    )

    # parser
    args = parser.parse_args(args)

    # run
    run(
        inp=args.inp,
        out=args.out,
        train_sfx=args.train_sfx,
        test_sfx=args.test_sfx,
        stratify=args.stratify,
        group=args.group,
        test_size=args.test_size,
    )
