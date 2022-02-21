import argparse
import os
from pathlib import Path
import logging
import sys

import anndata as ad

import morphelia

logger = logging.getLogger("Aggregate")
logging.basicConfig()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def run(
    inp,
    out,
    by=("BatchNumber", "PlateNumber", "Metadata_Well"),
    method="median",
):
    """Aggregate data."""
    if not os.path.exists(out):
        os.makedirs(out)

    # load data
    logger.info(f"Read AnnData object: {inp}")
    adata = ad.read_h5ad(inp)

    logger.info("Aggregate data...")
    adata = morphelia.pp.aggregate(adata, method=method, by=by)

    # write file
    fname = Path(inp).stem
    new_name = f"{fname}_aggregated.h5ad"
    logger.info(f"Write file as {new_name}")
    adata.write(os.path.join(out, new_name))


def main(args=None):
    """Implements the commandline tool to aggregate morphological data."""
    # initiate the arguments parser
    parser = argparse.ArgumentParser(description="Aggregate data.")

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
        "-m",
        "--method",
        type=str,
        default="median",
        help="Method to use for aggregation: mean, median or modz.",
    )
    parser.add_argument(
        "-b",
        "--by",
        nargs="+",
        default=["BatchNumber", "PlateNumber", "Metadata_Well"],
        help="Group data by those values and aggregate every group.",
    )

    # parser
    args = parser.parse_args(args)

    # run
    run(inp=args.inp, out=args.out, method=args.method, by=args.by)
