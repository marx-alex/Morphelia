import argparse
import os
from pathlib import Path
import logging

import anndata as ad

import morphelia

logger = logging.getLogger("Normalize")
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def run(
    inp,
    out,
    fname=None,
    by=("BatchNumber", "PlateNumber"),
    method="standard",
    pop_var="Metadata_Treatment",
    norm_pop=None,
):
    """Normalize data."""
    if not os.path.exists(out):
        os.makedirs(out)

    # load data
    logger.info(f"Read AnnData object: {inp}")
    adata = ad.read_h5ad(inp)

    logger.info("Nomalize data...")
    adata = morphelia.pp.normalize(
        adata,
        by=by,
        method=method,
        pop_var=pop_var,
        norm_pop=norm_pop,
        verbose=True,
    )

    # write file
    if fname is not None:
        new_name = f"{fname}.h5ad"
    else:
        fname = Path(inp).stem
        new_name = f"{fname}_norm.h5ad"
    logger.info(f"Write file as {new_name}")
    adata.write(os.path.join(out, new_name))


def main(args=None):
    """Implements the commandline tool to normalize morphological data."""
    # initiate the arguments parser
    parser = argparse.ArgumentParser(description="Subsample data.")

    parser.add_argument(
        "-i",
        "--inp",
        type=str,
        help="AnnData object.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="./",
        help="Output directory.",
    )
    parser.add_argument(
        "--fname",
        type=str,
        default=None,
        help="File name of new file.",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="standard",
        help="Nomalization method. One of: standard, robust, mad_robust, min_max.",
    )
    parser.add_argument(
        "-b",
        "--by",
        nargs="+",
        default=["BatchNumber", "PlateNumber"],
        help="Group data by those values and normalize every group separately.",
    )
    parser.add_argument(
        "--pop_var",
        type=str,
        default="Metadata_Treatment",
        help="Variable that stored control population identifier.",
    )
    parser.add_argument(
        "--norm_pop",
        type=str,
        default=None,
        help="Control population to be found under pop_var",
    )

    # parser
    args = parser.parse_args(args)

    # run
    run(
        inp=args.inp,
        out=args.out,
        fname=args.fname,
        by=args.by,
        method=args.method,
        pop_var=args.pop_var,
        norm_pop=args.norm_pop,
    )
