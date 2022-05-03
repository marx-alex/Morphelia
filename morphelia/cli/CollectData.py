import argparse
import os
from pathlib import Path
import logging

import anndata as ad

logger = logging.getLogger("ExpToAD")
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def run(
    inp,
    out,
    fname=None,
    pattern="*.h5ad",
):
    """
    Merge all AnnData files.

    """
    files = Path(inp).rglob(pattern)

    logger.info(f"Read AnnData objects at {inp} with pattern {pattern}.")
    plates = [ad.read_h5ad(f) for f in files]

    if fname is not None:
        new_name = f"{fname}.h5ad"
    else:
        new_name = "adata.h5ad"

    # concatenate
    if len(plates) > 1:
        logger.info(f"Merge {len(plates)} plates.")
        plates = plates[0].concatenate(*plates[1:])
        logger.info(f"Write file as {new_name}")
        plates.write(Path(os.path.join(out, new_name)))
    elif len(plates) == 1:
        logger.warning(f"Only one plate found: {plates[0]}.\n" "Exit without merging.")
    else:
        logger.warning("No plates found. Exit without merging.")


def main(args=None):
    """Implements the commandline tool to merge a collection of .h5ad files."""
    # initiate the arguments parser
    parser = argparse.ArgumentParser(description="Merge a collection of AnnData files.")

    parser.add_argument(
        "-i",
        "--inp",
        type=str,
        help="Input directory to annotated data.",
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
        "-p",
        "--pattern",
        type=str,
        default="*.h5ad",
        help="Filename pattern for .h5ad files.",
    )

    # parser
    args = parser.parse_args(args)

    # run
    run(inp=args.inp, out=args.out, fname=args.fname, pattern=args.pattern)
