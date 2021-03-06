import argparse
import logging
import os
from pathlib import Path
import anndata as ad
from morphelia.preprocessing import pseudostitch

logger = logging.getLogger("Pseudostitch")
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def run(inp, out, fname=None, tile_grid=(5, 5), img_overlap=0.2):
    """Pseudostitch."""
    if not os.path.exists(out):
        os.makedirs(out)

    # load data
    adata = ad.read_h5ad(inp)

    # feature agglomeration
    logger.info(f"Stitching object at location {inp}")
    adata = pseudostitch(
        adata, tile_grid=tile_grid, img_overlap=img_overlap, verbose=True
    )

    # write file
    if fname is not None:
        new_name = f"{fname}.h5ad"
    else:
        fname = Path(inp).stem
        new_name = f"{fname}_stitched.h5ad"
    logger.info(f"Write file as {new_name}")
    adata.write(os.path.join(out, new_name))


def main(args=None):
    """Implements the commandline tool to perform a pseudostitch on a CellProfiler output.
    This can be useful if images have been acquired with overlap and doubled cells in overlapping regions
    should be removed."""
    # initiate the arguments parser
    parser = argparse.ArgumentParser(description="Pseudostitching.")

    parser.add_argument("-i", "--inp", type=str, help="Input AnnData object.")
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="./",
        help="Where to store the stitched AnnData object.",
    )
    parser.add_argument(
        "--fname",
        type=str,
        default=None,
        help="File name of new file.",
    )
    parser.add_argument(
        "--tile_grid",
        nargs="+",
        default=[5, 5],
        help="Grid dimensions of fields in a single well.",
    )
    parser.add_argument(
        "--img_overlap", type=float, default=0.2, help="Overlap of tiles."
    )

    # parse
    args = parser.parse_args(args)

    # run
    run(
        inp=args.inp,
        out=args.out,
        fname=args.fname,
        tile_grid=args.tile_grid,
        img_overlap=args.img_overlap,
    )
