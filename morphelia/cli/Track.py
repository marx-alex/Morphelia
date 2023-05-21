import argparse
import os
from pathlib import Path
import logging

import anndata as ad

import morphelia

logger = logging.getLogger("Track")
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def run(
    inp,
    out,
    fname=None,
    time_var="Metadata_Time",
    group_vars=("Metadata_Well", "Metadata_Field"),
    x_loc="Cells_Location_Center_X",
    y_loc="Cells_Location_Center_Y",
    filter_dummies=False,
    filter_delayed_roots=False,
    drop_untracked=True,
    max_search_radius=100,
    field_size=(2048, 2048),
    approx=False,
):
    """Tracking of cells in a time series experiment"""
    if not os.path.exists(out):
        os.makedirs(out)

    # load data
    logger.info(f"Read AnnData object: {inp}")
    adata = ad.read_h5ad(inp)

    # tracking
    adata = morphelia.ts.track(
        adata,
        time_var=time_var,
        group_vars=group_vars,
        x_loc=x_loc,
        y_loc=y_loc,
        filter_dummies=filter_dummies,
        filter_delayed_roots=filter_delayed_roots,
        drop_untracked=drop_untracked,
        max_search_radius=max_search_radius,
        field_size=field_size,
        approx=approx,
        verbose=True,
    )

    # write file
    if fname is not None:
        new_name = f"{fname}.h5ad"
    else:
        fname = Path(inp).stem
        new_name = f"{fname}_tracked.h5ad"
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
        "-t",
        "--time_var",
        type=str,
        default="Metadata_Time",
        help="Variable with information about time.",
    )
    parser.add_argument(
        "-g",
        "--group_vars",
        nargs="+",
        default=["Metadata_Well", "Metadata_Field"],
        help="Group data by these variables to get single fields.",
    )
    parser.add_argument(
        "--x_loc",
        type=str,
        default="Cells_Location_Center_X",
        help="Variable with information about x location.",
    )
    parser.add_argument(
        "--y_loc",
        type=str,
        default="Cells_Location_Center_Y",
        help="Variable with information about y location.",
    )
    parser.add_argument(
        "--filter_dummies",
        type=bool,
        default=False,
        help="Filter tracks that contain dummies.",
    )
    parser.add_argument(
        "--filter_delayed_roots",
        type=bool,
        default=False,
        help="Filter tracks that start late.",
    )
    parser.add_argument(
        "--drop_untracked",
        type=bool,
        default=True,
        help="Drop all false positive tracks from the anndata object.",
    )
    parser.add_argument(
        "--max_search_radius",
        type=int,
        default=100,
        help="Local spatial search radius (pixels).",
    )
    parser.add_argument(
        "--field_size",
        nargs="+",
        default=[2048, 2048],
        help="Height and width in of field.",
    )
    parser.add_argument(
        "--approx",
        type=bool,
        default=False,
        help="Speed up processing on very large datasets.",
    )

    # parser
    args = parser.parse_args(args)

    # run
    run(
        inp=args.inp,
        out=args.out,
        fname=args.fname,
        time_var=args.time_var,
        group_vars=args.group_vars,
        x_loc=args.x_loc,
        y_loc=args.y_loc,
        filter_dummies=args.filter_dummies,
        filter_delayed_roots=args.filter_delayed_roots,
        drop_untracked=args.drop_untracked,
        max_search_radius=args.max_search_radius,
        field_size=args.field_size,
        approx=args.approx,
    )
