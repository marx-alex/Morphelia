import argparse
import os
from pathlib import Path
from collections import defaultdict
import logging

from morphelia.tools.load_project import LoadPlate

logger = logging.getLogger("StorePlates")
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def run(
    inp,
    out,
    files=("Cells", "Primarieswithoutborder", "Cytoplasm"),
    obj_sfx=".csv",
    obj_well_var="Metadata_Well",
    meta_var="Metadata",
    image_var="ImageNumber",
    obj_var="ObjectNumber",
    obj_delimiter=",",
    treat_file="Treatment",
    treat_well_var="well",
    treat_sfx=".csv",
    treat_delimiter=",",
):
    """Stores cell profiler from a single experiment to AnnData object.

    Input file structure should be as following:

    exp
        |----batch1
        |       |----plate1
        |       |       |----Cells.csv
        |       |       |----Primarieswithoutborder.csv
        |       |       |----Cytoplasm.csv
        |       |
        |       |----plate2
        |       ...
        |       |----platen
        |
        |----batch2
        ...
        |----batchn

    """
    if not os.path.exists(out):
        os.makedirs(out)
    if isinstance(files, str):
        files = [files]

    datadict = _batch_plate_paths(inp, files)

    # load every plate independently
    for batch_i, batch in enumerate(sorted(datadict.keys())):
        logger.info(f"Processing batch {batch}...")

        # iterate over plates
        for plate_i, plate_path in enumerate(datadict[batch]):
            logger.info(f"Processing plate at location {plate_path}...")

            plate = LoadPlate(
                plate_path,
                filenames=files,
                obj_sfx=obj_sfx,
                obj_well_var=obj_well_var,
                meta_var=meta_var,
                image_var=image_var,
                obj_var=obj_var,
                obj_delimiter=obj_delimiter,
                treat_file=treat_file,
                treat_well_var=treat_well_var,
                treat_sfx=treat_sfx,
                treat_delimiter=treat_delimiter,
            )

            plate.load()  # --> merge and load to pandas
            plate = plate.to_anndata()  # --> convert to anndata
            plate.obs["BatchNumber"] = batch_i + 1
            plate.obs["PlateNumber"] = plate_i + 1
            plate.write(os.path.join(out, f"{batch}_plate_{plate_i + 1:04d}.h5ad"))


def main(args=None):
    """Implements the commandline tool to convert CellProfiler output
    to hdf5 files."""
    # initiate the arguments parser
    parser = argparse.ArgumentParser(
        description="Convert CellProfiler output to AnnData."
    )

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
        help="Output directory.",
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        default=["Cells", "Primarieswithoutborder", "Cytoplasm"],
        help="Filenames of CellProfiler output files that should be stored.",
    )
    parser.add_argument(
        "--treat_file",
        type=str,
        default="Treatment",
        help="Name of treatment file (without suffix).",
    )
    parser.add_argument(
        "--treat_sfx",
        type=str,
        default=".csv",
        help="Suffix of treatment file.",
    )
    parser.add_argument(
        "--obj_sfx", type=str, default=".txt", help="Suffix of object files."
    )
    parser.add_argument(
        "--obj_well_var",
        type=str,
        default="Metadata_Well",
        help="Name of well variable in CellProfiler objects.",
    )
    parser.add_argument(
        "--treat_well_var",
        type=str,
        default="well",
        help="Name of well variable in the treatment file.",
    )
    parser.add_argument(
        "--meta_var",
        type=str,
        default="Metadata",
        help="Metadata label in CellProfiler outputs.",
    )
    parser.add_argument(
        "--image_var",
        type=str,
        default="ImageNumber",
        help="Name of image variable in CellProfiler outputs.",
    )
    parser.add_argument(
        "--obj_var",
        type=str,
        default="ObjectNumber",
        help="Name of object variable in CellProfiler outputs.",
    )
    parser.add_argument(
        "--obj_delimiter",
        type=str,
        default="\t",
        help="Delimiter of values from CellProfiler outputs.",
    )
    parser.add_argument(
        "--treat_delimiter",
        type=str,
        default=",",
        help="Delimiter for values of the treatment file.",
    )

    # parser
    args = parser.parse_args(args)

    # run
    run(
        inp=args.inp,
        out=args.out,
        files=args.files,
        obj_sfx=args.obj_sfx,
        obj_well_var=args.obj_well_var,
        meta_var=args.meta_var,
        image_var=args.image_var,
        obj_var=args.obj_var,
        obj_delimiter=args.obj_delimiter,
        treat_file=args.treat_file,
        treat_well_var=args.treat_well_var,
        treat_sfx=args.treat_sfx,
        treat_delimiter=args.treat_delimiter,
    )


def _batch_plate_paths(path, files, sfx=".txt"):
    """Get all batch and plate paths.

    Args:
        path (str): Path to experiment data.
        files (iterable): Object files that store data.
        sfx (str): Suffix of files.

    Returns:
        (dict)
    """
    if isinstance(files, str):
        files = [files]
    assert isinstance(
        files, (list, tuple)
    ), f"File must be string, list or tuple. Instead got {type(files)}."
    data = defaultdict(list)
    for file in files:
        for file_path in Path(path).rglob(f"*{file}*{sfx}"):
            batch = file_path.parts[-3]
            data[batch].append(file_path.parent)

    # make plates unique
    for batch in data:
        data[batch] = list(set(data[batch]))

    return data
