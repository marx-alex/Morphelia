import argparse
import os
import logging
import sys

from morphelia.tools.load_project import LoadPlate

logger = logging.getLogger('PlateToAD')
logging.basicConfig()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def run(inp, out,
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
        treat_delimiter=","):
    """Stores cell profiler from a single plate to AnnData object.

    Input file structure should be as following:

        |----plate1
        |     |----Cells.csv
        |     |----Primarieswithoutborder.csv
        |     |----Cytoplasm.csv
        |
        |----plate2
        ...
        |----platen

    """
    if not os.path.exists(out):
        os.makedirs(out)
    if isinstance(files, str):
        files = [files]

    # load plate
    logger.info(f'Processing plate at location {inp}...')

    plate = LoadPlate(inp,
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
                      treat_delimiter=treat_delimiter)

    plate.load()  # --> merge and load to pandas
    plate = plate.to_anndata()  # --> convert to anndata
    plate_ix = 1
    while os.path.isfile(os.path.join(out, f"plate{plate_ix}.h5ad")):
            plate_ix += 1
    plate.write(os.path.join(out, f"plate{plate_ix}.h5ad"))


def main(args=None):
    """Implements the commandline tool to convert CellProfiler output
    to hdf5 files."""
    # initiate the arguments parser
    parser = argparse.ArgumentParser(description=f'Convert CellProfiler output to AnnData.')

    parser.add_argument('-i', '--inp', type=str,
                        help='Input directory to Cellprofiler output for a whole experiment.')
    parser.add_argument('-f', '--files', nargs='+', default=["Cells", "Primarieswithoutborder", "Cytoplasm"],
                        help='Filenames of CellProfiler output files that should be stored.')
    parser.add_argument('-o', '--out', type=str, default='./adata/',
                        help='Output directory to Cellprofiler output for a whole experiment.')
    parser.add_argument('--treat_file', type=str, default="Treatment",
                        help="Name of treatment file (without suffix).")
    parser.add_argument('--treat_sfx', type=str, default=".csv",
                        help="Suffix of treatment file.")
    parser.add_argument('--obj_sfx', type=str, default=".csv",
                        help="Suffix of object files.")
    parser.add_argument('--obj_well_var', type=str, default="Metadata_Well",
                        help="Name of well variable in CellProfiler objects.")
    parser.add_argument('--treat_well_var', type=str, default="well",
                        help="Name of well variable in the treatment file.")
    parser.add_argument('--meta_var', type=str, default='Metadata',
                        help="Metadata label in CellProfiler outputs.")
    parser.add_argument('--image_var', type=str, default='ImageNumber',
                        help="Name of image variable in CellProfiler outputs.")
    parser.add_argument('--obj_var', type=str, default='ObjectNumber',
                        help='Name of object variable in CellProfiler outputs.')
    parser.add_argument('--obj_delimiter', type=str, default="\t",
                        help='Delimiter of values from CellProfiler outputs.')
    parser.add_argument('--treat_delimiter', type=str, default=",",
                        help="Delimiter for values of the treatment file.")

    # parser
    args = parser.parse_args(args)

    # run
    run(inp=args.inp,
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
        treat_delimiter=args.treat_delimiter)