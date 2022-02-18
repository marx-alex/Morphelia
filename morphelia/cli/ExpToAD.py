import argparse
import os
import pathlib
import logging
import sys

from morphelia.tools.load_project import LoadPlate

logger = logging.getLogger('ExpToAD')
logging.basicConfig()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def run(inp, out,
        files=("Cells", "Primarieswithoutborder", "Cytoplasm"),
        merge_plates=True,
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

    datadict = _create_datadict(inp, files)

    # store plates
    plates = []

    # load every plate independently
    for batch_i, batch in enumerate(sorted(datadict.keys())):
        logger.info(f'Processing batch {batch}...')

        # iterate over plates
        for plate_i, plate_path in enumerate(sorted(datadict[batch].keys())):
            logger.info(f'Processing plate at location {plate_path}...')

            plate = LoadPlate(plate_path,
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
            plate.obs['BatchNumber'] = batch_i + 1
            plate.obs['PlateNumber'] = plate_i + 1
            plate.write(os.path.join(out, f"{batch}_plate{plate_i}.h5ad"))
            if merge_plates:
                plates.append(plate)

    # concatenate
    if len(plates) > 0:
        plates = plates[0].concatenate(plates[1:])
        plates.write(os.path.join(out, "adata.h5ad"))
    else:
        logger.info("Do not merge plates.")


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
    parser.add_argument('--merge_plates', type=bool, default=True,
                        help="Whether to merge all plates to a single AnnData object.")
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
        merge_plates=args.merge_plates,
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


def _create_datadict(exp_path, files):
    """Create dictionary with data from a image-based experiment with
    experiment, trial, plate and object data as keys.

    Args:
        exp_path (str): Path to experiment data.
        files (iterable): Object files that store data.

    Returns:
        (dict)
    """
    datadict = {}
    for path, subdirs, filenames in os.walk(pathlib.Path(exp_path)):
        if files is not None:
            csv_files = [filename for filename in filenames
                         if (filename.endswith(".csv") or filename.endswith(".txt"))
                         and filename in files]
        else:
            csv_files = [filename for filename in filenames
                         if (filename.endswith(".csv") or filename.endswith(".txt"))]

        if len(csv_files) != 0:
            batch = path.split(os.sep)[-2]
            if batch not in datadict.keys():
                datadict[batch] = {}
            datadict[batch][path] = csv_files

    return datadict