import argparse
import os
import logging

import numpy as np
from morphelia.tools import LoadPlate

logging.basicConfig(level=logging.INFO)
logging.getLogger('numexpr').setLevel(logging.WARNING)


def main(src):
    """Read and write CellProfiler output."""

    plate = LoadPlate(src,
                      obj_sfx=".txt",
                      obj_delimiter="\t",
                      treat_file="Treatment")

    plate.load()
    plate = plate.to_anndata()

    # variables that should not contain any nan values
    not_nan = ['Cells_AreaShape_Area', 'Cytoplasm_AreaShape_Area', 'Primarieswithoutborder_AreaShape_Area',
               'Cytoplasm_AreaShape_FormFactor']

    # drop cells that have no cell size
    len_before = len(plate)
    plate = plate[~np.isnan(plate[:, not_nan].X).any(axis=1), :].copy()
    plate = plate[~np.isinf(plate[:, not_nan].X).any(axis=1), :].copy()
    logging.info(f"{len_before - len(plate)} cells with nan values dropped")

    # store raw data
    raw_name = "plate1_raw.h5ad"
    plate.write(os.path.join(src, raw_name))


if __name__ == "__main__":
    """Implements the commandline tool to convert CellProfiler output
    to hdf5 files."""
    # initiate the arguments parser
    parser = argparse.ArgumentParser(description=f'Plate to hdf5.')

    parser.add_argument('-p', '--path', type=str, help='path to plate')

    # parse
    args = parser.parse_args()

    exp_path = args.path

    logging.info(f"Loading experiment from source: {exp_path}")

    # run
    main(src=exp_path)
