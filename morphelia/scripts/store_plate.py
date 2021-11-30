import argparse
import os
import logging
import pickle

import numpy as np

from morphelia.tools import LoadPlate
from morphelia.preprocessing import *
from morphelia.features import *

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

    last_tp = plate.obs['Metadata_Time'].unique()[-1]
    logging.info(f"Feature selection based on last time point: {last_tp}h")
    fixed = plate[plate.obs['Metadata_Time'] == last_tp, :].copy()

    logging.info("Drop nan, duplicates and invariant features.")
    fixed = drop_nan(fixed, verbose=True)
    fixed = drop_duplicates(fixed, verbose=True)
    fixed = drop_invariant(fixed, verbose=True)

    logging.info("Normalize data based on 'ctrl'")
    fixed = normalize(fixed, method='mad_robust',
                      by=None,
                      norm_pop='ctrl',
                      drop_nan=True,
                      verbose=True)

    logging.info("Drope noise")
    fixed = drop_noise(fixed, verbose=True,
                       by=["Metadata_Treatment", "Metadata_Concentration"])

    logging.info("Drop features with near zero variance")
    fixed = drop_near_zero_variance(fixed, verbose=True)

    logging.info("Drop outlier")
    fixed = drop_outlier(fixed, axis=1, verbose=True)

    logging.info("Drop highly correlated values")
    fixed = drop_highly_correlated(fixed, verbose=True, show=False)

    logging.info("SVM-RFE")
    fixed = svm_rfe(fixed, subsample=False, verbose=True)

    logging.info("Select features")
    plate = plate[:, fixed.var_names].copy()

    logging.info("Save new plate with selected features")
    select_name = "plate1.h5ad"
    plate.write(os.path.join(src, select_name))

    with open(os.path.join(src, 'selected_features.ob'), 'wb') as fp:
        pickle.dump(plate.var_names, fp)


if __name__ == "__main__":
    """Implements the commandline tool to convert CellProfiler output
    to hdf5 files."""
    # initiate the arguments parser
    parser = argparse.ArgumentParser(description=f'Plate to hdf5.')

    parser.add_argument('path', type=str, help='path to plate')

    # parse
    args = parser.parse_args()

    exp_path = args.path

    logging.info(f"Loading experiment from source: {exp_path}")

    # run
    main(src=exp_path)
