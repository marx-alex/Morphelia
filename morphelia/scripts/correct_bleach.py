import argparse
import os
import logging

import anndata as ad
from morphelia.preprocessing import correct_bleaching

logging.basicConfig(level=logging.INFO)
logging.getLogger('numexpr').setLevel(logging.WARNING)


def main(src, fname, channels):
    """Read and write CellProfiler output."""

    plate = ad.read_h5ad(os.path.join(src, fname))

    plate = correct_bleaching(plate,
                              channels,
                              treat_var='Metadata_Treatment',
                              time_var='Metadata_Time',
                              exp_curve='bi',
                              ctrl='ctrl',
                              correct_X=True,
                              ignore_weak_fits=0.5,
                              verbose=True)

    # store data
    data_name = "plate1_bcorr.h5ad"
    plate.write(os.path.join(src, data_name))


if __name__ == "__main__":
    """Preprocess morphological data."""
    # initiate the arguments parser
    parser = argparse.ArgumentParser(description=f'Preprocessing')

    parser.add_argument('-p', '--path', type=str, help='path to plate')
    parser.add_argument('-f', '--fname', type=str, help='file name')
    parser.add_argument('-c', '--channels', nargs='+', default=[])

    # parse
    args = parser.parse_args()

    path = args.path
    fname = args.fname
    channels = args.channels

    logging.info(f"Loading path: {path}, file: {fname}")

    # run
    main(src=path, fname=fname, channels=channels)
