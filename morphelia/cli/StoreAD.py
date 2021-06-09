import argparse
import yaml
import os
import re

import anndata as ad

from morphelia.utils.morphome import MorphData
from morphelia.utils.pp import aggregate, subsample


def run(inp, files, treatment, out):
    """Read and write CellProfiler output."""
    if not os.path.exists(inp):
        try:
            os.makedirs(inp)
        except OSError:
            print(f"Input directory does not exist: {inp}")
    elif not os.path.exists(out):
        out = inp

    # initialize MorphData and save batches directly to disk
    # otherwise memory problems are expected
    MorphData().from_csv(inp, files=files, obj_delimiter="\t",
                         treat_file=treatment, to_disk='batch',
                         output=out)

    # get all hdf5 files
    files = []
    for filename in os.listdir(out):
        if re.match(".*batch\d*.h5ad", filename):
            files.append(filename)

    # create subsamples and aggregations
    subsamples = []
    aggs = []
    for file in files:
        adata = ad.read_h5ad(os.path.join(out, file))
        # subsample
        edit = subsample(adata, perc=0.1)
        # edit.write(os.path.join(out, f'subs_{file}'))
        subsamples.append(edit)
        # aggregate
        edit = aggregate(adata)
        # edit.write(os.path.join(out, f'agg_{file}'))
        aggs.append(edit)

    # concatenate
    subsamples = subsamples[0].concatenate(subsamples[1:])
    subsamples.write(os.path.join(out, "morph_data_ss_10.h5ad"))
    aggs = aggs[0].concatenate(aggs[1:])
    aggs.write(os.path.join(out, "morph_data_agg.h5ad"))


def main(args=None):
    """Implements the commandline tool to convert CellProfiler output
    to hdf5 files."""
    # initiate the arguments parser
    parser = argparse.ArgumentParser(description=f'Convert cellprofiler to hdf5.')

    parser.add_argument('config', type=str, help='config file in yaml format.')

    # parse
    args = parser.parse_args(args)

    yml_path = args.config

    with open(yml_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    # run
    run(data['input'], data['files'], data['treatment'], data['output'])
