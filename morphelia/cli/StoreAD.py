import argparse
import yaml
import os
import re

import anndata as ad

from morphelia.tools.morphome import MorphData
from morphelia.preprocessing import aggregate, subsample


def run(inp, agg_name, ss_name, files, treatment, out, store):
    """Read and write CellProfiler output."""
    if not os.path.exists(inp):
        raise OSError(f"Input directory does not exist: {inp}")
    elif not os.path.exists(out):
        try:
            os.makedirs(out)
        except ValueError:
            print(f"Output directory can not be made: {out}")

    # initialize MorphData and save batches directly to disk
    # otherwise memory problems are expected
    print("Load data from CSV and store plates directly to disk.")
    MorphData().from_csv(inp, files=files, obj_delimiter="\t",
                         treat_file=treatment, to_disk=store,
                         output=out)

    # get all hdf5 files
    files = []
    for filename in os.listdir(out):
        if re.match(f".*{store}\d*.h5ad", filename):
            files.append(filename)

    subsamples = []
    aggs = []
    for file in files:
        adata = ad.read_h5ad(os.path.join(out, file))
        # subsample
        edit = subsample(adata, perc=0.1)
        subsamples.append(edit)
        # aggregate
        edit = aggregate(adata)
        aggs.append(edit)

    # concatenate
    subsamples = subsamples[0].concatenate(subsamples[1:])
    subsamples.write(os.path.join(out, ss_name))
    aggs = aggs[0].concatenate(aggs[1:])
    aggs.write(os.path.join(out, agg_name))


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
    run(inp=data['input'],
        agg_name=data['agg_name'],
        ss_name=data['ss_name'],
        files=data['files'],
        treatment=data['treatment'],
        out=data['output'],
        store=data['to_disk'])
