import argparse
import yaml
import os

from morphelia.utils.morphome import MorphData
from morphelia.utils.pp import aggregate, subsample


def run(inp, files, treatment, out):
    """Read and write CellProfiler output."""
    if not os.path.exists(inp):
        try:
            os.makedirs(inp)
        except:
            raise OSError(f"Input directory does not exist: {inp}")
    elif not os.path.exists(out):
        out = inp

    md = MorphData.from_csv(inp, files=files, obj_delimiter="\t",
                            treat_file=treatment).to_anndata()

    # save file
    md.write(os.path.join(out, 'results.h5ad'))

    # get subsample
    ss = subsample(md, perc=0.1)
    # save subsample
    ss.write(os.path.join(out, 'results_ss_10.h5ad'))

    # get aggregated data
    md = aggregate(md)
    # save aggregated data
    md.write(os.path.join(out, 'results_agg.h5ad'))


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
