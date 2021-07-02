import argparse
import yaml
import os
import anndata as ad
from morphelia.utils.pp import *
from morphelia.utils.feature_agglo import feature_agglo


def run(inp):
    """Preprocess morphological annotated data."""
    # where to save figures
    figdir = os.path.join(inp['output'], './figures/')

    # load data
    inp_data = os.path.join(inp['output'], inp[inp['agglo_inp']])
    if not os.path.exists(inp_data):
        raise OSError(f"File with subsample data does not exist: {inp_data}")
    adata = ad.read_h5ad(inp_data)

    # feature agglomeration
    print("Feature agglomeration.")
    cluster_range = (inp['cluster_min'], inp['cluster_max'])
    adata = feature_agglo(adata, show=True, save=figdir, group_by=inp['treat_var'],
                          k=inp['k'], cluster_range=cluster_range)

    # write file
    print(f"Write file to {inp['output']}")
    adata.write(os.path.join(inp['output'], inp['agglo_name']))


def main(args=None):
    """Implements the commandline tool to Preprocess an AnnData object
    with morphological data from single cells."""
    # initiate the arguments parser
    parser = argparse.ArgumentParser(description=f'Preprocess data.')

    parser.add_argument('config', type=str, help='config file in yaml format.')

    # parse
    args = parser.parse_args(args)

    yml_path = args.config

    with open(yml_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    # run
    run(data)
