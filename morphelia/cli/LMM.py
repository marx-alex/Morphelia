import argparse
import yaml
import os
import anndata as ad
from morphelia.stats import lmm_feat_select


def run(inp):
    """Preprocess morphological annotated data."""
    # where to save figures
    figdir = os.path.join(inp['output'], './figures/')

    # load data
    inp_data = os.path.join(inp['output'], inp[inp['lmm_inp']])
    if not os.path.exists(inp_data):
        raise OSError(f"File with subsample data does not exist: {inp_data}")
    adata = ad.read_h5ad(inp_data)

    # feature agglomeration
    print("Feature selection with LMM.")
    adata = lmm_feat_select(adata, time_series=inp['time_series'], show=True, save=figdir)

    # filter
    if inp['drop_low_sign']:
        adata = adata[:, adata.var['wald_p'] < 0.05]

    # write file
    print(f"Write file to {inp['output']}")
    adata.write(os.path.join(inp['output'], inp['lmm_name']))


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
