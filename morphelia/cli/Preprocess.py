import argparse
import yaml
import os
import anndata as ad
from morphelia.preprocessing import *
from morphelia.features import *


def run(inp):
    """Preprocess morphological annotated data."""
    # where to save figures
    figdir = os.path.join(inp['output'], './figures/')

    # load data
    inp_data = os.path.join(inp['output'], inp[inp['pp_inp']])
    if not os.path.exists(inp_data):
        raise OSError(f"File with subsample data does not exist: {inp_data}")
    adata = ad.read_h5ad(inp_data)

    # clean data
    print("[STEP 1] Clean data: Drop features containing Nan values, "
          "duplicated features or invariant features.")
    adata = drop_nan(adata, verbose=True)
    adata = drop_duplicates(adata, verbose=True)
    adata = drop_invariant(adata, verbose=True)

    # filter debris
    print("[STEP 2] Filter debris.")
    if inp['filter_debris']:
        adata = filter_debris(adata, show=True, save=figdir, verbose=True)
    else:
        print('Skipped.')

    # filter cells
    print("[STEP 3] Normalize data.")

    adata = normalize(adata, method=inp['norm_method'],
                      by=inp['batch_id'],
                      pop_var=inp['treat_var'],
                      norm_pop=inp['ctrl_name'],
                      drop_nan=True,
                      verbose=True)

    print("[STEP 4] Drop noise.")

    adata = drop_noise(adata, verbose=True,
                       by=inp['condition_group'])

    print("[STEP 5] Drop features with near zero variance.")

    adata = drop_near_zero_variance(adata, verbose=True)

    print("[STEP 6] Drop outlier.")

    adata = drop_outlier(adata, verbose=True)

    print("[STEP 6] Drop highly correlated features.")
    adata = drop_highly_correlated(adata,
                                   thresh=0.95,
                                   verbose=True,
                                   show=True,
                                   save=figdir)

    # write file
    print("[STEP 7] Write file.")
    adata.write(os.path.join(inp['output'], inp['pp_name']))


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
