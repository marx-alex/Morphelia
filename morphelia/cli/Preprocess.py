import argparse
import yaml
import scanpy as sc
import os
from morphelia.preprocessing.pp import *
from morphelia.preprocessing.debris import filter_debris
from morphelia.extern.pl import highly_variable_genes


def run(inp):
    """Preprocess morphological annotated data."""
    # where to save figures
    figdir = os.path.join(inp['output'], './figures/')
    # scanpy settings
    sc.settings.autosave = True
    sc.settings.verbosity = 1
    sc.settings.file_format_figs = '.png'
    sc.settings.autoshow = False
    sc.settings.autosave = True
    sc.settings.figdir = figdir

    # load data
    inp_data = os.path.join(inp['output'], inp[inp['pp_inp']])
    if not os.path.exists(inp_data):
        raise OSError(f"File with subsample data does not exist: {inp_data}")
    adata = ad.read_h5ad(inp_data)

    # drop duplicate features
    print("[STEP 1] Get unique features.")
    adata = unique_feats(adata, verbose=True)

    # drop nan
    adata = drop_nan(adata, verbose=True)

    # filter cells
    print("[STEP 2] Filter cells.")

    for shape, intens in list(zip(inp['area_filter'], inp['intens_filter'])):

        sc.pl.violin(adata, shape,
                     jitter=0.4, save=f'_raw_{shape}.png')
        sc.pl.violin(adata, intens,
                     jitter=0.4, save=f'_filtered_{intens}.png')

        sc.pl.scatter(adata, x=shape, y=intens, save=f'_raw_{shape}_{intens}.png')

        adata = filter_cells(adata, shape, side='both')
        adata = filter_cells(adata, intens, side='right')

        sc.pl.scatter(adata, x=shape, y=intens, save=f'_filtered_{shape}_{intens}.png')

    # filter debris
    print("[STEP 3] Filter debris.")
    if inp['filter_debris']:
        adata = filter_debris(adata, show=True, save=figdir, verbose=True)
    else:
        print('Skipped.')

    # set raw attribute
    adata.raw = adata

    # min-max scaling
    print("[STEP 4] Min-max Scaling.")
    # drop nan
    adata = drop_nan(adata, verbose=True)
    adata = min_max_scaler(adata)

    # log-transform
    print("[STEP 5] Logarithmic transformation.")
    # drop nan
    adata = drop_nan(adata, verbose=True)
    sc.pp.log1p(adata)

    # drop nan
    adata = drop_nan(adata, verbose=True)

    # batch correction
    print("[STEP 6] Batch correction.")
    sc.pp.combat(adata, key=inp['batch_id'])

    # get highly variable features
    print("[STEP 7] Select highly variable features.")
    sc.pp.highly_variable_genes(adata)
    highly_variable_genes(adata, save='_hvg.png')

    # select highly variable features
    adata = adata[:, adata.var.highly_variable]

    # get z-scores
    print("[STEP 8] Building z-scores.")
    adata = z_transform(adata, clip=10)

    # write file
    print("[STEP 9] Write file.")
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
