import argparse
import yaml
import os
import anndata as ad
from morphelia.preprocessing.pseudostitch import pseudostitch


def run(inp):
    """Preprocess morphological annotated data."""

    # load data
    inp_data = os.path.join(inp['output'], inp[inp['stitched_inp']])
    if not os.path.exists(inp_data):
        raise OSError(f"File with subsample data does not exist: {inp_data}")
    adata = ad.read_h5ad(inp_data)

    # feature agglomeration
    print("Pseudostitch.")
    tile_grid = (inp['tile_rows'], inp['tile_cols'])
    adata = pseudostitch(adata, tile_grid=tile_grid, img_overlap=inp['img_overlap'], verbose=True)

    # write file
    print(f"Write file to {inp['output']}")
    adata.write(os.path.join(inp['output'], inp['stitched_name']))


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