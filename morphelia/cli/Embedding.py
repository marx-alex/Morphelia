import argparse
import yaml
import os
import anndata as ad
import scanpy as sc

from morphelia.plotting import pca_variance_ratio


def run(inp):
    """Calculate embeddings for morphological data."""
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
    inp_data = os.path.join(inp['output'], inp[inp['emb_inp']])
    if not os.path.exists(inp_data):
        raise OSError(f"File with subsample data does not exist: {inp_data}")
    adata = ad.read_h5ad(inp_data)

    # feature agglomeration
    print("Embedding.")
    # PCA
    sc.tl.pca(adata, svd_solver='arpack')
    pca_variance_ratio(adata, save=figdir)

    # compute neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=inp['neighbors'], n_pcs=inp['n_pcs'])

    # UMAP
    if inp['use_paga']:
        sc.tl.paga(adata)
        sc.pl.paga(adata, save=f'_paga.png')
        sc.tl.umap(adata, init_pos='paga')
    else:
        sc.tl.umap(adata)

    # write file
    print(f"Write file to {inp['output']}")
    adata.write(os.path.join(inp['output'], inp['emb_name']))


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
