import argparse
import os
import logging
from pathlib import Path
import sys

import anndata as ad

import morphelia

logger = logging.getLogger("C-More")
logging.basicConfig()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def main(
    inp,
    out,
    by=("BatchNumber", "PlateNumber"),
    well_var="Metadata_Well",
    treat_var="Metadata_Treatment",
    qc=True,
    min_cells=300,
):
    """C-More recipe."""

    # check by
    if isinstance(by, str):
        by = [by]
    else:
        assert isinstance(
            by, (list, tuple)
        ), f"by should be a string, list or tuple, got {by}"

    # load data
    logger.info(f"Read AnnData object: {inp}")
    adata = ad.read_h5ad(inp)

    logger.info("Clean data...")
    # nan values
    adata = morphelia.pp.drop_nan(adata, axis=0, min_nan_frac=0.5, verbose=True)
    adata = morphelia.pp.drop_nan(adata, axis=1, verbose=True)

    # duplicated features / cells
    adata = morphelia.pp.drop_duplicates(adata, axis=1, verbose=True)
    adata = morphelia.pp.drop_duplicates(adata, axis=0, verbose=True)

    # invariant features / cells
    adata = morphelia.pp.drop_invariant(adata, axis=0, verbose=True)
    adata = morphelia.pp.drop_invariant(adata, axis=1, verbose=True)

    logger.info("Quality control.")
    # aggregate
    agg = morphelia.pp.aggregate(
        adata, by=by + [well_var], qc=qc, min_cells=min_cells, drop_qc=False
    )
    # quality control
    morphelia.pl.plot_plate(agg, color="Metadata_Cellnumber", save=out)
    if qc:
        agg, qc_pops = agg
        pops = adata.obs[by + [well_var]].values
        qc_mask = (pops[:, None] == qc_pops).all(-1).any(-1)
        logger.info(f"Drop following populations after quality control: {qc_pops}")
        adata = adata[~qc_mask, :]

    logger.info("Thresholding: Debris, cell type and cell cycle")

    # debris filter
    nucl_area = "Primarieswithoutborder_AreaShape_Area"
    cell_area = "Cells_AreaShape_Area"
    debris_dist = adata[:, cell_area].X / adata[:, nucl_area].X

    # colors for dead and vital in plot
    class_colors = ["#DD4A48", "#C0D8C0"]
    class_labels = ["dead", "vital"]

    adata = morphelia.pp.assign_by_threshold(
        adata,
        dist=debris_dist,
        by=by,
        new_var="Debris",
        method="otsu",
        max_val=4,
        threshold_labels="Debris Threshold",
        make_plot=True,
        plt_xlim=(0, 50),
        class_colors=class_colors,
        class_labels=class_labels,
        xlabel="Cell Area / Nucleus Area",
        save=out,
    )
    # only keep vital cells
    adata = adata[adata.obs["Debris"] == "vital", :].copy()

    # cell type filter
    median_intens = "Primarieswithoutborder_Intensity_MedianIntensity_DNA"

    class_colors = ["#ECB390", "#886F6F"]
    class_labels = ["cardiomyocyte", "fibroblast"]

    adata = morphelia.pp.assign_by_threshold(
        adata,
        dist=median_intens,
        by=by,
        new_var="Cell_Type",
        method="otsu",
        threshold_labels="Cell Type Threshold",
        make_plot=True,
        class_colors=class_colors,
        class_labels=class_labels,
        xlabel="Median DAPI Intensity",
        save=out,
    )

    # only keep cardiomyocytes
    adata = adata[adata.obs["Cell_Type"] == "cardiomyocyte", :].copy()

    # cell cycle filter
    integr_intens = "Primarieswithoutborder_Intensity_IntegratedIntensity_DNA"

    class_colors = ["#CE7BB0", "#6867AC"]
    class_labels = ["G1", "S/G2"]

    adata = morphelia.pp.assign_by_threshold(
        adata,
        dist=integr_intens,
        by=by,
        new_var="Cell_Cycle",
        method="otsu",
        threshold_labels="Cell Cycle Threshold",
        make_plot=True,
        plt_xlim=(0, 1000),
        class_labels=class_labels,
        class_colors=class_colors,
        xlabel="Integrated DAPI Intensity",
        save=out,
    )

    logger.info("Normalize data based on 'ctrl'")
    adata = morphelia.pp.normalize(
        adata,
        method="mad_robust",
        by=by,
        pop_var="Metadata_Treatment",
        norm_pop="ctrl",
        drop_nan=True,
        verbose=True,
    )

    logger.info("Feature selection...")
    agg = morphelia.pp.aggregate(adata, by=by + [well_var], method="median")

    logger.info("LMEM feature selection.")
    agg = morphelia.ft.feature_lmem(
        agg, treat_var=treat_var, rand_var=by, fixed_var="Metadata_Concentration"
    )
    n_significant_features = agg.var["lme_combined_mask"].sum()
    logger.info(
        f"{n_significant_features}/{len(agg)} features found to be significant."
    )
    agg = agg[:, agg.var["lme_combined_mask"]]

    logger.info("Feature agglomeration...")
    best_k = morphelia.ft.estimate_k(
        agg, cluster_range=(2, 100), make_plot=True, save=out
    )
    logger.info(f"Best k: {best_k}")
    agg = morphelia.ft.feature_agglo(agg, k=best_k)

    fname = Path(inp).stem
    fname = os.path.join(out, f"{fname}_cmore.h5ad")
    logging.info(f"Save aggregated data to {fname}")
    agg.write(fname)


if __name__ == "__main__":
    """C-More recipe."""
    # initiate the arguments parser
    parser = argparse.ArgumentParser(description="Split data.")

    parser.add_argument(
        "-i",
        "--inp",
        type=str,
        help="AnnData file.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="./",
        help="Output directory.",
    )
    parser.add_argument(
        "-b",
        "--by",
        nargs="+",
        default=["BatchNumber", "PlateNumber"],
        help="Plate specific pointers.",
    )
    parser.add_argument(
        "--well_var", type=str, default="Metadata_Well", help="Well variable."
    )
    parser.add_argument(
        "--treat_var",
        type=str,
        default="Metadata_Treatment",
        help="Treatment variable.",
    )
    parser.add_argument(
        "--qc",
        type=bool,
        default=True,
        help="Add quality control based on cell count per well.",
    )
    parser.add_argument(
        "--min_cells", type=int, default=200, help="Minimum cell count per well."
    )

    # parser
    args = parser.parse_args()

    # run
    main(
        inp=args.inp,
        out=args.out,
        by=args.by,
        well_var=args.well_var,
        treat_var=args.treat_var,
        qc=args.qc,
        min_cells=args.min_cells,
    )
