import matplotlib.pyplot as plt
import numpy as np
import os


def rank_proliferation(
    adata,
    rank_by="children",
    track_id="Metadata_Track",
    root_id="Metadata_Track_Root",
    gen_id="Metadata_Gen",
    show=False,
    return_fig=False,
    clip=0.05,
    save=False,
):
    """
    Rank all available cell tracks by their proliferative activity.
    Track cells beforehand.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        rank_by (str): Method to rank tracks.
            gen: Rank by number of generations.
            children: rank by number of children.
        track_id (str): Variable name for tracks.
        root_id (str): Variable name for track roots.
        gen_id (str): Variable name for generation.
        show (bool)
        return_fig (bool)
        clip (float): Clip ranking until specified percentile for plotting.
        save (str): Path where to save plot.

    Returns:
        adata.uns[f'prolif_rank_{rank_by}']: Ranking.
    """
    avail_methods = ["gen", "children"]
    rank_by = rank_by.lower()
    assert (
        rank_by in avail_methods
    ), f"Rank by not available. Available methods: {avail_methods}."

    if rank_by == "gen":
        ylabel = "Number of generations"
        ranks = (
            adata.obs[[root_id, gen_id]]
            .groupby(root_id)
            .apply(lambda x: x[gen_id].max())
        )

    else:
        ylabel = "Number of children"
        ranks = (
            adata.obs[[root_id, track_id]]
            .groupby(root_id)
            .apply(lambda x: len(x[track_id].unique()))
        )

    ranks = ranks.sort_values(ascending=False)

    if show:
        if clip is not None:
            ranks_clipped = ranks.iloc[: int(len(ranks) * clip)]
        else:
            clip = 1
            ranks_clipped = ranks
        x = np.linspace(0, clip * 100, len(ranks_clipped))
        width = (clip * 100) / len(ranks_clipped)
        fig, axs = plt.subplots(figsize=(10, 5))

        axs.bar(
            x,
            ranks_clipped,
            width=width,
            facecolor="#74959A",
            edgecolor="#74959A",
        )
        axs.set_xlabel("Percentile")
        axs.set_title(ylabel)

        # save
        if save:
            try:
                plt.savefig(os.path.join(save, f"prolif_rank_{rank_by}.png"))
            except OSError:
                print(f"Can not save figure to {save}.")

        if return_fig:
            return adata, fig

    adata.uns[f"prolif_rank_{rank_by}"] = ranks

    return adata
