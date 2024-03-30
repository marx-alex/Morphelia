from typing import Optional, Sequence
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def rank_feats_groups(
    adata,
    feat_names: Optional[Sequence] = None,
    n_cols: int = 4,
    show: bool = True,
    save: Optional[str] = None,
):
    """Plot feature ranks for each group and test.
    Use morphelia.ft.rank_feats_groups beforehand.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated dataset with multidimensional morphological data
    feat_names : sequence, optional
        Name of features to show in the plot
    n_cols : int
        Number of columns
    show : bool
        Show and return axes
    save : str, optional
        Path where to save figure

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Only if show is False
    ax : matplotlib.pyplot.axes
    """
    result = adata.uns["rank_feats_groups"]["result"]
    groups = adata.uns["rank_feats_groups"]["params"]["groups"]
    method = adata.uns["rank_feats_groups"]["params"]["method"]
    feats = result.columns
    ref = adata.uns["rank_feats_groups"]["params"]["reference"]
    if ref is None:
        ref = "rest"
    if feat_names is None:
        feat_names = feats
    else:
        assert len(feat_names) == len(
            feats
        ), f"Length of feat_names must ({len(feat_names)}) must match number of features ({len(feats)})"
    n_groups = len(groups)

    # Y-Label
    if method == "wilcoxon":
        ylabel = "Wilcoxon rank-sum statistic"
    elif method == "t-test":
        ylabel = "t-statistic"
    else:
        ylabel = "Score"

    n_rows = n_groups // n_cols
    if (n_groups % n_cols) > 0:
        n_rows += 1

    width = 2 * n_cols
    height = 2 * n_rows

    with sns.axes_style("whitegrid"):

        fig, axs = plt.subplots(
            n_rows, n_cols, sharex=True, sharey=True, figsize=(width, height)
        )

        for i, ax in enumerate(axs.reshape(-1)):

            if i < n_groups:
                group_result = result.loc[(groups[i], "score")].abs()
                ranks = np.argsort(group_result)

                # phantom scatter
                ax.scatter(x=np.arange(len(ranks)), y=group_result.iloc[ranks], alpha=0)

                for j, rank in enumerate(reversed(ranks)):
                    ax.annotate(
                        feat_names[rank],
                        xy=(j, group_result.iloc[rank]),
                        rotation=90,
                        fontsize="small",
                    )

                    ax.set(
                        title=f"{groups[i]} vs. {ref}",
                        xlabel="",
                        ylabel="",
                        box_aspect=1,
                    )
            else:
                ax.axis("off")

        fig.supxlabel("Ranking")
        fig.supylabel(ylabel)
        plt.tight_layout()

    # save
    if save is not None:
        try:
            plt.savefig(os.path.join(save, "rank_feats_groups.png"))
        except OSError:
            print(f"Can not save figure to {save}.")

    if show:
        plt.show()
        return ax

    return fig, ax
