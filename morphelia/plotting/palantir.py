import matplotlib.pyplot as plt
import matplotlib as mtl
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import scanpy as sc


def plot_palantir_results(adata,
                          emb='X_tsne',
                          cmap=None,
                          show=True):
    """ Plot Palantir results on embedding
    """
    assert emb in adata.obsm, f'emb not found in .obsm: {emb}'

    annotations = ['pseudotime', 'entropy', 'branch']
    assert all(ann in adata.obs.columns for ann in annotations), "AnnData object is not annotated. " \
                                                                 "Use Palantir.annotate_data()."
    terminal_states = adata.uns['palantir']['terminal_states']
    branch_prob_anns = [f"branch_prob_{ts}" for ts in terminal_states]
    assert all(ann in adata.obs.columns for ann in branch_prob_anns), "AnnData object is not annotated. " \
                                                                      "Use Palantir.annotate_data()."

    # set up figure
    n_branches = len(branch_prob_anns)
    n_cols = 6
    n_rows = int(np.ceil(n_branches / n_cols))
    fig = plt.figure(figsize=[2 * n_cols, 2 * (n_rows + 2)])
    gs = plt.GridSpec(
        n_rows + 2, n_cols, height_ratios=np.append([0.75, 0.75], np.repeat(1, n_rows)), hspace=0.4, wspace=0.4
    )

    if cmap is None:
        cmap = mtl.cm.plasma

    # Pseudotime
    ax = plt.subplot(gs[:2, 0:2])
    ax = sc.pl.embedding(adata, basis=emb,
                         color='pseudotime', ax=ax, cmap=cmap,
                         show=False)
    ax.set_axis_off()
    ax.set_title("Pseudotime")

    # Entropy
    ax = plt.subplot(gs[:2, 2:4])
    ax = sc.pl.embedding(adata, basis=emb,
                         color='entropy', ax=ax, cmap=cmap,
                         show=False)
    ax.set_axis_off()
    ax.set_title("Differentiation potential")

    for i, ts in enumerate(terminal_states):
        branch_label = f"branch_prob_{ts}"
        row = int(np.floor(i / n_cols))
        ax = plt.subplot(gs[row + 2, np.remainder(i, n_cols)])
        ax = sc.pl.embedding(adata, basis=emb,
                             color=branch_label, ax=ax, cmap=cmap,
                             show=False)
        ax.scatter(adata.obsm[emb][ts, 0],
                   adata.obsm[emb][ts, 1],
                   s=35, c='black', marker='v')
        ax.set_axis_off()
        ax.set_title(f"Branch {i}", fontsize=10)

    if show:
        plt.show()
    else:
        return fig


def plot_branches(adata,
                  emb='X_tsne',
                  treat_var='Metadata_Treatment',
                  dist=None,
                  cutoff=0.7,
                  pie_explode=True,
                  pie_labels=False,
                  pie_pct=True,
                  cmap=None,
                  show=True):
    """
    Plot branches.

    :param dist:
    :param cutoff:
    :param pie_pct:
    :param pie_labels:
    :param cmap:
    :param show:
    :param pie_explode:
    :param treat_var:
    :param adata:
    :param emb:
    :return:
    """
    assert emb in adata.obsm, f'emb not found in .obsm: {emb}'

    annotations = ['pseudotime', 'entropy', 'branch']
    assert all(ann in adata.obs.columns for ann in annotations), "AnnData object is not annotated. " \
                                                                 "Use Palantir.annotate_data()."
    terminal_states = adata.uns['palantir']['terminal_states']
    branch_prob_anns = [f"branch_prob_{ts}" for ts in terminal_states]
    assert all(ann in adata.obs.columns for ann in branch_prob_anns), "AnnData object is not annotated. " \
                                                                      "Use Palantir.annotate_data()."

    assert treat_var in adata.obs.columns, f"treat_var not in .obs: {treat_var}"

    start_cell = int(adata.uns['palantir']['start_cell'])

    # set up figure
    n_branches = len(branch_prob_anns)
    n_cols = 4
    n_rows = int(np.ceil(n_branches / n_cols))
    fig = plt.figure(figsize=[2 * n_cols, 2 * (n_rows + 2)])
    gs = plt.GridSpec(
        n_rows + 2, n_cols, height_ratios=np.append([0.75, 0.75], np.repeat(1, n_rows)), hspace=0.4, wspace=0.4
    )
    
    if cmap is None:
        cmap = ListedColormap(sns.color_palette().as_hex())

    # Treatments
    ax = plt.subplot(gs[:2, 1:3])
    ax = sc.pl.embedding(adata, basis=emb,
                         color=treat_var, ax=ax,
                         show=False)
    ax.scatter(adata.obsm[emb][terminal_states, 0],
               adata.obsm[emb][terminal_states, 1], s=100, c='black', marker='v')
    ax.scatter(adata.obsm[emb][start_cell, 0],
               adata.obsm[emb][start_cell, 1], s=100, c='black', marker='*')

    black_star = mlines.Line2D([], [], color='black', marker='*', linestyle='None',
                               markersize=10, label='Early Cell')
    black_triangle = mlines.Line2D([], [], color='black', marker='v', linestyle='None',
                                   markersize=10, label='Terminal State')

    handles, labels = ax.get_legend_handles_labels()
    handles.append(black_star)
    handles.append(black_triangle)
    ax.legend(handles=handles, bbox_to_anchor=(0, 0.5), loc='center right', frameon=False)

    ax.set_axis_off()
    ax.set_title("Treatment Conditions", fontsize=14)

    # Branch Labels
    for i, ts in enumerate(terminal_states):
        branch_label = f"branch_prob_{ts}"
        if cutoff is not None:
            color_mask = adata.obs[branch_label] > cutoff
        else:
            color_mask = adata.obs['branch'] == i
        adata.obs['color'] = adata.obs[treat_var]
        adata.obs.loc[~color_mask, 'color'] = np.nan
        row = int(np.floor(i / n_cols))
        ax = plt.subplot(gs[row + 2, np.remainder(i, n_cols)])
        ax = sc.pl.embedding(adata, basis=emb,
                             color='color', ax=ax, cmap=cmap,
                             show=False)
        ax.set_axis_off()
        ax.get_legend().remove()

        # add pie chart
        if dist is not None:
            treat_labels = list(dist.columns)
            x_loc = adata.obsm[emb][ts, 0]
            y_loc = adata.obsm[emb][ts, 1]
            loc = ax.transLimits.transform((x_loc, y_loc))
            ins = ax.inset_axes([loc[0] - 0.1, loc[1] - 0.1, 0.3, 0.3])
            sizes = dist.iloc[i, :].tolist()
            explode_sizes = np.zeros(len(treat_labels))
            if pie_explode:
                max_size = np.argmax(sizes)
                explode_sizes[max_size] = 0.1
            autopct = None
            if pie_pct:
                autopct = '%1.1f%%'
            pl = None
            if pie_labels:
                pl = treat_labels
            ins.pie(sizes, explode=explode_sizes, labels=pl, autopct=autopct)
            ins.axis('equal')

        ax.set_title(f"Branch {i}", fontsize=10)

    if show:
        plt.show()
    else:
        return fig


def plot_branch_distribution(adata,
                             dist,
                             emb='X_tsne',
                             treat_var='Metadata_Treatment',
                             show=True):
    """
    Plot Treatment distribution for every branch as precalculated with
    morphlia.external.Palantir.branch_dist.

    :param adata:
    :param dist:
    :param emb:
    :param treat_var:
    :param count:
    :param show:
    :return:
    """

    assert treat_var in adata.obs.columns, f"treat_var not in .obs: {treat_var}"

    assert emb in adata.obsm, f'emb not found in .obsm: {emb}'

    annotations = ['pseudotime', 'entropy', 'branch']
    assert all(ann in adata.obs.columns for ann in annotations), "AnnData object is not annotated. " \
                                                                 "Use Palantir.annotate_data()."

    # create figure
    plt_df = dist.stack()
    plt_df.rename('val', inplace=True)
    plt_df.index.set_names(['Branch', 'Treatment'], inplace=True)
    plt_df = plt_df.reset_index()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=plt_df, x='Branch', y='val', hue='Treatment')

    ax.set_ylabel('')
    title_style = 'Distribution'
    if dist.name == 'count':
        title_style = 'Count'
    ax.set_title(f"Branch {title_style}")
    ax.legend(frameon=False, bbox_to_anchor=(1, 0.5), loc='center left')

    if show:
        plt.show()
    else:
        return fig, ax


def plot_trends(trends, cmap='Set2', show=True):
    """
    Plot precomputed feature trends.
    Use morphelia.ext.Palantir.compute_trends beforehand.

    :param trends:
    :param cmap:
    :param show:
    :return:
    """
    branches = list(trends.keys())
    feats = list(trends[branches[0]].keys())
    feats.remove('pseudotime')
    n_feats = len(feats)

    sns.set_theme(style="white")
    height = 3 * n_feats
    fig, axs = plt.subplots(n_feats, squeeze=False, figsize=(7, height))
    cmap = plt.get_cmap(cmap)

    # plot every feature on a separate axis
    for ix, feat in enumerate(feats):
        for branch_i, branch in enumerate(branches):
            x = trends[branch]['pseudotime']
            y = trends[branch][feat]['trends']
            ci = trends[branch][feat]['ci']

            # plot
            axs[ix, 0].plot(x, y, c=cmap(branch_i), label=branch)
            axs[ix, 0].fill_between(x, ci[:, 0], ci[:, 1],
                                    color=cmap(branch_i), alpha=0.1)

        axs[ix, 0].set_title(feat, fontsize=12)
        axs[ix, 0].set_xticks([0, 1])
        axs[ix, 0].set_xlabel('Pseudotime')
        axs[ix, 0].spines['top'].set_visible(False)
        axs[ix, 0].spines['right'].set_visible(False)
        if ix == 0:
            axs[ix, 0].legend(title='Branch', frameon=False)

    if show:
        plt.show()

    else:
        return fig, axs
