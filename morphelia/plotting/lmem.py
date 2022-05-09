import warnings
import logging
import os
from typing import Optional, Union, Tuple, List

import statsmodels.formula.api as smf
import seaborn as sns
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt

from morphelia.tools.utils import get_subsample
from morphelia.tools import get_cmap

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def plot_lmem(
    adata: ad.AnnData,
    var: str,
    treat_var: str = "Metadata_Treatment",
    ctrl_id: str = "ctrl",
    fixed_var: str = "Metadata_Concentration",
    rand_var: Union[str, List[str]] = "BatchNumber",
    palette: str = "Set3",
    subsample: bool = False,
    sample_size: int = 10000,
    seed: int = 0,
    show: bool = False,
    save: Optional[str] = None,
    **kwargs,
) -> Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]:
    """Plot result from a linear mixed model.

    Visually evaluate linear mixed models for specific features.
    A model is fitted with `var` as dependent variable, `fixed_var` as dependent variable and
    `rand_var` as random variable.

    Parameters
    ----------
    adata : anndata.AnnData
        Multidimensional morphological data
    var : str
        Variable in `.var_names`
    treat_var : str
        Treatment variable in `.obs`
    ctrl_id : str
        Name of control condition stored in `treat_var`
    fixed_var : str
        Name of variable with fixed effect
    rand_var : str or list of str
        Name or list of variables with random effects
    palette : str
        Matplotlib or morphelia palette
    subsample : bool
        Use method on subsample of the data
    sample_size : int
        Size of subsample
    seed : int
        Seed for reproducibility
    show : bool
        Show and return axis object
    save : str, optional
        Path where to save as `lmem.png`
    **kwargs
        Keyword arguments passed to `seaborn.boxplot`

    Returns
    -------
    matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure if show is False and Axes

    Raises
    ------
    AssertionError
        If `treat_var`, `fixed_var` or `rand_var` are not in `.obs`
    OSError
        If figure can not be saved at specified location
    """
    # check variables
    assert treat_var in adata.obs.columns, f"treat_var not in .obs: {treat_var}"
    assert fixed_var in adata.obs.columns, f"fixed_var not in .obs: {fixed_var}"
    if isinstance(rand_var, str):
        rand_var = [rand_var]
    assert all(
        rv in adata.obs.columns for rv in rand_var
    ), f"rand_var not in .obs: {rand_var}"
    # control group should be in treat_var
    unique_treats = list(adata.obs[treat_var].unique())
    if ctrl_id not in unique_treats:
        warnings.warn(
            f"Control group not in {treat_var}: {unique_treats}. Control will not be included in dose steps."
        )
    else:
        unique_treats.remove(ctrl_id)

    # evtl subsample data
    if subsample:
        adata = get_subsample(adata, sample_size=sample_size, seed=seed)
    else:
        adata = adata.copy()

    # get effects
    unique_x = adata.obs[fixed_var].unique()
    plt_x = adata.obs[treat_var].astype(str) + "_" + adata.obs[fixed_var].astype(str)
    sorted_ix = np.argsort(unique_x)
    unique_x = unique_x[sorted_ix]
    plt_x = plt_x.unique()[sorted_ix]
    plt_x_mapping = dict(enumerate(plt_x))
    x_mapping = {v: i for i, v in enumerate(unique_x)}
    x = adata.obs[fixed_var].map(x_mapping)
    logger.info(f"Independent variables: {x_mapping}")
    z = adata.obs[rand_var]
    if isinstance(z, pd.DataFrame):
        # merge columns
        z = z.apply(lambda col: "_".join(col.astype(str)), axis=1)
    z = z.astype("category")
    z = np.asarray(z.cat.codes)
    y = adata[:, var].X.copy().flatten()

    df = pd.DataFrame({"y": y, "x": x, "z": z, "treat": adata.obs[treat_var]})

    # create figure
    fig, axs = plt.subplots(len(unique_treats), squeeze=False)
    # get color
    if palette in plt.colormaps():
        palette = plt.get_cmap(palette).copy()
    else:
        palette = get_cmap(palette)

    for ix, treat in enumerate(unique_treats):
        logger.info(f"Fit LME on {treat} [{treat_var}]")
        df_treat = df[(df["treat"] == treat) | (df["treat"] == ctrl_id)]

        # fit model
        lmm = smf.mixedlm("y ~ x", df_treat, groups=df["z"])
        f_lmm = lmm.fit()

        # plot this treatment
        sns.boxplot(
            x="x",
            y="y",
            hue="z",
            data=df_treat,
            palette=palette.colors,
            ax=axs[ix, 0],
            **kwargs,
        )

        intercept, slope = f_lmm.params[0], f_lmm.params[1]
        re = f_lmm.random_effects

        def linear_fun(x, a, b):
            return a * x + b

        for group_ix, group in enumerate(df_treat["z"].unique()):
            rev = re[group]["Group"]
            y_hat = linear_fun(df_treat["x"], slope, intercept + rev)
            plt.plot(df_treat["x"], y_hat, color="k")

        axs[ix, 0].set_ylabel(var)
        axs[ix, 0].set_xlabel(fixed_var)
        axs[ix, 0].legend(title="Group")
        axs[ix, 0].set_xticklabels(plt_x_mapping.values())
        axs[0, ix].set_title(treat)

    if save is not None:
        if not os.path.exists(save):
            raise OSError(f"Path does not exist: {save}")
        fig.savefig(os.path.join(save, "lmem.png"), dpi=fig.dpi)

    if show:
        plt.show()
        return axs

    return fig, axs
