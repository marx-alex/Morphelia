# import internal libraries
from collections import defaultdict

# import external libraries
import numpy as np
import pandas as pd
import anndata as ad


def subsample(adata, perc=0.1, by=("BatchNumber", "PlateNumber", "Metadata_Well"),
              seed=0):
    """Gives a subsample of the data by selecting objects from given groups.

    Args:
        adata (anndata.AnnData): Annotated data object.
        perc (float): Percentage of objects to store in subsample.
        by (list of str): Variables to use for aggregation.
        seed (int): Seed for initialization.

    Returns:
        anndata.AnnData
    """
    # check that variables in by are in anndata
    if not all(var in adata.obs.columns for var in by):
        raise KeyError(f"Variables defined in 'by' are not in annotations: {by}")
    if perc < 0 or perc > 1:
        raise ValueError(f"Use a float between 0 and 1 for perc: {perc}")

    # store subsample data
    X_ss = []
    obs_ss = defaultdict(list)

    # iterate over md with grouping variables
    for groups, sub_df in adata.obs.groupby(list(by)):

        n = round(perc * len(sub_df))
        # cache indices subsample
        group_ix = sub_df.sample(n=n, random_state=seed).index

        if len(group_ix) > 0:
            # cache annotations
            for key, val in sub_df.loc[group_ix, :].to_dict('list').items():
                obs_ss[key].extend(val)

            # subsample group
            agg = adata[group_ix, :].X

            # concatenate aggregated groups
            X_ss.append(agg)

    # make anndata object from subsample
    X_ss = np.concatenate(X_ss, axis=0)
    obs_ss = pd.DataFrame(obs_ss)

    return ad.AnnData(X=X_ss, obs=obs_ss, var=adata.var)