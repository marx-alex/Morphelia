# import internal libraries
from collections import defaultdict

# import external libraries
import numpy as np
import pandas as pd
import anndata as ad


def aggregate(md, by=("BatchNumber", "PlateNumber", "Metadata_Well")):
    """Aggregate multidimensional morphological data by populations.

    Args:
        md (anndata.AnnData): Annotated data object.
        by (list of str): Variables to use for aggregation.

    Returns:
        anndata.AnnData
    """
    # check that variables in by are in anndata
    if not all(var in md.obs.columns for var in by):
        raise KeyError(f"Variables defined in 'by' are not in annotations: {by}")

    # store aggregated data
    X_agg = []
    obs_agg = defaultdict(list)

    # iterate over md with grouping variables
    for groups, sub_df in md.obs.groupby(list(by)):
        # cache annotations
        for key, val in sub_df.iloc[0, :].to_dict().items():
            obs_agg[key].append(val)

        # cache indices of group
        group_ix = sub_df.index

        # aggregate group
        agg = np.mean(md[group_ix, :].X, axis=0)
        # concatenate object number to agg
        agg = np.append(agg, [len(sub_df)]).reshape(1, -1)
        # concatenate aggregated groups
        X_agg.append(agg)

    # make anndata object from aggregated data
    X_agg = np.concatenate(X_agg, axis=0)
    obs_agg = pd.DataFrame(obs_agg)
    new_var = pd.DataFrame(index=['CellNumber'])
    var = md.var.append(new_var)

    return ad.AnnData(X=X_agg, obs=obs_agg, var=var)


def subsample(md, perc=0.1, by=("BatchNumber", "PlateNumber", "Metadata_Well")):
    """Gives a subsample of the data by selecting objects from given groups.

    Args:
        md (anndata.AnnData): Annotated data object.
        perc (float): Percentage of objects to store in subsample.
        by (list of str): Variables to use for aggregation.

    Returns:
        anndata.AnnData
    """
    # check that variables in by are in anndata
    if not all(var in md.obs.columns for var in by):
        raise KeyError(f"Variables defined in 'by' are not in annotations: {by}")
    if perc < 0 or perc > 1:
        raise ValueError(f"Use a float between 0 and 1 for perc: {perc}")

    # store subsample data
    X_ss = []
    obs_ss = defaultdict(list)

    # iterate over md with grouping variables
    for groups, sub_df in md.obs.groupby(list(by)):

        n = round(perc * len(sub_df))
        # cache indices subsample
        group_ix = sub_df.sample(n=n).index

        if len(group_ix) > 0:
            # cache annotations
            for key, val in sub_df.loc[group_ix, :].to_dict('list').items():
                obs_ss[key].extend(val)

            # subsample group
            agg = md[group_ix, :].X

            # concatenate aggregated groups
            X_ss.append(agg)

    # make anndata object from aggregated data
    X_ss = np.concatenate(X_ss, axis=0)
    obs_ss = pd.DataFrame(obs_ss)

    return ad.AnnData(X=X_ss, obs=obs_ss, var=md.var)
