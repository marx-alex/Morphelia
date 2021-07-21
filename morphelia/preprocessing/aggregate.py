# import internal libraries
from collections import defaultdict

# import external libraries
import numpy as np
import pandas as pd
import anndata as ad


def aggregate(adata, by=("BatchNumber", "PlateNumber", "Metadata_Well"),
              obs_ids='Metadata', min_cells=300, verbose=False):
    """Aggregate multidimensional morphological data by populations.

    Args:
        adata (anndata.AnnData): Annotated data object.
        by (list of str): Variables to use for aggregation.
        obs_ids (list of str): Identifiers for observations to keep.
        min_cells (int): Minimum number of cells per population.
            Population is deleted from data if below threshold.
        verbose (bool)

    Returns:
        anndata.AnnData
    """
    # check that variables in by are in anndata
    if isinstance(by, str):
        by = [by]
    else:
        by = list(by)
    if not all(var in adata.obs.columns for var in by):
        raise KeyError(f"Variables defined in 'by' are not in annotations: {by}")

    # delete observations not needed for aggregation
    if obs_ids is not None:
        if isinstance(obs_ids, str):
            obs_ids = [obs_ids]
        if isinstance(obs_ids, list):
            drop_obs = [obs for obs in adata.obs.columns if not any(identifier in obs for identifier in obs_ids)]
            for elem in by:
                if elem in drop_obs:
                    drop_obs.remove(elem)
            adata.obs.drop(drop_obs, axis=1, inplace=True)
        else:
            raise TypeError(f"obs_ids is expected to be string or list, instead got {type(obs_ids)}")

    # store aggregated data
    X_agg = []
    obs_agg = defaultdict(list)

    # iterate over md with grouping variables
    for groups, sub_df in adata.obs.groupby(list(by)):
        # cache annotations
        for key, val in sub_df.iloc[0, :].to_dict().items():
            obs_agg[key].append(val)
        # add object number to observations
        obs_agg['Metadata_CellNumber'].append(len(sub_df))

        # cache indices of group
        group_ix = sub_df.index

        # aggregate group
        agg = np.median(adata[group_ix, :].X, axis=0).reshape(1, -1)
        # concatenate aggregated groups
        X_agg.append(agg)

    # make anndata object from aggregated data
    X_agg = np.concatenate(X_agg, axis=0)
    obs_agg = pd.DataFrame(obs_agg)

    adata = ad.AnnData(X=X_agg, obs=obs_agg, var=adata.var)

    # quality control
    if min_cells is not None:
        if verbose:
            dropped_pops = adata[adata.obs['Metadata_CellNumber'] < min_cells, :].obs[by].values.tolist()
            print(f"Dropped populations: {dropped_pops}")
        adata = adata[adata.obs['Metadata_CellNumber'] >= min_cells, :]

    return adata
