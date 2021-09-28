import numpy as np
from tqdm import tqdm


def drop_low_variance(
        adata,
        freq_thresh=0.05,
        unique_thresh=0.01,
        drop=True,
        verbose=False
):
    """Drop features that have low variance and therefore low information content.
    Low variance is assumed if single values appear more than one time in a feature vector.
    The rules is as following:
    1. A feature vector is of low variance if second_max_count / max_count < freq_thresh
    2. A feature vector is of low variance if num_unique_values / n_samples < unique_thresh
    with:
        second_max_count: counts of second frequent value
        max_count: counts of most frequent value
        freq_thresh: threshold for rule 1.
        num_unique_values: number of unique values in a feature vector
        n_samples: number of total samples
        unique_thresh: threshold for rule 2.

    This idea is modified from caret::nearZeroVar():
    https://www.rdocumentation.org/packages/caret/versions/6.0-88/topics/nearZeroVar

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        freq_thresh (float): Threshold for frequency.
        unique_thresh (float): Threshold for uniqueness.
        drop (bool): Drop features with low variance directly.
        verbose (bool)
    """
    # check variables
    assert 0 <= freq_thresh <= 1, f"freq_thresh must be between 0 and 1, " \
                                  f"instead got {freq_thresh}"
    assert 0 <= unique_thresh <= 1, f"unique_thresh must be between 0 and 1, " \
                                    f"instead got {unique_thresh}"

    # store dropped features
    drop_feats = []
    # get number of samples
    n_samples = adata.shape[0]

    # test first and second rule
    # iterate over features
    for feat in tqdm(adata.var_names, desc="Iterating of features"):
        # get unique features and their counts
        unique, counts = np.unique(adata[:, feat].X, return_counts=True)
        counts = np.sort(counts)

        if len(counts) > 1:
            # fist rule
            max_count = counts[-1]
            second_max_count = counts[-2]

            freq_ratio = second_max_count / max_count

            # apply freq_thresh
            if freq_ratio < freq_thresh:
                drop_feats.append(feat)

            # second rule
            unique_ratio = len(unique) / n_samples

            if unique_ratio < unique_thresh:
                drop_feats.append(feat)

    drop_feats = list(set(drop_feats))
    mask = [False if var in drop_feats else True for var in adata.var_names]

    if verbose:
        print(f"Drop {len(drop_feats)} features with low variance: {drop_feats}")

    if drop:
        adata = adata[:, mask]
    else:
        adata.var["high_var"] = mask

    return adata





