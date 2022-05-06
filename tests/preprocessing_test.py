import pytest
import numpy as np
import morphelia as mp


@pytest.fixture
def subsample(test_data):
    len_before = len(test_data)
    new_data = mp.pp.subsample(test_data, perc=0.15, by="Metadata_Well")
    len_after = len(new_data)

    assert (0.1 * len_before) < len_after < (0.2 * len_before)
    return new_data


@pytest.fixture
def dropped_nan(subsample):
    new_data = mp.pp.drop_nan(subsample)
    assert not np.isnan(new_data.X).any()
    return new_data


@pytest.fixture
def dropped_duplicates(dropped_nan):
    new_data = mp.pp.drop_duplicates(dropped_nan)

    _, index = np.unique(new_data.X, axis=1, return_index=True)
    mask = np.ones(new_data.X.shape[1])
    mask[index] = 0
    assert mask.sum() == 0

    return new_data


@pytest.fixture
def dropped_invariant(dropped_duplicates):
    new_data = mp.pp.drop_invariant(dropped_duplicates)

    comp = new_data.X[0, :]
    mask = np.all(new_data.X == comp[None, :], axis=0)
    assert mask.sum() == 0

    return new_data


def test_final_data(dropped_invariant):
    assert True
