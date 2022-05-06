import os
import pytest
import morphelia as mp

TEST_DATA = "../data/cp_output"


@pytest.fixture(scope="session", autouse=True)
def test_data():
    """
    This function is called once per test session to read the
    test data.
    """
    path = os.path.dirname(__file__)
    test_data_path = os.path.join(path, TEST_DATA)

    plate = mp.tl.LoadPlate(
        test_data_path, obj_sfx=".txt", obj_delimiter="\t", treat_file="Treatment"
    )

    plate.load()
    plate = plate.to_anndata()
    return plate
