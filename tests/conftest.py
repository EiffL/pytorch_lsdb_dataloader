import lsdb
import pytest


@pytest.fixture(scope="session")
def catalog_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("cat") / "a"
    lsdb.generate_catalog(2000, 10, seed=1).write_catalog(path, catalog_name="a", overwrite=True)
    return path


@pytest.fixture(scope="session")
def catalog(catalog_path):
    return lsdb.open_catalog(catalog_path)


@pytest.fixture(scope="session")
def crossmatch(catalog):
    # Self-match: every row matches itself, and row counts per partition are not known up front.
    return catalog.crossmatch(catalog, radius_arcsec=1, n_neighbors=1, suffix_method="all_columns")


@pytest.fixture(scope="session")
def frame(catalog):
    return catalog.compute(progress_bar=False)


@pytest.fixture(scope="session")
def expected_index(frame):
    return sorted(frame.index.to_list())
