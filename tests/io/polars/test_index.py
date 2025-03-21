import polars as pl
import pytest

from plateau.io.polars import build_dataset_indices, store_dataframes_as_dataset


def test_build_dataset_indices(store):
    df = pl.DataFrame({"P": [1, 2], "L": [1, 2], "TARGET": [1, 2]})
    store_dataframes_as_dataset(store=store, dataset_uuid="dataset_uuid", dfs=[df])

    build_dataset_indices(store=store, dataset_uuid="dataset_uuid", columns=["P"])

    # Check that index files were created
    assert store.get("dataset_uuid/indices/P.parquet") is not None


def test_build_dataset_indices_invalid_column(store):
    df = pl.DataFrame({"P": [1, 2], "L": [1, 2], "TARGET": [1, 2]})
    store_dataframes_as_dataset(store=store, dataset_uuid="dataset_uuid", dfs=[df])

    with pytest.raises(ValueError):
        build_dataset_indices(
            store=store, dataset_uuid="dataset_uuid", columns=["INVALID"]
        )
