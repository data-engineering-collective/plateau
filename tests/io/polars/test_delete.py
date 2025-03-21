import polars as pl
import pytest

from plateau.io.polars import delete_dataset, store_dataframes_as_dataset


def test_delete_dataset(store):
    df = pl.DataFrame({"P": [1, 2], "L": [1, 2], "TARGET": [1, 2]})
    store_dataframes_as_dataset(store=store, dataset_uuid="dataset_uuid", dfs=[df])

    delete_dataset(store=store, dataset_uuid="dataset_uuid")

    with pytest.raises(KeyError):
        store.get("dataset_uuid/table/auto_dataset_uuid.parquet")
