import polars as pl
import pytest

from plateau.io.polars import garbage_collect_dataset, store_dataframes_as_dataset


def test_garbage_collect_dataset(store):
    df = pl.DataFrame({"P": [1, 2], "L": [1, 2], "TARGET": [1, 2]})
    store_dataframes_as_dataset(store=store, dataset_uuid="dataset_uuid", dfs=[df])

    # Add some garbage files
    store.put("dataset_uuid/garbage.parquet", b"garbage")
    store.put("dataset_uuid/table/garbage.parquet", b"garbage")

    garbage_collect_dataset(store=store, dataset_uuid="dataset_uuid")

    with pytest.raises(KeyError):
        store.get("dataset_uuid/garbage.parquet")
    with pytest.raises(KeyError):
        store.get("dataset_uuid/table/garbage.parquet")
    # Original data should still be there
    assert store.get("dataset_uuid/table/auto_dataset_uuid.parquet") is not None
