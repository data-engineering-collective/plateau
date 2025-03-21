import polars as pl
import pytest

from plateau.io.polars import (
    store_dataframes_as_dataset,
    update_dataset_from_dataframes,
)


def test_update_dataset_from_dataframes(store):
    # Create initial dataset
    df1 = pl.DataFrame({"P": [1, 2], "L": [1, 2], "TARGET": [1, 2]})
    store_dataframes_as_dataset(store=store, dataset_uuid="dataset_uuid", dfs=[df1])

    # Update with new data
    df2 = pl.DataFrame({"P": [3, 4], "L": [3, 4], "TARGET": [3, 4]})
    update_dataset_from_dataframes(store=store, dataset_uuid="dataset_uuid", dfs=[df2])

    # Read back and verify
    result = store_dataframes_as_dataset(
        store=store, dataset_uuid="dataset_uuid", dfs=[df1, df2]
    )
    assert len(result.partitions) == 2


def test_update_dataset_from_dataframes_invalid_schema(store):
    # Create initial dataset
    df1 = pl.DataFrame({"P": [1, 2], "L": [1, 2], "TARGET": [1, 2]})
    store_dataframes_as_dataset(store=store, dataset_uuid="dataset_uuid", dfs=[df1])

    # Try to update with incompatible schema
    df2 = pl.DataFrame({"P": [3, 4], "L": [3, 4], "DIFFERENT": [3, 4]})
    with pytest.raises(ValueError):
        update_dataset_from_dataframes(
            store=store, dataset_uuid="dataset_uuid", dfs=[df2]
        )


def test_update_dataset_from_dataframes_partition_on(store):
    # Create initial dataset with partitioning
    df1 = pl.DataFrame({"P": [1, 2], "L": [1, 2], "TARGET": [1, 2]})
    store_dataframes_as_dataset(
        store=store,
        dataset_uuid="dataset_uuid",
        dfs=[df1],
        partition_on=["P"],
    )

    # Update with new data
    df2 = pl.DataFrame({"P": [1, 2], "L": [3, 4], "TARGET": [3, 4]})
    update_dataset_from_dataframes(
        store=store,
        dataset_uuid="dataset_uuid",
        dfs=[df2],
        partition_on=["P"],
    )

    # Read back and verify
    result = store_dataframes_as_dataset(
        store=store,
        dataset_uuid="dataset_uuid",
        dfs=[df1, df2],
        partition_on=["P"],
    )
    assert (
        len(result.partitions) == 2
    )  # Should have 2 partitions due to partitioning on P
