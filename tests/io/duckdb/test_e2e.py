from tempfile import TemporaryDirectory

import duckdb
import numpy as np
import pandas as pd
import pytest
from minimalkv import get_store_from_url

from plateau.io.duckdb.dataframe import (
    read_table_as_ddb as fast_read_table_as_ddb,
    store_dataset_from_ddb as fast_store_dataframes_as_dataset,
)


@pytest.fixture
def store_url():
    dataset_dir = TemporaryDirectory()
    store_url = f"hfs://{dataset_dir.name}"

    yield get_store_from_url(store_url)

    dataset_dir.cleanup()


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": 1.0,
            "B": [
                pd.Timestamp("20130102"),
                pd.Timestamp("20130102"),
                pd.Timestamp("20130103"),
                pd.Timestamp("20130103"),
            ],
            "C": pd.Series(1, index=list(range(4)), dtype="float32"),
            "D": np.array([3] * 4, dtype="int32"),
            "E": pd.Categorical(["test", "train", "test", "train"]),
            "F": "foo",
        }
    )


def test_example_fast_read(df: pd.DataFrame, store_url):
    table = duckdb.execute("CREATE TABLE my_df AS SELECT * FROM df").table("my_df")

    fast_store_dataframes_as_dataset(
        store_url, "partitioned_dataset", [table], partition_on=["B", "E"]
    )

    con2 = fast_read_table_as_ddb(
        "partitioned_dataset",
        store_url,
        table="my_df",
        categoricals="E",
    )

    round_trip_df = con2.table("my_df").to_df()
    round_trip_df = round_trip_df[df.columns]  # align column order

    assert round_trip_df.compare(df).empty
