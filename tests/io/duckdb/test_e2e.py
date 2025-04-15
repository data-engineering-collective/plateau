from collections.abc import Generator
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest
from minimalkv import get_store_from_url
from minimalkv._key_value_store import KeyValueStore

from plateau.io.duckdb.dataframe import (
    read_table_as_ddb,
    store_dataset_from_ddb,
)
from plateau.io.eager import store_dataframes_as_dataset


@pytest.fixture
def store_url(tmpdir) -> Generator[KeyValueStore, Any, None]:
    yield get_store_from_url(f"hfs://{tmpdir}")


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


def test_read_write(df: pd.DataFrame, store_url):
    table = duckdb.execute("CREATE TABLE my_df AS SELECT * FROM df").table("my_df")

    store_dataset_from_ddb(
        store_url, "partitioned_dataset", [table], partition_on=["B", "E"]
    )

    con2 = read_table_as_ddb(
        "partitioned_dataset",
        store_url,
        as_table="my_df",
    )

    round_trip_df = con2.table("my_df").to_df()
    round_trip_df = round_trip_df[df.columns]  # align column order

    assert round_trip_df.compare(df).empty


def test_filter(df: pd.DataFrame, store_url):
    store_dataframes_as_dataset(store=store_url, dataset_uuid="dataset", dfs=[df])

    con = read_table_as_ddb(
        "dataset", store_url, predicates=[[("E", "==", "train")]], as_table="my_df"
    )
    round_trip_df = con.table("my_df").to_df()
    round_trip_df = round_trip_df[df.columns]  # align column order

    expected = df.iloc[[1, 3]].reset_index(drop=True)
    assert round_trip_df.compare(expected).empty
