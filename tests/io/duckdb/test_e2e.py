from tempfile import TemporaryDirectory

import duckdb
import numpy as np
import pandas as pd
import pytest
from minimalkv import get_store_from_url

from plateau.io.duckdb.dataframe import read_table_as_ddb, store_dataset_from_ddb
from plateau.io.eager import read_table, store_dataframes_as_dataset


@pytest.fixture
def store_url():
    dataset_dir = TemporaryDirectory()
    store_url = f"hfs://{dataset_dir.name}"

    yield get_store_from_url(store_url)

    dataset_dir.cleanup()


def test_example(store_url):
    con = duckdb.connect()
    con.execute("CREATE TABLE my_df (a INTEGER, b VARCHAR)")
    con.execute("INSERT INTO my_df VALUES (1, 'a'), (2, 'b')")

    store_dataset_from_ddb(
        store=store_url,
        dataset_uuid="a_new_dataset",
        duckdb=[
            con.table("my_df")
            # can also use con.execute("SELECT * FROM my_df")
        ],
    )

    con2 = read_table_as_ddb("a_new_dataset", store_url, table="my_df")
    assert (
        con2.table("my_df").fetchall() == con.table("my_df").fetchall()
    )  # compare underlying data


def test_example_normal_read(store_url):
    df = pd.DataFrame(
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

    store_dataframes_as_dataset(
        store_url, "partitioned_dataset", [df], partition_on="B"
    )
    df = read_table("partitioned_dataset", store_url)


def test_example_fast_read(store_url):
    # con = duckdb.connect()
    # con.execute("CREATE TABLE my_df (a INTEGER, b VARCHAR)")
    # con.execute("INSERT INTO my_df VALUES (1, 'a'), (2, 'b')")
    df = pd.DataFrame(
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

    store_dataframes_as_dataset(
        store_url, "partitioned_dataset", [df], partition_on="B"
    )

    # con2 = fast_read_table_as_ddb("partitioned_dataset", store_url, table="my_df")
