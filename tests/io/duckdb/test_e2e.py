from tempfile import TemporaryDirectory

import duckdb
import pytest
from minimalkv import get_store_from_url

from plateau.io.duckdb.dataframe import read_table_as_ddb, store_dataset_from_ddb
from plateau.io.duckdb.dataframe_fast import (
    read_table_as_ddb as fast_read_table_as_ddb,
    store_dataset_from_ddb as fast_store_dataset_from_ddb,
)


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


def test_example_fast(store_url):
    con = duckdb.connect()
    con.execute("CREATE TABLE my_df (a INTEGER, b VARCHAR)")
    con.execute("INSERT INTO my_df VALUES (1, 'a'), (2, 'b')")

    fast_store_dataset_from_ddb(
        store=store_url,
        dataset_uuid="a_new_dataset",
        duckdb=[
            con.table("my_df")
            # can also use con.execute("SELECT * FROM my_df")
        ],
    )

    con2 = fast_read_table_as_ddb("a_new_dataset", store_url, table="my_df")
    assert (
        con2.table("my_df").fetchall() == con.table("my_df").fetchall()
    )  # compare underlying data
