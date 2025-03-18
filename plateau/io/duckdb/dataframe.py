from typing import Any
import duckdb
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from minimalkv import KeyValueStore

from plateau.io.eager import (
    read_table,
    store_dataframes_as_dataset,
)


def read_table_as_ddb(
    uuid: str,
    store: KeyValueStore,
    table: str,
    predicates: list[list[tuple[str, str, Any]]] | None = None,
    **kwargs,  # support for everything else
) -> duckdb.DuckDBPyConnection:
    df = read_table(uuid, store=store, predicates=predicates, **kwargs)
    con = duckdb.connect()
    con.register(table, df)
    return con


def store_dataset_from_ddb(
    store: KeyValueStore,
    dataset_uuid: str,
    duckdb: list[DuckDBPyConnection | DuckDBPyRelation],
    partition_on: list[str] | None = None,
    **kwargs,  # support for everything else
):
    store_dataframes_as_dataset(
        store=store,
        dataset_uuid=dataset_uuid,
        dfs=[
            item.fetch_df() if isinstance(item, DuckDBPyConnection) else item.fetchdf()
            for item in duckdb  # TODO: use arrow
        ],
        partition_on=partition_on,
        **kwargs,
    )


# TODO: update_dataset_from_partitions
