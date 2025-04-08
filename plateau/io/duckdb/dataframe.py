import logging
from typing import Any

import duckdb
import pyarrow as pa
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from minimalkv import KeyValueStore

from plateau.core.factory import DatasetFactory, _ensure_factory
from plateau.io.duckdb.helper import (
    align_categories,
    cast_categoricals_to_dictionary,
    empty_table_from_schema,
)
from plateau.io.eager import (
    read_dataset_as_metapartitions,
    store_dataframes_as_dataset,
)

LOGGER = logging.getLogger(__name__)


def read_table_as_ddb(
    uuid: str,
    store: KeyValueStore,
    table: str,
    predicates: list[list[tuple[str, str, Any]]] | None = None,
    **kwargs,  # support for everything else
) -> duckdb.DuckDBPyConnection:
    if "categoricals" in kwargs:
        LOGGER.warning(
            "'categoricals' argument will be ignored as arrow dictionary is"
            " mapped to VARCHAR by default in DuckDB."
            " You can manually cast the column to ENUM type in DuckDB or use"
            " the 'read_table_as_arrow' method directly."
        )

    table_obj = read_table_as_arrow(uuid, store=store, predicates=predicates, **kwargs)
    con = duckdb.connect()
    con.register(table, table_obj)
    return con


def read_table_as_arrow(
    dataset_uuid: str | None = None,
    store=None,
    columns: dict[str, list[str]] | None = None,
    predicate_pushdown_to_io: bool = True,
    categoricals: list[str] | None = None,
    dates_as_object: bool = True,
    predicates: list[list[tuple[str, str, Any]]] | None = None,
    factory: DatasetFactory | None = None,
) -> pa.Table:
    ds_factory = _ensure_factory(
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
    )
    partitions = read_dataset_as_arrow_tables(
        columns=columns,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        categoricals=categoricals,
        dates_as_object=dates_as_object,
        predicates=predicates,
        factory=ds_factory,
    )

    empty_table = empty_table_from_schema(
        schema=ds_factory.schema.internal(),
        columns=columns,
    )

    if categoricals:
        empty_table = cast_categoricals_to_dictionary(empty_table, categoricals)

    tables = list(partitions) + [empty_table]

    for table in tables:
        print(table)
        print()

    if categoricals:
        tables = align_categories(tables, categoricals)

    table = pa.concat_tables(tables, promote_options="permissive")

    # Ensure column order matches that of the empty table.
    if empty_table.num_columns > 0 and empty_table.column_names != table.column_names:
        table = table.select(empty_table.column_names)

    return table


def read_dataset_as_arrow_tables(
    dataset_uuid: str | None = None,
    store=None,
    columns: dict[str, list[str]] | None = None,
    predicate_pushdown_to_io: bool = True,
    categoricals: list[str] | None = None,
    dates_as_object: bool = True,
    predicates: list[list[tuple[str, str, Any]]] | None = None,
    factory: DatasetFactory | None = None,
    dispatch_by: list[str] | None = None,
):
    ds_factory = _ensure_factory(
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
    )

    mps = read_dataset_as_metapartitions(
        columns=columns,
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        categoricals=categoricals,
        dates_as_object=dates_as_object,
        predicates=predicates,
        factory=ds_factory,
        dispatch_by=dispatch_by,
        arrow_mode=True,
    )
    return [mp.data for mp in mps]


def store_dataset_from_ddb(
    store: KeyValueStore,
    dataset_uuid: str,
    duckdb: list[DuckDBPyConnection | DuckDBPyRelation],
    partition_on: list[str] | None = None,
    **kwargs,  # support for everything else
):
    arrow_tables = [item.arrow() for item in duckdb]

    store_dataframes_as_dataset(
        store=store,
        dataset_uuid=dataset_uuid,
        dfs=arrow_tables,  # TODO: naming of variable?
        partition_on=partition_on,
        **kwargs,
    )


# TODO: update_dataset_from_partitions
