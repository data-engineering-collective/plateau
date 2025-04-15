Pyarrow/Table Backend
==========================

If you don't want to utilize DuckDB but e.g. polars instead,
you can use the pyarrow backend (that duckdb uses under the hood) and convert
the pyarrow table to a polars dataframe (zero-copy).

.. code-block:: python

    import polars as pl
    import pyarrow as pa
    from minimalkv import get_store_from_url

    from plateau.io.duckdb import read_table_as_arrow
    from plateau.io.eager import store_dataframes_as_dataset

    store = get_store_from_url("hfs:///tmp/store")

    data = {
        "A": [1, 2, 3],
        "B": ["foo", "bar", "baz"],
        "C": [0.1, 0.2, 0.3],
    }
    table = pa.table(data)

    store_dataframes_as_dataset(
        store=store,
        dataset_uuid="my_dataset",
        dfs=[table], # supports pyarrow tables
        partition_on=["A", "B"],
    )

    result = read_table_as_arrow("my_dataset", store)

    df = pl.from_arrow(result)
    print(df)
