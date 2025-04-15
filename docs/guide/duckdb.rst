DuckDB Integration Example
==========================

The `plateau.io.duckdb` module allows seamless integration with DuckDB for reading and writing partitioned datasets.

Here's a minimal example of writing and reading a partitioned dataset using DuckDB:

.. code-block:: python

    import duckdb
    import numpy as np
    import pandas as pd
    from minimalkv import get_store_from_url

    from plateau.io.duckdb import (
        read_table_as_ddb,
        store_dataset_from_ddb,
    )

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

    # Create DuckDB table
    duckdb.execute("CREATE TABLE my_df AS SELECT * FROM df")

    store = get_store_from_url("hfs:///tmp/store")

    # Store dataset with partitioning
    store_dataset_from_ddb(store, "my_dataset", [duckdb.table("my_df")], partition_on=["B", "E"])

    # Read back the table using DuckDB
    con = read_table_as_ddb("my_dataset", store, as_table="my_df")
    result = con.table("my_df")

    print(result)
