import pandas as pd
import pandas.testing as pdt
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from plateau.io.duckdb.helper import (
    align_categories,
    cast_categoricals_to_dictionary,
    empty_table_from_schema,
)


@pytest.fixture
def sample_schema():
    return pa.schema(
        [
            ("a", pa.int32()),
            ("b", pa.string()),
            ("c", pa.float64()),
        ]
    )


def test_empty_table_from_schema_full_schema(sample_schema):
    table = empty_table_from_schema(sample_schema)
    assert table.num_rows == 0
    assert table.schema.equals(sample_schema)


def test_empty_table_from_schema_subset(sample_schema):
    table = empty_table_from_schema(sample_schema, columns=["b", "c"])
    assert table.num_rows == 0
    assert table.column_names == ["b", "c"]


def test_cast_categoricals_to_dictionary():
    data = {"cat": pa.array(["a", "b", "a", None]), "num": pa.array([1, 2, 3, 4])}
    table = pa.Table.from_pydict(data)
    assert not pa.types.is_dictionary(table["cat"].type)
    new_table = cast_categoricals_to_dictionary(table, ["cat"])
    assert pa.types.is_dictionary(new_table["cat"].type)
    decoded = pc.dictionary_decode(new_table["cat"]).to_pylist()
    assert decoded == ["a", "b", "a", None]


def test_align_categories():
    # Create three PyArrow Tables from DataFrames with categorical columns
    table1 = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "col_A": pd.Categorical(["A1", "A3", "A3"]),
                "col_B": pd.Categorical(["B1", "B3", "B3"]),
            }
        ),
        preserve_index=False,
    )
    table2 = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "col_A": pd.Categorical(["A2", "A3", "A4"]),
                "col_B": pd.Categorical(["B2", "B3", "B4"]),
            }
        ),
        preserve_index=False,
    )
    table3 = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "col_A": pd.Categorical(["A4", "A5", "A1"]),
                "col_B": pd.Categorical(["B4", "B5", "B1"]),
            }
        ),
        preserve_index=False,
    )

    in_tables = [table1, table2, table3]
    out_tables = align_categories(in_tables, categoricals=["col_A", "col_B"])

    out_df1 = out_tables[0].to_pandas()
    out_df2 = out_tables[1].to_pandas()
    out_df3 = out_tables[2].to_pandas()

    for prefix in ["A", "B"]:
        col_name = f"col_{prefix}"
        expected_categories = [
            f"{prefix}1",
            f"{prefix}3",
            f"{prefix}2",
            f"{prefix}4",
            f"{prefix}5",
        ]

        expected_series1 = pd.Series(
            pd.Categorical(
                [f"{prefix}1", f"{prefix}3", f"{prefix}3"],
                categories=expected_categories,
            ),
            name=col_name,
        )
        expected_series2 = pd.Series(
            pd.Categorical(
                [f"{prefix}2", f"{prefix}3", f"{prefix}4"],
                categories=expected_categories,
            ),
            name=col_name,
        )
        expected_series3 = pd.Series(
            pd.Categorical(
                [f"{prefix}4", f"{prefix}5", f"{prefix}1"],
                categories=expected_categories,
            ),
            name=col_name,
        )

        pdt.assert_series_equal(out_df1[col_name], expected_series1)
        pdt.assert_series_equal(out_df2[col_name], expected_series2)
        pdt.assert_series_equal(out_df3[col_name], expected_series3)
