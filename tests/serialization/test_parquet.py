import os
from datetime import date, datetime

import minimalkv
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pyarrow as pa
import pytest
from pyarrow.parquet import ParquetFile

from plateau.core._compat import pandas_infer_string
from plateau.serialization import ParquetSerializer
from plateau.serialization._parquet import (
    MAX_NB_RETRIES,
    ParquetReadError,
    _predicate_accepts,
    _reset_dictionary_columns,
)
from plateau.serialization._util import _check_contains_null


@pytest.fixture
def reference_store():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "..",
        "reference-data",
        "pyarrow-bugs",
    )
    return minimalkv.get_store_from_url(f"hfs://{path}")


@pytest.fixture(params=[1, 5, None])
def chunk_size(request, mocker):
    chunk_size = request.param
    if chunk_size == 1:
        # Test for a chunk size of one and mock the filter_df call. This way we can ensure that
        # the predicate for IO is properly evaluated and pushed down
        mocker.patch(
            "plateau.serialization._parquet.filter_df",
            new=lambda x, *args, **kwargs: x,
        )
    return chunk_size


@pytest.mark.parametrize("chunk_size", ["1", 1.0])
def test_parquet_serializer_raises_on_invalid_chunk_size_type(chunk_size):
    with pytest.raises(
        TypeError,
        match="Cannot initialize ParquetSerializer because chunk size is not integer type",
    ):
        ParquetSerializer(chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_size", [-5, 0])
def test_parquet_serializer_raises_when_chunk_size_smaller_one(chunk_size):
    with pytest.raises(
        ValueError, match="Cannot initialize ParquetSerializer because chunk size < 1"
    ):
        ParquetSerializer(chunk_size=chunk_size)


@pytest.mark.parametrize("use_categorical", [True, False])
def test_rowgroup_writing(store, use_categorical, chunk_size):
    df = pd.DataFrame({"string": ["abc", "affe", "banane", "buchstabe"] * 5})
    serialiser = ParquetSerializer(chunk_size=chunk_size)
    # Arrow 0.9.0 has a bug in writing categorical columns to more than a single
    # RowGroup: "ArrowIOError: Column 2 had 2 while previous column had 4".
    # We have special handling for that in pandas-serialiser that should be
    # removed once we switch to 0.10.0
    if use_categorical:
        df_write = df.astype({"string": "category"})
    else:
        df_write = df
    key = serialiser.store(store, "prefix", df_write)

    parquet_file = ParquetFile(store.open(key))
    if chunk_size is None:
        assert parquet_file.num_row_groups == 1
    else:
        assert parquet_file.num_row_groups == 20 / chunk_size


_INT_TYPES = ["int8", "int16", "int32", "int64"]

# uint64 will fail since numexpr cannot safe case from uint64 to int64
_UINT_TYPES = ["uint8", "uint16", "uint32"]

_FLOAT_TYPES = ["float32", "float64"]

_STR_TYPES = ["unicode", "bytes"]

_DATE_TYPES = ["date"]
_DATETIME_TYPES = ["datetime64"]


def _validate_predicate_pushdown(df, column, value, store, chunk_size):
    serialiser = ParquetSerializer(chunk_size=chunk_size)
    key = serialiser.store(store, "prefix", df)

    predicates = [[(column, "==", value)]]

    df_restored = serialiser.restore_dataframe(store, key, predicates=predicates)
    # date objects are converted to datetime in pyarrow
    df_restored["date"] = df_restored["date"].dt.date

    expected = df.iloc[[3]]
    # ARROW-5138 index isn't preserved when doing predicate pushdown
    pdt.assert_frame_equal(
        df_restored.reset_index(drop=True), expected.reset_index(drop=True)
    )


@pytest.mark.parametrize("column", _INT_TYPES)
@pytest.mark.parametrize(
    "input_values",
    [
        (3, None),
        (3.0, TypeError),
        ("3", TypeError),
        ("3.0", TypeError),
        (b"3", TypeError),
        (b"3.0", TypeError),
    ],
)
def test_predicate_evaluation_integer(
    store, dataframe_not_nested, column, input_values, chunk_size
):
    value, exception = input_values
    if exception:
        with pytest.raises(exception):
            _validate_predicate_pushdown(
                dataframe_not_nested, column, value, store, chunk_size
            )
    else:
        _validate_predicate_pushdown(
            dataframe_not_nested, column, value, store, chunk_size
        )


@pytest.mark.parametrize("column", _UINT_TYPES)
@pytest.mark.parametrize(
    "input_values",
    [
        (3, None),
        (3.0, TypeError),
        ("3", TypeError),
        ("3.0", TypeError),
        (b"3", TypeError),
        (b"3.0", TypeError),
    ],
)
def test_predicate_evaluation_unsigned_integer(
    store, dataframe_not_nested, column, input_values, chunk_size
):
    value, exception = input_values
    if exception:
        with pytest.raises(exception):
            _validate_predicate_pushdown(
                dataframe_not_nested, column, value, store, chunk_size
            )
    else:
        _validate_predicate_pushdown(
            dataframe_not_nested, column, value, store, chunk_size
        )


@pytest.mark.parametrize("column", _FLOAT_TYPES)
@pytest.mark.parametrize(
    "input_values",
    [
        (3, TypeError),
        (3.0, None),
        ("3", TypeError),
        ("3.0", TypeError),
        (b"3", TypeError),
        (b"3.0", TypeError),
    ],
)
def test_predicate_evaluation_float(
    store, dataframe_not_nested, column, input_values, chunk_size
):
    value, exception = input_values
    if exception:
        with pytest.raises(exception):
            _validate_predicate_pushdown(
                dataframe_not_nested, column, value, store, chunk_size
            )
    else:
        _validate_predicate_pushdown(
            dataframe_not_nested, column, value, store, chunk_size
        )


@pytest.mark.parametrize("column", _STR_TYPES)
@pytest.mark.parametrize(
    "input_values", [(3, TypeError), (3.0, TypeError), ("3", None), (b"3", None)]
)
def test_predicate_evaluation_string(
    store, dataframe_not_nested, column, input_values, chunk_size
):
    value, exception = input_values
    if exception:
        with pytest.raises(exception):
            _validate_predicate_pushdown(
                dataframe_not_nested, column, value, store, chunk_size
            )
    else:
        _validate_predicate_pushdown(
            dataframe_not_nested, column, value, store, chunk_size
        )


@pytest.mark.parametrize("column", _DATE_TYPES)
@pytest.mark.parametrize(
    "input_values",
    [
        # it's the fifth due to the day % 31 in the testdata
        (date(2018, 1, 5), None),
        ("2018-01-05", None),
        (b"2018-01-05", None),
        (datetime(2018, 1, 1, 1, 1), TypeError),
        (3, TypeError),
        (3.0, TypeError),
        ("3", ValueError),
        ("3.0", ValueError),
        (b"3", ValueError),
        (b"3.0", ValueError),
    ],
)
def test_predicate_evaluation_date(
    store, dataframe_not_nested, column, input_values, chunk_size
):
    value, exception = input_values
    if exception:
        with pytest.raises(exception):
            _validate_predicate_pushdown(
                dataframe_not_nested, column, value, store, chunk_size
            )
    else:
        _validate_predicate_pushdown(
            dataframe_not_nested, column, value, store, chunk_size
        )


@pytest.mark.parametrize("column", _DATETIME_TYPES)
@pytest.mark.parametrize(
    "input_values",
    [
        (datetime(2018, 1, 5), None),
        (np.datetime64(datetime(2018, 1, 5)), None),
        (pd.Timestamp(datetime(2018, 1, 5)), None),
        (np.datetime64(datetime(2018, 1, 5), "s"), None),
        (np.datetime64(datetime(2018, 1, 5), "ms"), None),
        (np.datetime64(datetime(2018, 1, 5), "us"), None),
        (np.datetime64(datetime(2018, 1, 5), "ns"), None),
        (date(2018, 1, 4), TypeError),
        ("2018-01-04", TypeError),
        (b"2018-01-04", TypeError),
        (1, TypeError),
        (1.0, TypeError),
    ],
)
def test_predicate_evaluation_datetime(
    store, dataframe_not_nested, column, input_values, chunk_size
):
    value, exception = input_values
    if exception:
        with pytest.raises(exception):
            _validate_predicate_pushdown(
                dataframe_not_nested, column, value, store, chunk_size
            )
    else:
        _validate_predicate_pushdown(
            dataframe_not_nested, column, value, store, chunk_size
        )


def test_ensure_binaries(binary_value):
    assert isinstance(binary_value, bytes)


def test_pushdown_binaries(store, dataframe_not_nested, binary_value, chunk_size):
    if _check_contains_null(binary_value):
        pytest.xfail("Null-terminated binary strings are not supported")
    serialiser = ParquetSerializer(chunk_size=chunk_size)
    key = serialiser.store(store, "prefix", dataframe_not_nested)

    predicates = [[("bytes", "==", binary_value)]]

    df_restored = serialiser.restore_dataframe(store, key, predicates=predicates)
    assert len(df_restored) == 1
    assert df_restored.iloc[0].bytes == binary_value


@pytest.mark.xfail(reason="Requires parquet-cpp 1.5.0.")
def test_pushdown_null_intermediate(store):
    binary = b"\x8f\xb6\xe5@\x90\xdc\x11\xe8\x00\xae\x02B\xac\x12\x01\x06"
    df = pd.DataFrame({"byte_with_null": [binary]})
    serialiser = ParquetSerializer(chunk_size=1)
    key = serialiser.store(store, "key", df)
    predicate = [[("byte_with_null", "==", binary)]]
    restored = serialiser.restore_dataframe(store, key, predicates=predicate)
    pdt.assert_frame_equal(restored, df)


@pytest.mark.parametrize("chunk_size", [None, 1])
def test_date_as_object(store, chunk_size):
    ser = ParquetSerializer(chunk_size=chunk_size)
    df = pd.DataFrame({"date": [date(2000, 1, 1), date(2000, 1, 2)]})
    key = ser.store(store, "key", df)
    restored_df = ser.restore_dataframe(
        store, key, categories=["date"], date_as_object=True
    )
    categories = pd.Series([date(2000, 1, 1), date(2000, 1, 2)])
    expected_df = pd.DataFrame({"date": pd.Categorical(categories)})
    # expected_df.date = expected_df.date.cat.rename_categories([date(2000, 1, 1)])
    pdt.assert_frame_equal(restored_df, expected_df)

    restored_df = ser.restore_dataframe(
        store, key, date_as_object=True, predicates=[[("date", "==", "2000-01-01")]]
    )
    expected_df = pd.DataFrame({"date": [date(2000, 1, 1)]})
    pdt.assert_frame_equal(restored_df, expected_df)


@pytest.mark.parametrize("chunk_size", [None, 1])
def test_predicate_not_in_columns(store, chunk_size):
    ser = ParquetSerializer(chunk_size=chunk_size)
    df = pd.DataFrame(
        {
            "date": [date(2000, 1, 1), date(2000, 1, 2), date(2000, 1, 2)],
            "col": [1, 2, 1],
        }
    )
    key = ser.store(store, "key", df)
    restored_df = ser.restore_dataframe(
        store, key, columns=[], predicates=[[("col", "==", 1)]]
    )
    if chunk_size:
        expected_df = pd.DataFrame(
            index=[0, 1],
            columns=pd.Index([], dtype=str if pandas_infer_string() else "O"),
        )
    else:
        expected_df = pd.DataFrame(
            index=[0, 2],
            columns=pd.Index([], dtype=str if pandas_infer_string() else "O"),
        )

    pdt.assert_frame_equal(restored_df, expected_df)


def test_read_empty_file_with_predicates(store):
    ser = ParquetSerializer()
    df = pd.DataFrame({"col": pd.Series([], dtype=str)})
    key = ser.store(store, "key", df)
    restored_df = ser.restore_dataframe(
        store, key, columns=["col"], predicates=[[("col", "==", "1")]]
    )
    pdt.assert_frame_equal(restored_df, df)


@pytest.mark.parametrize("predicate_pushdown_to_io", [True, False])
def test_int64_statistics_overflow(reference_store, predicate_pushdown_to_io):
    # Test case for ARROW-5166
    ser = ParquetSerializer()

    v = 705449463447499237
    predicates = [[("x", "==", v)]]
    result = ser.restore_dataframe(
        reference_store,
        "int64_statistics_overflow.parquet",
        predicate_pushdown_to_io=predicate_pushdown_to_io,
        predicates=predicates,
    )
    assert not result.empty
    assert (result["x"] == v).all()


@pytest.mark.parametrize(
    ["predicate_value", "expected"],
    [
        ([0, 4, 1], True),
        ([-2, 44], False),
        ([-3, 0], True),
        ([-1, 10**4], False),
        ([2, 3], True),
        ([-1, 20], True),
        ([-30, -5, 50, 10], True),
        ([], False),
    ],
)
def test_predicate_accept_in(store, predicate_value, expected):
    df = pd.DataFrame({"A": [0, 4, 13, 29]})  # min = 0, max = 29
    predicate = ("A", "in", predicate_value)
    serialiser = ParquetSerializer(chunk_size=None)
    key = serialiser.store(store, "prefix", df)

    parquet_file = ParquetFile(store.open(key))
    row_meta = parquet_file.metadata.row_group(0)
    arrow_schema = parquet_file.schema.to_arrow_schema()
    parquet_reader = parquet_file.reader
    assert (
        _predicate_accepts(
            predicate,
            row_meta=row_meta,
            arrow_schema=arrow_schema,
            parquet_reader=parquet_reader,
        )
        == expected
    )


def test_read_categorical(store):
    df = pd.DataFrame({"col": ["a"]}).astype({"col": "category"})

    serialiser = ParquetSerializer()
    key = serialiser.store(store, "prefix", df)

    df = serialiser.restore_dataframe(store, key)
    assert df.dtypes["col"] == "O" if not pandas_infer_string() else "string"

    df = serialiser.restore_dataframe(store, key, categories=["col"])
    assert df.dtypes["col"] == pd.CategoricalDtype(["a"], ordered=False)


def test_read_empty_categorical(store):
    df = pd.DataFrame({"col": [None]}).astype({"col": "category"})

    serialiser = ParquetSerializer()
    key = serialiser.store(store, "prefix", df)

    df = serialiser.restore_dataframe(store, key)
    assert df.dtypes["col"] == "O" if not pandas_infer_string() else "string"

    df = serialiser.restore_dataframe(store, key, categories=["col"])
    assert df.dtypes["col"] == pd.CategoricalDtype([], ordered=False)


def test_read_categorical_empty_dataframe(store):
    df = pd.DataFrame({"col": ["a"]}).astype({"col": "category"}).iloc[:0]
    serialiser = ParquetSerializer()
    key = serialiser.store(store, "prefix", df)

    df = serialiser.restore_dataframe(store, key)
    assert df.dtypes["col"] == "O" if not pandas_infer_string() else "string"

    df = serialiser.restore_dataframe(store, key, categories=["col"])

    assert df.dtypes["col"] == pd.CategoricalDtype([], ordered=False)


def test_reset_dict_cols(store):
    df = pd.DataFrame({"col": ["a"], "colB": ["b"]}).astype(
        {"col": "category", "colB": "category"}
    )
    table = pa.Table.from_pandas(df)
    orig_schema = table.schema

    assert pa.types.is_dictionary(orig_schema.field("col").type)
    assert pa.types.is_dictionary(orig_schema.field("colB").type)

    all_reset = _reset_dictionary_columns(table).schema
    assert not pa.types.is_dictionary(all_reset.field("col").type)
    assert not pa.types.is_dictionary(all_reset.field("colB").type)

    only_a_reset = _reset_dictionary_columns(table, exclude=["colB"]).schema
    assert not pa.types.is_dictionary(only_a_reset.field("col").type)
    assert pa.types.is_dictionary(only_a_reset.field("colB").type)


def test_retry_on_IOError(monkeypatch, caplog, store):  # noqa: N802
    """See https://github.com/JDASoftwareGroup/plateau/issues/407 :

    We are testing a retry-workaround for the above issue here. Once the
    issue is resolved, this test and the workaround can be removed.
    """

    df = pd.DataFrame({"A": [0, 1, 2, 3]})

    retry_count = 0

    def patched__restore_dataframe(**kwargs):
        nonlocal retry_count
        retry_count += 1

        if not retry_count > 1:
            # fail for the first try
            raise OSError()
        elif retry_count > 1:
            # simulate a successful retry
            return df

    monkeypatch.setattr(
        ParquetSerializer, "_restore_dataframe", patched__restore_dataframe
    )
    serializer = ParquetSerializer()
    key = serializer.store(store, "key", df)
    df_result = serializer.restore_dataframe(store=store, key=key)
    pdt.assert_frame_equal(df, df_result)


def test_retries_on_IOError_logs(monkeypatch, caplog, store):  # noqa: N802
    """See https://github.com/JDASoftwareGroup/plateau/issues/407 :

    We are testing a retry-workaround for the above issue here. Once the
    issue is resolved, this test and the workaround can be removed.
    """

    def patched__restore_dataframe(**kwargs):
        # This kind of exception should be captured by the retry mechanism.
        raise OSError()

    df = pd.DataFrame({"A": [0, 1, 2, 3]})
    monkeypatch.setattr(
        ParquetSerializer, "_restore_dataframe", patched__restore_dataframe
    )
    serializer = ParquetSerializer()
    key = serializer.store(store, "key", df)

    with pytest.raises(ParquetReadError):
        serializer.restore_dataframe(store=store, key=key)

    assert len(caplog.records) == MAX_NB_RETRIES
    for log_record in caplog.records:
        assert "Failed to restore dataframe" in log_record.message


def test_retry_fail_on_other_error(monkeypatch, caplog, store):
    """See https://github.com/JDASoftwareGroup/plateau/issues/407 : We are
    testing a retry-workaround for the above issue here. Once the issue is
    resolved, this test and the workaround can be removed.

    We only want to retry on OSErrors (and inherited exceptions) -- all
    other exceptions should be raised.
    """

    df = pd.DataFrame({"A": [0, 1, 2, 3]})

    def patched__restore_dataframe(**kwargs):
        # This should not be retried but raised immediately.
        raise ValueError()

    monkeypatch.setattr(
        ParquetSerializer, "_restore_dataframe", patched__restore_dataframe
    )
    serializer = ParquetSerializer()
    key = serializer.store(store, "key", df)

    with pytest.raises(ValueError):
        serializer.restore_dataframe(store=store, key=key)

    assert len(caplog.records) == 0
