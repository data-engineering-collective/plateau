"""This module contains functionality for persisting/serialising DataFrames."""

import datetime
import logging
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from minimalkv import KeyValueStore
from pyarrow.parquet import ParquetFile

from ._generic import (
    ConjunctionType,
    DataFrameSerializer,
    PredicatesType,
    check_predicates,
    filter_df,
    filter_df_from_predicates,
)
from ._io_buffer import BlockBuffer
from ._util import ensure_unicode_string_type, schema_metadata_bytes_to_object

try:
    # Only check for BotoStore instance if boto is really installed
    from minimalkv.net.botostore import BotoStore

    HAVE_BOTO = True
except ImportError:
    HAVE_BOTO = False

_logger = logging.getLogger(__name__)


EPOCH_ORDINAL = datetime.date(1970, 1, 1).toordinal()
MAX_NB_RETRIES = 6  # longest retry backoff = BACKOFF_TIME * 2**(MAX_NB_RETRIES - 2)
BACKOFF_TIME = 0.01  # 10 ms
PARQUET_VERSION = "2.4"


def _empty_table_from_schema(parquet_file):
    schema = parquet_file.schema.to_arrow_schema()

    return schema.empty_table()


def _reset_dictionary_columns(table, exclude=None):
    """We need to ensure that the dtype is exactly as requested, see GH227."""
    if exclude is None:
        exclude = []

    schema = table.schema
    for i in range(len(schema)):
        field = schema[i]
        if field.name in exclude:
            continue
        if pa.types.is_dictionary(field.type):
            new_field = pa.field(
                field.name,
                field.type.value_type,
                field.nullable,
                field.metadata,
            )
            schema = schema.remove(i).insert(i, new_field)

    table = table.cast(schema)
    return table


class ParquetReadError(IOError):
    """Internal plateau error while attempting to read Parquet file."""

    pass


class ParquetSerializer(DataFrameSerializer):
    """Serializer to store a :class:`pandas.DataFrame` as parquet.

    On top of the plain serialization, this class handles forward and
    backwards compatibility between pyarrow versions.

    Parameters
    ----------
    compression
        The compression algorithm to be used for the parquet file. For a
        comprehensive list of available compression algorithms, please
        see :func:`pyarrow.parquet.write_table`.
        The default is set to "SNAPPY" which usually offers a good balance
        between performance and compression rate. Depending on your data,
        picking a different algorithm may have vastly different
        characteristics and we can only recommend to test this on your own
        data. Depending on the reader parquet implementation, some
        compression algorithms may not be supported and we recommend to
        consult the documentation of the reader libraries first.
    chunk_size
        The number of rows stored in a Parquet RowGroup. To leverage
        predicate pushdown, it is necessary to set this value. We do not
        apply any default value since a good choice is very sensitive to the
        kind of data you are using and what kind of storage.
        A typical range to try out would be somewhere between 50k-200k. To fully leverage row group statistics, it is highly recommended to sort the file before serialization.

    Notes
    -----

    Regarding type stability and supported types there are a few known limitations users should be aware of.


    .. ipython:: python
        :suppress:

        from plateau.core.utils import ensure_store
        import pandas as pd
        from plateau.serialization import ParquetSerializer

        store = ensure_store("hmemory://")

    * `pandas.Categorical`

        plateau offers the keyword argument `categories` which contains a list of field names which are supposed to retrieved as a `pandas.Categorical`.

        See also :ref:`Dictionary Encoding`

        .. ipython:: python
            :okwarning:

            ser = ParquetSerializer()

            df = pd.DataFrame({"cat_field": pd.Categorical(["A"])})
            df.dtypes
            ser.restore_dataframe(store, ser.store(store, "cat", df))
            ser.restore_dataframe(store, ser.store(store, "cat", df), categories=["cat_field"])

    * Timestamps with nanosecond resolution

        Timestamps can only be stored in micro second (`us`) accuracy. Trying to do differently may raise an exception.

        See also :ref:`timestamp`

        .. ipython:: python
            :okwarning:
            :okexcept:

            import pyarrow as pa

            pa.__version__

            df = pd.DataFrame({"nanosecond": [pd.Timestamp("2021-01-01 00:00:00.0000001")]})
            # nanosecond resolution
            ser.store(store, "key", df)
    """

    type_stable = True

    def __init__(
        self, compression: str = "SNAPPY", chunk_size: int | None = None
    ) -> None:
        self.compression = compression

        if chunk_size is not None:
            if not isinstance(chunk_size, int):
                raise TypeError(
                    "Cannot initialize ParquetSerializer because chunk size is not integer type"
                )
            if chunk_size < 1:
                raise ValueError(
                    "Cannot initialize ParquetSerializer because chunk size < 1"
                )
        self.chunk_size = chunk_size

    def __eq__(self, other):
        return (
            isinstance(other, ParquetSerializer)
            and (self.compression == other.compression)
            and (self.chunk_size == other.chunk_size)
        )

    def __repr__(self):
        return f"ParquetSerializer(compression={self.compression!r}, chunk_size={self.chunk_size!r})"

    @staticmethod
    def _restore_dataframe(
        store: KeyValueStore,
        key: str,
        filter_query: str | None = None,
        columns: Iterable[str] | None = None,
        predicate_pushdown_to_io: bool = True,
        categories: Iterable[str] | None = None,
        predicates: PredicatesType | None = None,
        date_as_object: bool = False,
    ) -> pd.DataFrame:
        check_predicates(predicates)
        _coerce = {"coerce_temporal_nanoseconds": True}

        # If we want to do columnar access we can benefit from partial reads
        # otherwise full read en block is the better option.
        if (not predicate_pushdown_to_io) or (columns is None and predicates is None):
            with pa.BufferReader(store.get(key)) as reader:
                table = pq.read_pandas(reader, columns=columns)
        else:
            if HAVE_BOTO and isinstance(store, BotoStore):
                # Parquet and seeks on S3 currently leak connections thus
                # we omit column projection to the store.
                reader = pa.BufferReader(store.get(key))
            else:
                reader = store.open(key)
                # Buffer at least 4 MB in requests. This is chosen because the default block size of the Azure
                # storage client is 4MB.
                reader = BlockBuffer(reader, 4 * 1024 * 1024)
            try:
                parquet_file = ParquetFile(reader)

                # Check earlier for missing columns to produce the same error
                # as we ever did with earlier pyarrow versions
                if columns is not None:
                    missing_columns = set(columns) - set(
                        parquet_file.schema.to_arrow_schema().names
                    )
                    if missing_columns:
                        raise ValueError(
                            "Columns cannot be found in stored dataframe: {missing}".format(
                                missing=", ".join(sorted(missing_columns))
                            )
                        )

                if predicates and parquet_file.metadata.num_rows > 0:
                    # We need to calculate different predicates for predicate
                    # pushdown and the later DataFrame filtering. This is required
                    # e.g. in the case where we have an `in` predicate as this has
                    # different normalized values.
                    columns_to_io = _columns_for_pushdown(columns, predicates)
                    predicates_for_pushdown = _normalize_predicates(
                        parquet_file, predicates, True
                    )
                    predicates = _normalize_predicates(parquet_file, predicates, False)
                    tables = _read_row_groups_into_tables(
                        parquet_file, columns_to_io, predicates_for_pushdown
                    )

                    if len(tables) == 0:
                        table = _empty_table_from_schema(parquet_file)
                    else:
                        table = pa.concat_tables(tables)
                else:
                    # ARROW-5139 Column projection with empty columns returns a table w/out index
                    if columns == []:
                        # Create an arrow table with expected index length.
                        df = (
                            parquet_file.schema.to_arrow_schema()
                            .empty_table()
                            .to_pandas(date_as_object=date_as_object, **_coerce)
                        )
                        index = pd.Index(
                            pd.RangeIndex(start=0, stop=parquet_file.metadata.num_rows),
                            dtype="int64",
                        )
                        df = pd.DataFrame(df, index=index)
                        # convert back to table to keep downstream code untouched by this patch
                        table = pa.Table.from_pandas(df)
                    else:
                        table = pq.read_pandas(reader, columns=columns)
            finally:
                reader.close()

        if columns is not None:
            missing_columns = set(columns) - set(table.schema.names)
            if missing_columns:
                raise ValueError(
                    "Columns cannot be found in stored dataframe: {missing}".format(
                        missing=", ".join(sorted(missing_columns))
                    )
                )

        table = _reset_dictionary_columns(table, exclude=categories)

        # HACK: Cast bytes to object in metadata until Pandas bug is fixed: https://github.com/pandas-dev/pandas/issues/50127
        table = table.cast(schema_metadata_bytes_to_object(table.schema))

        df = table.to_pandas(date_as_object=date_as_object, **_coerce)

        # XXX: Patch until Pyarrow bug is resolved: https://github.com/apache/arrow/issues/33297
        if categories:
            for col in categories:
                if col in df:
                    df[col] = df[col].astype("category")

        df.columns = df.columns.map(ensure_unicode_string_type)
        if predicates:
            df = filter_df_from_predicates(
                df, predicates, strict_date_types=date_as_object
            )
        else:
            df = filter_df(df, filter_query)
        if columns is not None:
            return df.reindex(columns=columns, copy=False)
        else:
            return df

    @classmethod
    def restore_dataframe(
        cls,
        store: KeyValueStore,
        key: str,
        filter_query: str | None = None,
        columns: Iterable[str] | None = None,
        predicate_pushdown_to_io: bool = True,
        categories: Iterable[str] | None = None,
        predicates: PredicatesType | None = None,
        date_as_object: bool = False,
    ) -> pd.DataFrame:
        # https://github.com/JDASoftwareGroup/plateau/issues/407  We have been seeing weird `IOError`s while reading
        # Parquet files from Azure Blob Store. These errors have caused long running computations to fail.
        # The workaround is to retry the serialization here and gain more stability for long running tasks.
        # This code should not live forever, it should be removed once the underlying cause has been resolved.
        for nb_retry in range(MAX_NB_RETRIES):
            try:
                return cls._restore_dataframe(
                    store=store,
                    key=key,
                    filter_query=filter_query,
                    columns=columns,
                    predicate_pushdown_to_io=predicate_pushdown_to_io,
                    categories=categories,
                    predicates=predicates,
                    date_as_object=date_as_object,
                )
            # We only retry OSErrors (note that IOError inherits from OSError), as these kind of errors may benefit
            # from retries.
            except OSError as err:
                raised_error = err
                _logger.warning(
                    msg=(
                        f"Failed to restore dataframe, attempt {nb_retry + 1} of {MAX_NB_RETRIES} with parameters "
                        f"key: {key}, filter_query: {filter_query}, columns: {columns}, "
                        f"predicate_pushdown_to_io: {predicate_pushdown_to_io}, categories: {categories}, "
                        f"predicates: {predicates}, date_as_object: {date_as_object}."
                    ),
                    exc_info=True,
                )
                # we don't sleep when we're done with the last attempt
                if nb_retry < (MAX_NB_RETRIES - 1):
                    time.sleep(BACKOFF_TIME * 2**nb_retry)

        raise ParquetReadError(
            f"Failed to restore dataframe after {MAX_NB_RETRIES} attempts. Parameters: "
            f"key: {key}, filter_query: {filter_query}, columns: {columns}, "
            f"predicate_pushdown_to_io: {predicate_pushdown_to_io}, categories: {categories}, "
            f"date_as_object: {date_as_object}, predicates: {predicates}."
        ) from raised_error

    def store(self, store, key_prefix, df):
        key = f"{key_prefix}.parquet"
        if isinstance(df, pa.Table):
            table = df
        else:
            table = pa.Table.from_pandas(df)
        buf = pa.BufferOutputStream()

        pq.write_table(
            table,
            buf,
            version=PARQUET_VERSION,
            chunk_size=self.chunk_size,
            compression=self.compression,
            coerce_timestamps="us",
        )
        store.put(key, buf.getvalue().to_pybytes())
        return key


def _columns_for_pushdown(columns, predicates):
    if columns is None:
        return
    new_cols = columns[:]
    for conjunction in predicates:
        for literal in conjunction:
            if literal[0] not in columns:
                new_cols.append(literal[0])
    return new_cols


def _read_row_groups_into_tables(parquet_file, columns, predicates_in):
    """For each RowGroup check if the predicate in DNF applies and then read
    the respective RowGroup."""
    arrow_schema = parquet_file.schema.to_arrow_schema()
    parquet_reader = parquet_file.reader

    def all_predicates_accept(row):
        # Check if the predicates evaluate on this RowGroup.
        # As the predicate is in DNF, we only need a single of the
        # inner lists to match. Once we have found a positive match,
        # there is no need to check whether the remaining ones apply.
        row_meta = parquet_file.metadata.row_group(row)
        for predicate_list in predicates_in:
            if all(
                _predicate_accepts(predicate, row_meta, arrow_schema, parquet_reader)
                for predicate in predicate_list
            ):
                return True
        return False

    # Iterate over the RowGroups and evaluate the list of predicates on each
    # one of them. Only access those that could contain a row where we could
    # get an exact match of the predicate.
    result = []
    for row in range(parquet_file.num_row_groups):
        if all_predicates_accept(row):
            row_group = parquet_file.read_row_group(row, columns=columns)
            result.append(row_group)
    return result


def _normalize_predicates(
    parquet_file, predicates: list[ConjunctionType], for_pushdown
):
    schema = parquet_file.schema.to_arrow_schema()

    normalized_predicates = []
    for conjunction in predicates:
        evaluates_to_false = False
        new_conjunction: list[Any] = []

        for literal in conjunction:
            col, op, val = literal
            col_idx = parquet_file.reader.column_name_idx(col)
            pa_type = schema[col_idx].type
            column_name = schema[col_idx].name

            if pa.types.is_null(pa_type):
                # early exit, the entire conjunction evaluates to False
                evaluates_to_false = True
                break

            if op == "in":
                normalized_value = [
                    _normalize_value(lit, pa_type, column_name=column_name)
                    for lit in literal[2]
                ]
            else:
                normalized_value = _normalize_value(
                    literal[2], pa_type, column_name=column_name
                )
            new_literal = (literal[0], literal[1], normalized_value)
            new_conjunction.append(new_literal)

        if not evaluates_to_false:
            normalized_predicates.append(new_conjunction)
    return normalized_predicates


def _normalize_value(value, pa_type, column_name=None):
    if pa.types.is_dictionary(pa_type):
        pa_type = pa_type.value_type

    if pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        elif isinstance(value, str):
            return value
        elif value is None:
            return value
    elif pa.types.is_binary(pa_type):
        if isinstance(value, bytes):
            return value
        elif isinstance(value, str):
            return str(value).encode("utf-8")
    elif (
        pa.types.is_integer(pa_type)
        and pd.api.types.is_integer(value)
        or pa.types.is_floating(pa_type)
        and pd.api.types.is_float(value)
        or pa.types.is_boolean(pa_type)
        and pd.api.types.is_bool(value)
        or pa.types.is_timestamp(pa_type)
        and not isinstance(value, bytes | str)
        and (
            pd.api.types.is_datetime64_dtype(value)
            or isinstance(value, datetime.datetime)
        )
    ):
        return value
    elif pa.types.is_date(pa_type):
        if isinstance(value, str):
            return datetime.datetime.strptime(value, "%Y-%m-%d").date()
        elif isinstance(value, bytes):
            value = value.decode("utf-8")
            return datetime.datetime.strptime(value, "%Y-%m-%d").date()
        elif isinstance(value, datetime.date):
            if isinstance(value, datetime.datetime):
                raise TypeError(
                    f"Unexpected type for predicate: Column {column_name!r} is an "
                    f"Arrow date ({pa_type}), but predicate value has type {type(value)}. "
                    f"Use a Python 'datetime.date' object instead."
                )
            else:
                return value
    predicate_value_dtype = pd.Series(value).dtype
    raise TypeError(
        f"Unexpected type for predicate: Column {column_name!r} has pandas type "
        f"{pa_type.to_pandas_dtype()} (Arrow type {pa_type}), but predicate value "
        f"{value!r} has pandas type '{predicate_value_dtype}' (Python type '{type(value)}')"
    )


def _predicate_accepts(predicate, row_meta, arrow_schema, parquet_reader):
    """Checks if a predicate evaluates on a column.

    This method first casts the value of the predicate to the type used
    for this column in the statistics and then applies the relevant
    operator. The operation applied here is done in a fashion to check
    if the predicate would evaluate to True for any possible row in the
    RowGroup. Thus e.g. for the `==` predicate, we check if the
    predicate value is in the (min, max) range of the RowGroup.
    """
    col, op, val = predicate
    col_idx = parquet_reader.column_name_idx(col)
    pa_type = arrow_schema[col_idx].type
    parquet_statistics = row_meta.column(col_idx).statistics

    # In case min/max is not set, we have to assume that the predicate matches.
    if not parquet_statistics.has_min_max:
        return True

    min_value = parquet_statistics.min
    max_value = parquet_statistics.max
    # Transform the predicate value to the respective type used in the statistics.

    # integer overflow protection since statistics are stored as signed integer, see ARROW-5166
    if pa.types.is_integer(pa_type) and (max_value < min_value):
        return True

    if pa.types.is_timestamp(pa_type):
        # timestamps in the parquet statistic might be of type datetime.datetime, which is not compatible w/ numpy
        min_value = np.datetime64(min_value)
        max_value = np.datetime64(max_value)

    # The statistics for floats only contain the 6 most significant digits.
    # So a suitable epsilon has to be considered below min and above max.
    if isinstance(val, float):
        min_value -= _epsilon(min_value)
        max_value += _epsilon(max_value)

    # op can only be "==" or "!=" for scalar null values.
    if op == "==":
        if pd.isnull(val):
            return parquet_statistics.null_count > 0
        else:
            return (min_value <= val) and (max_value >= val)
    elif op == "!=":
        if pd.isnull(val):
            return parquet_statistics.null_count < row_meta.num_rows
        else:
            return not ((min_value >= val) and (max_value <= val))
    elif op == "<=":
        return min_value <= val
    elif op == ">=":
        return max_value >= val
    elif op == "<":
        return min_value < val
    elif op == ">":
        return max_value > val
    elif op == "in":
        # This implementation is chosen for performance reasons. See
        # https://github.com/JDASoftwareGroup/kartothek/pull/130 for more information/benchmarks.
        # We accept the predicate if there is any value in the provided array which is equal to or between
        # the parquet min and max statistics. Otherwise, it is rejected.
        for x in val:
            if pd.isnull(x):
                if parquet_statistics.null_count > 0:
                    return True
            elif min_value <= x <= max_value:
                return True
        return False
    else:
        raise NotImplementedError("op not supported")


def _highest_significant_position(num):
    """
    >>> _highest_significant_position(1.0)
    1
    >>> _highest_significant_position(9.0)
    1
    >>> _highest_significant_position(39.0)
    2
    >>> _highest_significant_position(0.1)
    -1
    >>> _highest_significant_position(0.9)
    -1
    >>> _highest_significant_position(0.000123)
    -4
    >>> _highest_significant_position(1234567.0)
    7
    >>> _highest_significant_position(-0.1)
    -1
    >>> _highest_significant_position(-100.0)
    3
    """
    abs_num = np.absolute(num)
    log_of_abs = np.log10(abs_num)
    position = int(np.floor(log_of_abs))

    # is position left of decimal point?
    if abs_num >= 1.0:
        position += 1

    return position


def _epsilon(num):
    """
    >>> _epsilon(123456)
    1
    >>> _epsilon(0.123456)
    1e-06
    >>> _epsilon(0.123)
    1e-06
    >>> _epsilon(0)
    0
    >>> _epsilon(-0.123456)
    1e-06
    >>> _epsilon(-123456)
    1
    >>> _epsilon(np.inf)
    0
    >>> _epsilon(-np.inf)
    0
    """
    SIGNIFICANT_DIGITS = 6

    if num == 0 or np.isinf(num):
        return 0

    epsilon_position = _highest_significant_position(num) - SIGNIFICANT_DIGITS

    # is position right of decimal point?
    if epsilon_position < 0:
        epsilon_position += 1

    return 10**epsilon_position
