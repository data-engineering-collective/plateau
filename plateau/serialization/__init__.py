from ._csv import CsvSerializer
from ._generic import (
    ConjunctionType,
    DataFrameSerializer,
    LiteralType,
    LiteralValue,
    PredicatesType,
    check_predicates,
    columns_in_predicates,
    filter_array_like,
    filter_df,
    filter_df_from_predicates,
    filter_predicates_by_column,
)
from ._parquet import ParquetSerializer

DataFrameSerializer.register_serializer(".csv.gz", CsvSerializer)
DataFrameSerializer.register_serializer(".csv", CsvSerializer)
DataFrameSerializer.register_serializer(".parquet", ParquetSerializer)


def default_serializer():
    return ParquetSerializer()


__all__ = (
    # Serializer classes
    "CsvSerializer",
    "DataFrameSerializer",
    "default_serializer",
    "ParquetSerializer",
    # functions
    "check_predicates",
    "columns_in_predicates",
    "filter_array_like",
    "filter_df_from_predicates",
    "filter_df",
    "filter_predicates_by_column",
    # types
    "ConjunctionType",
    "LiteralType",
    "LiteralValue",
    "PredicatesType",
)
