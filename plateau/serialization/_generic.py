#!/usr/bin/env python
"""This module contains functionality for persisting/serialising DataFrames.

Available constants

**PredicatesType** - A type describing the format of predicates which is a list of ConjunctionType
**ConjunctionType** - A type describing a single Conjunction which is a list of literals
**LiteralType**  - A type for a single literal

**LiteralValue** - A type indicating the value of a predicate literal


:meta public:
"""

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import pandas as pd
from minimalkv import KeyValueStore
from pandas.api.types import is_list_like

from plateau.serialization._util import _check_contains_null

from ._util import ensure_unicode_string_type

if TYPE_CHECKING:
    import numpy.typing as npt

LiteralValue = TypeVar("LiteralValue")
LiteralType = tuple[str, str, LiteralValue]
ConjunctionType = list[LiteralType]
# Optional is part of the actual type since predicate=None
# is a sential for: All values
PredicatesType = list[ConjunctionType] | None


class DataFrameSerializer:
    """Abstract class that supports serializing DataFrames to/from minimalkv
    stores.

    :meta public:
    """

    _serializers: dict[str, "DataFrameSerializer"] = {}
    type_stable = False

    def __ne__(self, other):
        return not (self == other)

    @classmethod
    def register_serializer(cls, suffix, serializer):
        cls._serializers[suffix] = serializer

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
        """Load a DataFrame from the specified store. The key is also used to
        detect the used format.

        Parameters
        ----------
        store
            store engine
        key
            Key that specifies a path where object should be
            retrieved from the store resource.
        filter_query
            Optional query to filter the DataFrame. Must adhere to the specification
            of pandas.DataFrame.query.
        columns
            Only read in listed columns. When set to None, the full file
            will be read in.
        predicate_pushdown_to_io
            Push predicates through to the I/O layer, default True. Disable
            this if you see problems with predicate pushdown for the given
            file even if the file format supports it. Note that this option
            only hides problems in the store layer that need to be addressed
            there.
        categories
            Columns that should be loaded as categoricals.
        predicates
            Optional list of predicates, like [[('x', '>', 0), ...], that are used
            to filter the resulting DataFrame, possibly using predicate pushdown,
            if supported by the file format.
            This parameter is not compatible with filter_query.

            Predicates are expressed in disjunctive normal form (DNF). This means
            that the innermost tuple describe a single column predicate. These
            inner predicate make are all combined with a conjunction (AND) into a
            larger predicate. The most outer list then combines all predicates
            with a disjunction (OR). By this, we should be able to express all
            kinds of predicates that are possible using boolean logic.
        date_as_object
            Retrieve all date columns as an object column holding datetime.date objects
            instead of pd.Timestamp. Note that this option only works for type-stable
            serializers, e.g. ``ParquetSerializer``.
        """
        if filter_query and predicates:
            raise ValueError("Can only specify one of filter_query and predicates")

        for suffix, serializer in cls._serializers.items():
            if key.endswith(suffix):
                df = serializer.restore_dataframe(
                    store,
                    key,
                    filter_query,
                    columns,
                    predicate_pushdown_to_io=predicate_pushdown_to_io,
                    categories=categories,
                    predicates=predicates,
                    date_as_object=date_as_object,
                )
                df.columns = df.columns.map(ensure_unicode_string_type)
                return df

        # No serialiser matched
        raise ValueError(f"The specified file format for '{key}' is not supported")

    def store(self, store: KeyValueStore, key_prefix: str, df: pd.DataFrame) -> str:
        """Persist a DataFrame to the specified store.

        The used store format (e.g. Parquet) will be appended to the key.

        Parameters
        ----------
        store: minimalkv.KeyValueStore
                store engine
        key_prefix: str
                Key prefix that specifies a path where object should be
                stored on the store resource. The used file format will be
                appended to the key.
        df: pandas.DataFrame or pyarrow.Table
                DataFrame that shall be persisted

        Returns
        -------
        str
            The actual key where the DataFrame is stored.
        """
        raise NotImplementedError("Abstract method called.")


def filter_df(df, filter_query=None):
    """General implementation of query filtering.

    Serialisation formats such as Parquet that support predicate push-
    down may pre-filter in their own implementations.
    """
    if df.shape[0] > 0 and filter_query is not None:
        df = df.query(filter_query)
    return df


def check_predicates(predicates: PredicatesType) -> None:
    """Check if predicates are well-formed."""
    if predicates is None:
        return

    if len(predicates) == 0:
        raise ValueError("Empty predicates")

    for conjunction_idx, conjunction in enumerate(predicates):
        if not isinstance(conjunction, list):
            raise ValueError(
                f"Invalid predicates: Conjunction {conjunction_idx} should be a "
                f"list of 3-tuples, got object of type {type(conjunction)} instead."
            )
        if len(conjunction) == 0:
            raise ValueError(
                f"Invalid predicates: Conjunction {conjunction_idx} is empty"
            )
        for clause_idx, clause in enumerate(conjunction):
            if not isinstance(clause, tuple) and len(clause) == 3:
                raise ValueError(
                    f"Invalid predicates: Clause {clause_idx} in conjunction {conjunction_idx} "
                    f"should be a 3-tuple, got object of type {type(clause)} instead"
                )
            _, op, val = clause
            if (
                isinstance(val, list)
                and any(_check_contains_null(v) for v in val)
                or _check_contains_null(val)
            ):
                raise NotImplementedError(
                    "Null-terminated binary strings are not supported as predicate values."
                )
            if (
                pd.api.types.is_scalar(val)
                and pd.isnull(val)
                and op not in ["==", "!=", "is distinct from"]
            ):
                raise ValueError(
                    f"Invalid predicates: Clause {clause_idx} in conjunction {conjunction_idx} "
                    f"with null value and operator {op}. Only operators supporting null values "
                    "are '==', '!=', 'in' and 'is distinct from'."
                )


def filter_predicates_by_column(
    predicates: PredicatesType, columns: list[str]
) -> PredicatesType | None:
    """Takes a predicate list and removes all literals which are not
    referencing one of the given column.

    .. ipython:: python

        from plateau.serialization import filter_predicates_by_column

        predicates = [[("A", "==", 1), ("B", "<", 5)], [("C", "==", 4)]]

        filter_predicates_by_column(predicates, ["A"])

    Parameters
    ----------
    predicates:
        A list of predicates to be filtered
    columns:
        A list of all columns allowed in the output
    """
    if predicates is None:
        return None
    check_predicates(predicates)
    filtered_predicates = []
    for predicate in predicates:
        new_conjunction = []
        for col, op, val in predicate:
            if col in columns:
                new_conjunction.append((col, op, val))
        if new_conjunction:
            filtered_predicates.append(new_conjunction)
    if filtered_predicates:
        return filtered_predicates
    else:
        return None


def columns_in_predicates(predicates: PredicatesType) -> set[str]:
    """Determine all columns which are mentioned in the list of predicates.

    Parameters
    ----------
    predicates:
        The predicates to be scanned.
    """
    if predicates is None:
        return set()
    check_predicates(predicates)
    # Determine the set of columns that are part of a predicate
    columns = set()
    for predicates_inner in predicates:
        for col, _, _ in predicates_inner:
            columns.add(col)
    return columns


def filter_df_from_predicates(
    df: pd.DataFrame,
    predicates: PredicatesType | None,
    strict_date_types: bool = False,
) -> pd.DataFrame:
    """Filter a `pandas.DataFrame` based on predicates in disjunctive normal
    form.

    Parameters
    ----------
    df
        The pandas DataFrame to be filtered
    predicates
        Predicates in disjunctive normal form (DNF). For a thorough documentation, see
        :class:`DataFrameSerializer.restore_dataframe`
        If None, the df is returned unmodified
    strict_date_types
        If False (default), cast all datelike values to datetime64 for comparison.

    See Also
    --------
    * :ref:`predicate_pushdown`
    """
    if predicates is None:
        return df
    indexer: npt.NDArray[np.bool_] = np.zeros(len(df), dtype=bool)
    for conjunction in predicates:
        inner_indexer: npt.NDArray[np.bool_] = np.ones(len(df), dtype=bool)
        for column, op, value in conjunction:
            column_name = ensure_unicode_string_type(column)
            filter_array_like(
                df[column_name].values,
                op,
                value,
                inner_indexer,
                inner_indexer,
                strict_date_types=strict_date_types,
                column_name=column_name,
            )
        indexer = inner_indexer | indexer
    return df[indexer]


def _handle_categorical_data(array_like, require_ordered):
    if require_ordered and isinstance(array_like.dtype, pd.CategoricalDtype):
        if isinstance(array_like, pd.Categorical):
            categorical = array_like
        else:
            categorical = array_like.cat
        array_value_type = categorical.categories.dtype
        if categorical.categories.is_monotonic_increasing:
            array_like = categorical.as_ordered()
        else:
            array_like = categorical.reorder_categories(
                categorical.categories.sort_values(), ordered=True
            )
    else:
        array_value_type = array_like.dtype
    return array_like, array_value_type


def _handle_null_arrays(array_like, value_dtype):
    # NULL types might not be preserved well, so try to cast floats (pandas default type) to the value type
    # Determine the type using the `kind` interface since this is common for a numpy array, pandas series and pandas extension arrays
    if array_like.dtype.kind == "f" and np.isnan(array_like).all():
        if array_like.dtype.kind != value_dtype.kind:
            if isinstance(value_dtype, pd.api.extensions.ExtensionDtype):
                array_like = pd.Series(array_like, dtype=value_dtype).values
            else:
                array_like = array_like.astype(value_dtype)
    return array_like, array_like.dtype


def _handle_timelike_values(array_value_type, value, value_dtype, strict_date_types):
    def _cast_one_value(val):
        return pd.Series(val, dtype=array_value_type).iloc[0].to_datetime64()

    if is_list_like(value):
        value = [_cast_one_value(val) for val in value]
    else:
        value = _cast_one_value(value)
    value_dtype = pd.Series(value).dtype
    return value, value_dtype


def _ensure_type_stability(
    array_like, value, strict_date_types, require_ordered, column_name=None
):
    """Ensure that the provided value and the provided array will have
    compatible types, such that comparisons are unambiguous.

    The type check is based on the numpy type system and accesses the arrays
    `kind` attribute and asserts equality. The provided value will be
    interpreted as a scalar in this case. For scalars which do not have a proper
    python representation, we will relax the strictness as long as there is a
    valid and unambiguous interpretation of a comparison operation. In
    particular we consider the following combinations valid:

        * unsigned integer (u) <> integer (i)
        * zero-terminated bytes (S) <> Python Object (O)
        * Unicode string (U) <> Python Object (O)

    Parameters
    ----------
    strict_date_types: bool
        If False, assume that datetime.date and datetime.datetime are
        compatible types. In this case, the value is cast appropriately
    require_ordered: bool
        Indicate if the operator to be evaluated will require a notion of
        ordering. In the case of pd.Categorical we will then assume a
        lexicographical ordering and cast the pd.CategoricalDtype accordingly
    column_name: str, optional
        Name of the column where `array_like` originates from, used for nicer
        error messages.
    """

    value_dtype = pd.Series(value if is_list_like(value) else [value]).dtype
    array_like, array_value_type = _handle_categorical_data(array_like, require_ordered)
    array_like, array_value_type = _handle_null_arrays(array_like, value_dtype)

    compatible_types = [
        # UINT and INT
        ("u", "i"),
        ("i", "u"),
        # various string kinds
        ("O", "S"),
        ("O", "U"),
        ("S", "O"),
        ("U", "O"),
        # bool w/ Nones
        ("b", "O"),
    ]

    if not strict_date_types:
        # objects (datetime.date) and datetime64
        compatible_types.append(("O", "M"))

    type_comp = (value_dtype.kind, array_value_type.kind)

    if len(set(type_comp)) > 1 and type_comp not in compatible_types:
        if column_name is None:
            column_name = "<unknown>"
        raise TypeError(
            f"Unexpected type for predicate: Column {column_name!r} has pandas "
            f"type '{array_value_type}', but predicate value {value!r} has "
            f"pandas type '{value_dtype}' (Python type '{type(value)}')."
        )
    if "M" in type_comp:
        value, value_dtype = _handle_timelike_values(
            array_value_type, value, value_dtype, strict_date_types
        )
    return array_like, value


def filter_array_like(
    array_like,
    op: str,
    value,
    mask=None,
    out=None,
    strict_date_types: bool = False,
    column_name: str | None = None,
):
    """Filter an array-like object using operations defined in the predicates.

    Parameters
    ----------
    array_like
        The array like object to be filtered

        See also `pandas.api.types.is_array_like`
    op
    value
    mask
        A boolean array like object which will be combined with the result
        of this evaluation using a logical AND. If an array with all True is
        given, it will be the same result as if left empty
    out
        An array into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated
        array is returned.
    strict_date_types
        If False (default), cast all datelike values to datetime64 for comparison.
    column_name
        Name of the column where `array_like` originates from, used for nicer
        error messages.

    See Also
    --------
    * :ref:`predicate_pushdown`
    """
    if mask is None:
        mask = np.ones(len(array_like), dtype=bool)

    if out is None:
        out = np.zeros(len(array_like), dtype=bool)

    # In the case of an empty list, don't bother with evaluating types, etc.
    if is_list_like(value) and len(value) == 0:
        false_arr: npt.NDArray[np.bool_] = np.zeros(len(array_like), dtype=bool)
        np.logical_and(false_arr, mask, out=out)
        return out

    require_ordered = "<" in op or ">" in op
    array_like, value = _ensure_type_stability(
        array_like, value, strict_date_types, require_ordered, column_name
    )

    with np.errstate(invalid="ignore"):
        if op == "==":
            # If the RHS is null, carry out a null check instead of equality.
            if pd.isnull(value):
                np.logical_and(pd.isnull(array_like), mask, out=out)
            else:
                np.logical_and(array_like == value, mask, out=out)
        elif op == "!=":
            warnings.warn(
                "The current behaviour of the != will be deprecated in version 5.0.0. "
                "The new behaviour will be SQL-like, i.e. NaN values will be filtered out.",
                DeprecationWarning,
                stacklevel=1,
            )
            if pd.isnull(value):
                np.logical_and(~pd.isnull(array_like), mask, out=out)
            else:
                np.logical_and(array_like != value, mask, out=out)
        elif op == "is distinct from":
            # Currently the same as != which will have a change in behaviour in 5.0.
            if pd.isnull(value):
                np.logical_and(~pd.isnull(array_like), mask, out=out)
            else:
                np.logical_and(array_like != value, mask, out=out)
        elif op == "<=":
            np.logical_and(array_like <= value, mask, out=out)
        elif op == ">=":
            np.logical_and(array_like >= value, mask, out=out)
        elif op == "<":
            np.logical_and(array_like < value, mask, out=out)
        elif op == ">":
            np.logical_and(array_like > value, mask, out=out)
        elif op == "in":
            value = np.asarray(value)
            nullmask = pd.isnull(value)
            if value.dtype.kind in ("U", "S", "O"):
                # See GH358

                # If the values include duplicates, this would blow up with the
                # join below, rendering the mask useless
                unique_vals = np.unique(value[~nullmask])
                value_ser = pd.Series(unique_vals, name="value")
                if value_ser.dtype.kind == "S":
                    # pandas 1.4.3+ supports S|*-typed series but cannot handle null
                    # values inside of them. Thus fall back to the old behaviour of
                    # using an object typed series.
                    value_ser = value_ser.astype("object")
                arr_ser = pd.Series(array_like, name="array").to_frame()
                matching_idx = (
                    ~arr_ser.merge(
                        value_ser, left_on="array", right_on="value", how="left"
                    )
                    .value.isna()
                    .values
                )
            else:
                matching_idx = (
                    np.isin(array_like, value)
                    if len(value) > 0
                    else np.zeros(len(array_like), dtype=bool)
                )

            if any(nullmask):
                matching_idx |= pd.isnull(array_like)

            np.logical_and(
                matching_idx,
                mask,
                out=out,
            )
        else:
            raise NotImplementedError("op not supported")

    return out
