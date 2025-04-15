"""This module is a collection of helper functions."""

import collections
import inspect
import logging
from collections.abc import Iterable
from typing import Literal, cast, overload

import decorator
import pandas as pd
import polars as pl
import pyarrow as pa

from plateau.core.dataset import DatasetMetadata, DatasetMetadataBase
from plateau.core.factory import _ensure_factory
from plateau.core.typing import StoreFactory, StoreInput
from plateau.core.utils import ensure_store, lazy_store

signature = inspect.signature


LOGGER = logging.getLogger(__name__)


class InvalidObject:
    """Sentinel to mark keys for removal."""

    pass


def combine_metadata(dataset_metadata: list[dict], append_to_list: bool = True) -> dict:
    """Merge a list of dictionaries.

    The merge is performed in such a way, that only keys which
    are present in **all** dictionaries are kept in the final result.

    If lists are encountered, the values of the result will be the
    concatenation of all list values in the order of the supplied dictionary list.
    This behaviour may be changed by using append_to_list

    Parameters
    ----------
    dataset_metadata
        The list of dictionaries (usually metadata) to be combined.
    append_to_list
        If True, all values are concatenated. If False, only unique values are kept
    """
    meta = _combine_metadata(dataset_metadata, append_to_list)
    return _remove_invalids(meta)


def _remove_invalids(dct):
    if not isinstance(dct, dict):
        return {}

    new_dict = {}
    for key, value in dct.items():
        if isinstance(value, dict):
            tmp = _remove_invalids(value)
            # Do not propagate empty dicts
            if tmp:
                new_dict[key] = tmp
        elif not isinstance(value, InvalidObject):
            new_dict[key] = value
    return new_dict


def _combine_metadata(dataset_metadata, append_to_list):
    assert isinstance(dataset_metadata, list)
    if len(dataset_metadata) == 1:
        return dataset_metadata.pop()

    # In case the input list has only two elements, we can do simple comparison
    if len(dataset_metadata) > 2:
        first = _combine_metadata(dataset_metadata[::2], append_to_list)
        second = _combine_metadata(dataset_metadata[1::2], append_to_list)
        final = _combine_metadata([first, second], append_to_list)
        return final
    else:
        first = dataset_metadata.pop()
        second = dataset_metadata.pop()
        if first == second:
            return first
        # None is harmless and may occur if a key appears in one but not the other dict
        elif first is None or second is None:
            return first if first is not None else second
        elif isinstance(first, dict) and isinstance(second, dict):
            new_dict = {}
            keys = set(first.keys())
            keys.update(second.keys())
            for key in keys:
                new_dict[key] = _combine_metadata(
                    [first.get(key), second.get(key)], append_to_list
                )
            return new_dict
        elif isinstance(first, list) and isinstance(second, list):
            new_list = first + second
            if append_to_list:
                return new_list
            else:
                return list(set(new_list))
        else:
            return InvalidObject()


def _ensure_compatible_indices(
    dataset: DatasetMetadataBase | None,
    secondary_indices: Iterable[str],
) -> list[str]:
    if dataset:
        ds_secondary_indices = sorted(dataset.secondary_indices.keys())

        if secondary_indices and not set(secondary_indices).issubset(
            ds_secondary_indices
        ):
            raise ValueError(
                f"Incorrect indices provided for dataset.\n"
                f"Expected: {ds_secondary_indices}\n"
                f"But got: {secondary_indices}"
            )

        return ds_secondary_indices
    return sorted(secondary_indices)


def group_table_by_partition_keys(table: pa.Table, partition_on: list[str]):
    """Yield tuples of partition keys and pyarrow tables (excluding the partition_on columns) using polars.

    Pyarrow's groupby is not really useful for this specific purpose, thus the detour through polars.
    """

    df = pl.from_arrow(table)

    groups = df.group_by(partition_on, maintain_order=True)

    for key, group in groups:
        arrow_table = group.drop(partition_on).to_arrow()  # drop partition keys
        yield key, arrow_table


def validate_partition_keys(
    dataset_uuid,
    store,
    ds_factory,
    default_metadata_version,
    partition_on,
):
    if ds_factory or DatasetMetadata.exists(dataset_uuid, ensure_store(store)):
        ds_factory = _ensure_factory(
            dataset_uuid=dataset_uuid,
            store=store,
            factory=ds_factory,
        )

        ds_metadata_version = ds_factory.metadata_version
        if partition_on:
            if not isinstance(partition_on, list):
                partition_on = [partition_on]
            if partition_on != ds_factory.partition_keys:
                raise ValueError(
                    "Incompatible set of partition keys encountered. "
                    f"Input partitioning was `{partition_on}` while actual dataset was `{ds_factory.partition_keys}`"
                )
        else:
            partition_on = ds_factory.partition_keys
    else:
        ds_factory = None
        ds_metadata_version = default_metadata_version
    return ds_factory, ds_metadata_version, partition_on


_NORMALIZE_ARGS_LIST = [
    "partition_on",
    "delete_scope",
    "secondary_indices",
    "sort_partitions_by",
    "bucket_by",
]

_NORMALIZE_ARGS = _NORMALIZE_ARGS_LIST + ["store", "dispatch_by"]
_NormalizeArgsLiteral = Literal[
    "partition_on",
    "delete_scope",
    "secondary_indices",
    "sort_partitions_by",
    "bucket_by",
    "store",
    "dispatch_by",
]


@overload
def normalize_arg(
    arg_name: Literal[
        "partition_on",
        "delete_scope",
        "secondary_indices",
        "bucket_by",
        "sort_partitions_by",
        "dispatch_by",
    ],
    old_value: None,
) -> None: ...


@overload
def normalize_arg(
    arg_name: Literal[
        "partition_on",
        "delete_scope",
        "secondary_indices",
        "bucket_by",
        "sort_partitions_by",
        "dispatch_by",
    ],
    old_value: str | list[str],
) -> list[str]: ...


@overload
def normalize_arg(
    arg_name: Literal["store"], old_value: StoreInput | None
) -> StoreFactory: ...


def normalize_arg(arg_name, old_value):
    """Normalizes an argument according to pre-defined types.

    Type A:

    * "partition_on"
    * "delete_scope"
    * "secondary_indices"
    * "dispatch_by"

    will be converted to a list. If it is None, an empty list will be created

    Type B:
    * "store"

    Will be converted to a callable returning

    :meta private:
    """

    def _make_list(_args):
        if isinstance(_args, str | bytes | int | float):
            return [_args]
        if _args is None:
            return []
        if isinstance(_args, set | frozenset | dict):
            raise ValueError(f"{type(_args)} is incompatible for normalisation.")
        return list(_args)

    if arg_name in _NORMALIZE_ARGS_LIST:
        if old_value is None:
            return []
        elif isinstance(old_value, list):
            return old_value
        else:
            return _make_list(old_value)
    elif arg_name == "dispatch_by":
        if old_value is None:
            return old_value
        elif isinstance(old_value, list):
            return old_value
        else:
            return _make_list(old_value)
    elif arg_name == "store" and old_value is not None:
        return lazy_store(old_value)

    return old_value


@decorator.decorator
def normalize_args(function, *args, **kwargs):
    sig = signature(function)

    def _wrapper(*args, **kwargs):
        for arg_name in _NORMALIZE_ARGS:
            arg_name = cast(_NormalizeArgsLiteral, arg_name)
            if arg_name in sig.parameters.keys():
                ix = inspect.getfullargspec(function).args.index(arg_name)
                if arg_name in kwargs:
                    kwargs[arg_name] = normalize_arg(arg_name, kwargs[arg_name])
                elif len(args) > ix:
                    new_args = list(args)
                    new_args[ix] = normalize_arg(arg_name, args[ix])
                    args = tuple(new_args)
                else:
                    kwargs[arg_name] = normalize_arg(arg_name, None)
        return function(*args, **kwargs)

    return _wrapper(*args, **kwargs)


def extract_duplicates(lst):
    """Return all items of a list that occur more than once.

    Parameters
    ----------
    lst: List[Any]

    Returns
    -------
    lst: List[Any]
    """

    return [item for item, count in collections.Counter(lst).items() if count > 1]


def align_categories(dfs, categoricals):
    """Takes a list of dataframes with categorical columns and determines the
    superset of categories. All specified columns will then be cast to the same
    `pd.CategoricalDtype`

    Parameters
    ----------
    dfs: List[pd.DataFrame]
        A list of dataframes for which the categoricals should be aligned
    categoricals: List[str]
        Columns holding categoricals which should be aligned

    Returns
    -------
    List[pd.DataFrame]
        A list with aligned dataframes
    """
    if len(categoricals) == 0:
        return dfs

    col_dtype = {}

    for column in categoricals:
        position_largest_df = None
        categories = set()
        largest_df_categories = set()
        for ix, df in enumerate(dfs):
            ser = df[column]
            if not isinstance(ser.dtype, pd.CategoricalDtype):
                cats = ser.dropna().unique()
                LOGGER.info(
                    "Encountered non-categorical type where categorical was expected\n"
                    f"Found at index position {ix} for column {column}\n"
                    f"Dtypes: {df.dtypes}"
                )
            else:
                cats = ser.cat.categories
                length = len(df)
                if position_largest_df is None or length > position_largest_df[0]:
                    position_largest_df = (length, ix)
                if position_largest_df[1] == ix:
                    largest_df_categories = cats
            categories.update(cats)

        # use the categories of the largest DF as a baseline to avoid having
        # to rewrite its codes. Append the remainder and sort it for reproducibility
        categories_lst = list(largest_df_categories) + sorted(
            set(categories) - set(largest_df_categories)
        )
        cat_dtype = pd.api.types.CategoricalDtype(categories_lst, ordered=False)
        col_dtype[column] = cat_dtype

    return_dfs = []
    for df in dfs:
        try:
            new_df = df.astype(col_dtype, copy=False)
        except ValueError as verr:
            cat_types = {
                col: dtype.categories.dtype for col, dtype in col_dtype.items()
            }
            # Should be fixed by pandas>=0.24.0
            if "buffer source array is read-only" in str(verr):
                new_df = df.astype(cat_types)
                new_df = new_df.astype(col_dtype)
            else:
                raise verr
        return_dfs.append(new_df)
    return return_dfs


def sort_values_categorical(df: pd.DataFrame, columns: list[str] | str) -> pd.DataFrame:
    """Sort a dataframe lexicographically by the categories of column
    `column`"""
    if not isinstance(columns, list):
        columns = [columns]
    for col in columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            cat_accessor = df[col].cat
            df[col] = cat_accessor.reorder_categories(
                sorted(cat_accessor.categories), ordered=True
            )
    return df.sort_values(by=columns).reset_index(drop=True)


def raise_if_indices_overlap(partition_on, secondary_indices):
    partition_secondary_overlap = set(partition_on) & set(secondary_indices)
    if partition_secondary_overlap:
        raise RuntimeError(
            f"Cannot create secondary index on partition columns: {partition_secondary_overlap}"
        )
