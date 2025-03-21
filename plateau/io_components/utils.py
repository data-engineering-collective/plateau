"""This module is a collection of helper functions."""

import collections
import inspect
import logging
from collections.abc import Iterable
from typing import Literal, cast, overload

import decorator
import pandas as pd
import polars as pl

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
    """Extract duplicate values from a list.

    Parameters
    ----------
    lst: List[Any]
        The list to check for duplicates

    Returns
    -------
    List[Any]
        The list of duplicate values
    """
    return [item for item, count in collections.Counter(lst).items() if count > 1]


def align_categories(dfs, categoricals):
    """Align the categories of a list of DataFrames.

    Parameters
    ----------
    dfs: List[Union[pandas.DataFrame, polars.DataFrame]]
        The list of DataFrames to align
    categoricals: List[str]
        The list of categorical columns to align

    Returns
    -------
    List[Union[pandas.DataFrame, polars.DataFrame]]
        The list of DataFrames with aligned categories
    """
    if not dfs:
        return dfs

    # Handle Polars DataFrames
    if isinstance(dfs[0], pl.DataFrame):
        for col in categoricals:
            # Get all unique values across all DataFrames
            unique_values = set()
            for df in dfs:
                if col in df.columns:
                    unique_values.update(df[col].unique().to_list())

            # Convert to list and sort for deterministic order
            unique_values = sorted(unique_values)

            # Update each DataFrame to use the same categories
            for i, df in enumerate(dfs):
                if col in df.columns:
                    dfs[i] = df.with_columns(
                        pl.col(col)
                        .cast(pl.Categorical)
                        .cast(pl.Categorical(unique_values))
                    )
        return dfs

    # Handle Pandas DataFrames (existing code)
    if not isinstance(dfs[0], pd.DataFrame):
        return dfs

    # Existing Pandas implementation...
    categories = {}
    for cat in categoricals:
        for df in dfs:
            if cat in df.columns:
                if cat not in categories:
                    categories[cat] = set()
                categories[cat].update(df[cat].dropna().unique())

    for cat in categories:
        categories[cat] = list(categories[cat])

    dfs = [df.copy() for df in dfs]
    for cat in categories:
        for df in dfs:
            if cat in df.columns:
                df[cat] = pd.Categorical(df[cat], categories=categories[cat])
    return dfs


def sort_values_categorical(df, columns):
    """Sort a DataFrame by categorical columns.

    Parameters
    ----------
    df: Union[pandas.DataFrame, polars.DataFrame]
        The DataFrame to sort
    columns: Union[str, List[str]]
        The columns to sort by

    Returns
    -------
    Union[pandas.DataFrame, polars.DataFrame]
        The sorted DataFrame
    """
    if isinstance(df, pl.DataFrame):
        if isinstance(columns, str):
            columns = [columns]
        return df.sort(columns)
    elif isinstance(df, pd.DataFrame):
        return df.sort_values(columns)
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(df)}")


def raise_if_indices_overlap(partition_on, secondary_indices):
    """Raise an error if partition keys and secondary indices overlap.

    Parameters
    ----------
    partition_on: List[str]
        The list of partition keys
    secondary_indices: List[str]
        The list of secondary indices

    Raises
    ------
    ValueError
        If there is overlap between partition keys and secondary indices
    """
    if partition_on and secondary_indices:
        duplicates = set(partition_on) & set(secondary_indices)
        if duplicates:
            raise ValueError(
                f"Partition columns and secondary indices overlap: {duplicates}"
            )
