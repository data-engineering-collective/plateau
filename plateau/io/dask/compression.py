import logging
from functools import partial

import dask
import dask.dataframe as dd
import pandas as pd

from plateau.core._compat import PANDAS_3

_logger = logging.getLogger()
_PAYLOAD_COL = "__ktk_shuffle_payload"

try:
    # Technically distributed is an optional dependency
    from distributed.protocol import deserialize_bytes, serialize_bytes

    HAS_DISTRIBUTED = True
except ImportError:
    HAS_DISTRIBUTED = False
    serialize_bytes = None
    deserialize_bytes = None

__all__ = (
    "pack_payload_pandas",
    "pack_payload",
    "unpack_payload_pandas",
    "unpack_payload",
)


def pack_payload_pandas(partition: pd.DataFrame, group_key: list[str]) -> pd.DataFrame:
    if not HAS_DISTRIBUTED:
        _logger.warning(
            "Shuffle payload columns cannot be compressed since distributed is not installed."
        )
        return partition

    if partition.empty:
        res = partition[group_key]
        res[_PAYLOAD_COL] = b""
    else:
        res = partition.groupby(
            group_key,
            sort=False,
            observed=True,
            # Keep the as_index s.t. the group values are not dropped. With this
            # the behaviour seems to be consistent along pandas versions
            as_index=True,
        ).apply(lambda x: pd.Series({_PAYLOAD_COL: serialize_bytes(x)}))

        res = res.reset_index()
    return res


def pack_payload(df: dd.DataFrame, group_key: list[str] | str) -> dd.DataFrame:
    """Pack all payload columns (everything except of group_key) into a single
    columns. This column will contain a single byte string containing the
    serialized and compressed payload data. The payload data is just dead
    weight when reshuffling. By compressing it once before the shuffle starts,
    this saves a lot of memory and network/disk IO.

    Example::

        >>> import pandas as pd
        ... import dask.dataframe as dd
        ... from dask.dataframe.shuffle import pack_payload
        ...
        ... df = pd.DataFrame({"A": [1, 1] * 2 + [2, 2] * 2 + [3, 3] * 2, "B": range(12)})
        ... ddf = dd.from_pandas(df, npartitions=2)

        >>> ddf.partitions[0].compute()

        A  B
        0  1  0
        1  1  1
        2  1  2
        3  1  3
        4  2  4
        5  2  5

        >>> pack_payload(ddf, "A").partitions[0].compute()

        A                               __dask_payload_bytes
        0  1  b'\x03\x00\x00\x00\x00\x00\x00\x00)\x00\x00\x03...
        1  2  b'\x03\x00\x00\x00\x00\x00\x00\x00)\x00\x00\x03...


    See also https://github.com/dask/dask/pull/6259
    """

    if (
        # https://github.com/pandas-dev/pandas/issues/34455
        df._meta.index.dtype == "float64"
        # TODO: Try to find out what's going on an file a bug report
        # For datetime indices the apply seems to be corrupt
        # s.t. apply(lambda x:x) returns different values
        or (df._meta.index.dtype == "datetime64[ns]")
    ):
        return df
    if not HAS_DISTRIBUTED:
        _logger.warning(
            "Shuffle payload columns cannot be compressed since distributed is not installed."
        )
        return df

    if not isinstance(group_key, list):
        group_key = [group_key]

    packed_meta = df._meta[group_key]
    packed_meta[_PAYLOAD_COL] = b""

    _pack_payload = partial(pack_payload_pandas, group_key=group_key)

    with dask.config.set({"dataframe.convert-string": False}):
        return df.map_partitions(_pack_payload, meta=packed_meta)


def unpack_payload_pandas(
    partition: pd.DataFrame, unpack_meta: pd.DataFrame
) -> pd.DataFrame:
    """Revert ``pack_payload_pandas`` and restore packed payload.

    unpack_meta:
        A dataframe indicating the schema of the unpacked data. This will be returned in case the input is empty
    """
    if not HAS_DISTRIBUTED:
        _logger.warning(
            "Shuffle payload columns cannot be compressed since distributed is not installed."
        )
        return partition

    if partition.empty:
        return unpack_meta.iloc[:0]
    group_cols = list(set(partition.columns) - {_PAYLOAD_COL})

    def _inner(partition):
        return pd.concat(
            partition[_PAYLOAD_COL].map(deserialize_bytes).values,
            ignore_index=True,
        )

    if not group_cols:
        return _inner(partition)
    return (
        partition.groupby(
            group_cols,
            sort=False,
            observed=True,
        )
        .apply(_inner)
        .reset_index(drop=not PANDAS_3)[unpack_meta.columns]
    )


def unpack_payload(df: dd.DataFrame, unpack_meta: pd.DataFrame) -> dd.DataFrame:
    """Revert payload packing of ``pack_payload`` and restores full
    dataframe."""

    if (
        # https://github.com/pandas-dev/pandas/issues/34455
        (df._meta.index.dtype == "float64")
        # TODO: Try to find out what's going on an file a bug report
        # For datetime indices the apply seems to be corrupt
        # s.t. apply(lambda x:x) returns different values
        or df._meta.index.dtype == "datetime64[ns]"
    ):
        return df

    if not HAS_DISTRIBUTED:
        _logger.warning(
            "Shuffle payload columns cannot be compressed since distributed is not installed."
        )
        return df

    with dask.config.set({"dataframe.convert-string": False}):
        return df.map_partitions(
            unpack_payload_pandas, unpack_meta=unpack_meta, meta=unpack_meta
        )
