import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import (
    SupportsFloat,
    cast,
)

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from plateau.core.common_metadata import empty_dataframe_from_schema
from plateau.core.docs import default_docs
from plateau.core.factory import DatasetFactory, _ensure_factory
from plateau.core.naming import DEFAULT_METADATA_VERSION
from plateau.core.typing import StoreFactory, StoreInput
from plateau.io.dask.compression import pack_payload, unpack_payload_pandas
from plateau.io_components.metapartition import (
    _METADATA_SCHEMA,
    SINGLE_TABLE,
    MetaPartition,
    parse_input_to_metapartition,
)
from plateau.io_components.read import dispatch_metapartitions_from_factory
from plateau.io_components.update import update_dataset_from_partitions
from plateau.io_components.utils import (
    _ensure_compatible_indices,
    normalize_args,
    validate_partition_keys,
)
from plateau.io_components.write import (
    raise_if_dataset_exists,
    store_dataset_from_partitions,
    write_partition,
)
from plateau.serialization import DataFrameSerializer, PredicatesType

from ._shuffle import shuffle_store_dask_partitions
from ._utils import _maybe_get_categoricals_from_index

__all__ = (
    "read_dataset_as_ddf",
    "store_dataset_from_ddf",
    "update_dataset_from_ddf",
    "collect_dataset_metadata",
    "hash_dataset",
)
from dask.dataframe import from_map
from dask.dataframe.io.utils import DataFrameIOFunction


class ReadPlateauPartition(DataFrameIOFunction):
    def __init__(
        self,
        columns,
    ) -> None:
        self._columns = columns
        super().__init__()

    @property
    def columns(self):
        """Return the current column projection."""
        return self._columns

    def project_columns(self, columns):
        """Return a new DataFrameIOFunction object with a new column
        projection."""
        return type(self)(columns=columns)

    def __call__(self, mps, *args, columns: Sequence[str] | None = None, **kwargs):
        """Return a new DataFrame partition."""
        cols = (
            self.columns
            if columns is None
            else [c for c in columns if c in self.columns]
        )
        return MetaPartition.concat_metapartitions(
            [mp.load_dataframes(*args, columns=cols, **kwargs) for mp in mps]
        ).data


@default_docs
@normalize_args
def read_dataset_as_ddf(
    dataset_uuid=None,
    store=None,
    table=SINGLE_TABLE,
    columns=None,
    predicate_pushdown_to_io=True,
    categoricals: Sequence[str] | None = None,
    dates_as_object: bool = True,
    predicates=None,
    factory=None,
    dask_index_on=None,
    dispatch_by=None,
):
    """Retrieve a single table from a dataset as partition-individual
    :class:`~dask.dataframe.DataFrame` instance.

    Please take care when using categoricals with Dask. For index columns, this function will construct dataset
    wide categoricals. For all other columns, Dask will determine the categories on a partition level and will
    need to merge them when shuffling data.

    Parameters
    ----------
    dask_index_on: str
        Reconstruct (and set) a dask index on the provided index column. Cannot be used
        in conjunction with `dispatch_by`.

        For details on performance, see also `dispatch_by`
    """
    if dask_index_on is not None and not isinstance(dask_index_on, str):
        raise TypeError(
            f"The parameter `dask_index_on` must be a string but got {type(dask_index_on)}"
        )

    if dask_index_on is not None and dispatch_by is not None and len(dispatch_by) > 0:
        raise ValueError(
            "`read_dataset_as_ddf` got parameters `dask_index_on` and `dispatch_by`. "
            "Note that `dispatch_by` can only be used if `dask_index_on` is None."
        )

    ds_factory = _ensure_factory(
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
    )

    if isinstance(columns, dict):
        columns = columns[table]
    meta = _get_dask_meta_for_dataset(
        ds_factory, columns, categoricals, dates_as_object
    )

    if columns is None:
        columns = list(meta.columns)
    mps = list(
        dispatch_metapartitions_from_factory(
            dataset_factory=ds_factory,
            predicates=predicates,
            dispatch_by=dispatch_by or dask_index_on,  # type: ignore
        )
    )
    divisions_lst = None
    if dask_index_on:
        ds_factory.load_index(dask_index_on)
        divisions = ds_factory.indices[dask_index_on].observed_values()
        divisions_lst = list(divisions)
        divisions_lst = sorted(divisions_lst)
        divisions_lst.append(divisions_lst[-1])

    with dask.config.set({"dataframe.convert-string": False}):
        ddf = from_map(
            ReadPlateauPartition(columns=columns),
            mps,
            meta=meta,
            columns=columns,
            label="read-plateau",
            divisions=divisions_lst,
            store=ds_factory.store_factory,
            categoricals=categoricals,
            predicate_pushdown_to_io=predicate_pushdown_to_io,
            dates_as_object=dates_as_object,
            predicates=predicates,
        )
    if dask_index_on:
        return ddf.set_index(dask_index_on, divisions=divisions_lst, sorted=True)
    else:
        return ddf


def _get_dask_meta_for_dataset(ds_factory, columns, categoricals, dates_as_object):
    """Calculate a schema suitable for the dask dataframe meta from the
    dataset."""
    table_schema = ds_factory.schema
    meta = empty_dataframe_from_schema(
        table_schema, columns=columns, date_as_object=dates_as_object
    )

    if categoricals:
        meta = meta.astype(dict.fromkeys(categoricals, "category"))
        meta = dd.utils.clear_known_categories(meta, categoricals)

    categoricals_from_index = _maybe_get_categoricals_from_index(
        ds_factory, categoricals
    )
    if categoricals_from_index:
        meta = meta.astype(categoricals_from_index)
    return meta


def _shuffle_docs(func):
    func.__doc__ += """

    .. admonition:: Behavior without ``shuffle==False``

        In the case without ``partition_on`` every dask partition is mapped to a single plateau partition

        In the case with ``partition_on`` every dask partition is mapped to N plateau partitions, where N
        depends on the content of the respective partition, such that every resulting plateau partition has
        only a single value in the respective ``partition_on`` columns.

    .. admonition:: Behavior with ``shuffle==True``

        ``partition_on`` is mandatory

        Perform a data shuffle to ensure that every primary key will have at most ``num_bucket``.

        .. note::
            The number of allowed buckets will have an impact on the required resources and runtime.
            Using a larger number of allowed buckets will usually reduce resource consumption and in some
            cases also improves runtime performance.

        :Example:

            >>> partition_on="primary_key"
            >>> num_buckets=2  # doctest: +SKIP
            primary_key=1/bucket1.parquet
            primary_key=1/bucket2.parquet

    .. note:: This can only be used for datasets with a single table!

    See also, :ref:`partitioning_dask`.

    Parameters
    ----------
    ddf: Union[dask.dataframe.DataFrame, None]
        The dask.Dataframe to be used to calculate the new partitions from. If this parameter is `None`, the update pipeline
        will only delete partitions without creating new ones.
    shuffle: bool
        If `True` and `partition_on` is requested, shuffle the data to reduce number of output partitions.

        See also, :ref:`shuffling`.

        .. warning::

            Dask uses a heuristic to determine how data is shuffled and there are two options, `partd` for local disk shuffling and `tasks` for distributed shuffling using a task graph. If there is no :class:`distributed.Client` in the context and the option is not set explicitly, dask will choose `partd` which may cause data loss when the graph is executed on a distributed cluster.

            Therefore, we recommend to specify the dask shuffle method explicitly, e.g. by using a context manager.

            .. code::

                with dask.config.set(shuffle='tasks'):
                    graph = update_dataset_from_ddf(...)
                graph.compute()

    repartition_ratio: Optional[Union[int, float]]
        If provided, repartition the dataframe before calculation starts to ``ceil(ddf.npartitions / repartition_ratio)``
    num_buckets: int
        If provided, the output partitioning will have ``num_buckets`` files per primary key partitioning.
        This effectively splits up the execution ``num_buckets`` times. Setting this parameter may be helpful when
        scaling.
        This only has an effect if ``shuffle==True``
    bucket_by:
        The subset of columns which should be considered for bucketing.

        This parameter ensures that groups of the given subset are never split
        across buckets within a given partition.

        Without specifying this the buckets will be created randomly.

        This only has an effect if ``shuffle==True``

        .. admonition:: Secondary indices

            This parameter has a strong effect on the performance of secondary
            indices. Since it guarantees that a given tuple of the subset will
            be entirely put into the same file you can build efficient indices
            with this approach.

        .. note::

            Only columns with data types which can be hashed are allowed to be used in this.
"""
    return func


def _id(x):
    return x


def _commit_update_from_reduction(df_mps, **kwargs):
    partitions = pd.Series(
        filter(
            lambda mp: mp is not np.nan and not mp.is_sentinel, df_mps.values.flatten()
        )
    ).dropna()
    return update_dataset_from_partitions(
        partition_list=partitions,
        **kwargs,
    )


def _commit_store_from_reduction(df_mps, **kwargs):
    partitions = pd.Series(
        filter(
            lambda mp: mp is not np.nan and not mp.is_sentinel, df_mps.values.flatten()
        )
    ).dropna()
    return store_dataset_from_partitions(
        partition_list=partitions,
        **kwargs,
    )


@default_docs
@_shuffle_docs
@normalize_args
def store_dataset_from_ddf(
    ddf: dd.DataFrame,
    store: StoreInput,
    dataset_uuid: str,
    table: str = SINGLE_TABLE,
    secondary_indices: list[str] | None = None,
    shuffle: bool = False,
    repartition_ratio: SupportsFloat | None = None,
    num_buckets: int = 1,
    sort_partitions_by: list[str] | str | None = None,
    metadata: Mapping | None = None,
    df_serializer: DataFrameSerializer | None = None,
    metadata_merger: Callable | None = None,
    metadata_version: int = DEFAULT_METADATA_VERSION,
    partition_on: list[str] | None = None,
    bucket_by: list[str] | str | None = None,
    overwrite: bool = False,
):
    """Store a dataset from a dask.dataframe."""
    # normalization done by normalize_args but mypy doesn't recognize this
    sort_partitions_by = cast(list[str], sort_partitions_by)
    secondary_indices = cast(list[str], secondary_indices)
    bucket_by = cast(list[str], bucket_by)
    partition_on = cast(list[str], partition_on)

    if table is None:
        raise TypeError("The parameter `table` is not optional.")

    ds_factory = _ensure_factory(dataset_uuid=dataset_uuid, store=store, factory=None)

    if not overwrite:
        raise_if_dataset_exists(dataset_uuid=dataset_uuid, store=store)

    with dask.config.set(
        {"dataframe.convert-string": False, "dataframe.shuffle.method": "tasks"}
    ):
        mp_ser = _write_dataframe_partitions(
            ddf=ddf,
            store=ds_factory.store_factory,
            dataset_uuid=dataset_uuid,
            table=table,
            secondary_indices=secondary_indices,
            shuffle=shuffle,
            repartition_ratio=repartition_ratio,
            num_buckets=num_buckets,
            sort_partitions_by=sort_partitions_by,
            df_serializer=df_serializer,
            metadata_version=metadata_version,
            partition_on=partition_on,
            bucket_by=bucket_by,
        )

    return mp_ser.reduction(
        chunk=_id,
        aggregate=_commit_store_from_reduction,
        split_every=False,
        token="commit-dataset",
        meta=object,
        aggregate_kwargs={
            "store": ds_factory.store_factory,
            "dataset_uuid": ds_factory.dataset_uuid,
            "dataset_metadata": metadata,
            "metadata_merger": metadata_merger,
        },
    )


def _write_dataframe_partitions(
    ddf: dd.DataFrame,
    store: StoreFactory,
    dataset_uuid: str,
    table: str,
    secondary_indices: list[str],
    shuffle: bool,
    repartition_ratio: SupportsFloat | None,
    num_buckets: int,
    sort_partitions_by: list[str],
    df_serializer: DataFrameSerializer | None,
    metadata_version: int,
    partition_on: list[str],
    bucket_by: list[str],
) -> dd.Series:
    if repartition_ratio and ddf is not None:
        ddf = ddf.repartition(
            npartitions=int(np.ceil(ddf.npartitions / repartition_ratio))
        )

    if ddf is None:
        mps = dd.from_pandas(
            pd.Series(
                [
                    parse_input_to_metapartition(
                        None,
                        metadata_version=metadata_version,
                        table_name=table,
                    )
                ]
            ),
            npartitions=1,
        )
    else:
        if shuffle:
            mps = shuffle_store_dask_partitions(
                ddf=ddf,
                table=table,
                secondary_indices=secondary_indices,
                metadata_version=metadata_version,
                partition_on=partition_on,
                store_factory=store,
                df_serializer=df_serializer,
                dataset_uuid=dataset_uuid,
                num_buckets=num_buckets,
                sort_partitions_by=sort_partitions_by,
                bucket_by=bucket_by,
            )
        else:
            mps = ddf.map_partitions(
                write_partition,
                secondary_indices=secondary_indices,
                metadata_version=metadata_version,
                partition_on=partition_on,
                store_factory=store,
                df_serializer=df_serializer,
                dataset_uuid=dataset_uuid,
                sort_partitions_by=sort_partitions_by,
                dataset_table_name=table,
                meta=(MetaPartition),
            )
    return mps


@default_docs
@_shuffle_docs
@normalize_args
def update_dataset_from_ddf(
    ddf: dd.DataFrame,
    store: StoreInput | None = None,
    dataset_uuid: str | None = None,
    table: str = SINGLE_TABLE,
    secondary_indices: list[str] | None = None,
    shuffle: bool = False,
    repartition_ratio: SupportsFloat | None = None,
    num_buckets: int = 1,
    sort_partitions_by: list[str] | str | None = None,
    delete_scope: Iterable[Mapping[str, str]] | None = None,
    metadata: Mapping | None = None,
    df_serializer: DataFrameSerializer | None = None,
    metadata_merger: Callable | None = None,
    default_metadata_version: int = DEFAULT_METADATA_VERSION,
    partition_on: list[str] | None = None,
    factory: DatasetFactory | None = None,
    bucket_by: list[str] | str | None = None,
):
    """Update a dataset from a dask.dataframe.

    See Also
    --------
    :ref:`mutating_datasets`
    """
    if table is None:
        raise TypeError("The parameter `table` is not optional.")

    # normalization done by normalize_args but mypy doesn't recognize this
    sort_partitions_by = cast(list[str], sort_partitions_by)
    secondary_indices = cast(list[str], secondary_indices)
    bucket_by = cast(list[str], bucket_by)
    partition_on = cast(list[str], partition_on)

    ds_factory, metadata_version, partition_on = validate_partition_keys(
        dataset_uuid=dataset_uuid,
        store=store,
        default_metadata_version=default_metadata_version,
        partition_on=partition_on,
        ds_factory=factory,
    )

    inferred_indices = _ensure_compatible_indices(ds_factory, secondary_indices)
    del secondary_indices

    if ds_factory:
        store_factory: StoreFactory = ds_factory.store_factory
    else:
        if not callable(store):
            raise TypeError(
                "You must either pass in a DatasetFactory or a StoreFactor via store."
            )
        store_factory = store

    with dask.config.set(
        {"dataframe.convert-string": False, "dataframe.shuffle.method": "tasks"}
    ):
        mp_ser = _write_dataframe_partitions(
            ddf=ddf,
            store=store_factory,
            dataset_uuid=dataset_uuid or ds_factory.dataset_uuid,
            table=table,
            secondary_indices=inferred_indices,
            shuffle=shuffle,
            repartition_ratio=repartition_ratio,
            num_buckets=num_buckets,
            sort_partitions_by=sort_partitions_by,
            df_serializer=df_serializer,
            metadata_version=metadata_version,
            partition_on=cast(list[str], partition_on),
            bucket_by=bucket_by,
        )

    final = mp_ser.reduction(
        chunk=_id,
        aggregate=_commit_update_from_reduction,
        split_every=False,
        token="commit-dataset",
        meta=object,
        aggregate_kwargs={
            "store_factory": store,
            "dataset_uuid": dataset_uuid,
            "ds_factory": ds_factory,
            "delete_scope": delete_scope,
            "metadata": metadata,
            "metadata_merger": metadata_merger,
        },
    )
    return final


@default_docs
@normalize_args
def collect_dataset_metadata(
    store: StoreInput | None = None,
    dataset_uuid: str | None = None,
    predicates: PredicatesType | None = None,
    frac: float = 1.0,
    factory: DatasetFactory | None = None,
) -> dd.DataFrame:
    """Collect parquet metadata of the dataset. The `frac` parameter can be
    used to select a subset of the data.

    .. warning::
      If the size of the partitions is not evenly distributed, e.g. some partitions might be larger than others,
      the metadata returned is not a good approximation for the whole dataset metadata.
    .. warning::
      Using the `frac` parameter is not encouraged for a small number of total partitions.


    Parameters
    ----------
    predicates
      plateau predicates to apply filters on the data for which to gather statistics

      .. warning::
          Filtering will only be applied for predicates on indices.
          The evaluation of the predicates therefore will therefore only return an approximate result.

    frac
      Fraction of the total number of partitions to use for gathering statistics. `frac == 1.0` will use all partitions.

    Returns
    -------
    dask.dataframe.DataFrame:
        A dask.DataFrame containing the following information about dataset statistics:
        * `partition_label`: File name of the parquet file, unique to each physical partition.
        * `row_group_id`: Index of the row groups within one parquet file.
        * `row_group_compressed_size`: Byte size of the data within one row group.
        * `row_group_uncompressed_size`: Byte size (uncompressed) of the data within one row group.
        * `number_rows_total`: Total number of rows in one parquet file.
        * `number_row_groups`: Number of row groups in one parquet file.
        * `serialized_size`: Serialized size of the parquet file.
        * `number_rows_per_row_group`: Number of rows per row group.

    Raises
    ------
    ValueError
      If no metadata could be retrieved, raise an error.
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(
            f"Invalid value for parameter `frac`: {frac}."
            "Please make sure to provide a value larger than 0.0 and smaller than or equal to 1.0 ."
        )
    dataset_factory = _ensure_factory(
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
    )

    mps = list(
        dispatch_metapartitions_from_factory(dataset_factory, predicates=predicates)
    )

    with dask.config.set({"dataframe.convert-string": False}):
        if mps:
            random.shuffle(mps)
            # ensure that even with sampling at least one metapartition is returned
            cutoff_index = max(1, int(len(mps) * frac))
            mps = mps[:cutoff_index]
            ddf = dd.from_map(
                MetaPartition.get_parquet_metadata,
                mps,
                meta=_METADATA_SCHEMA,
                store=dataset_factory.store_factory,
            )
        else:
            df = pd.DataFrame(columns=_METADATA_SCHEMA.keys())
            df = df.astype(_METADATA_SCHEMA)
            ddf = dd.from_pandas(df, npartitions=1)

    return ddf


def _unpack_hash(df, group_key, unpack_meta, subset):
    df = unpack_payload_pandas(df, unpack_meta).set_index(group_key, drop=True)
    return df.groupby(df.index).apply(_hash_partition, subset=subset)


def _hash_partition(part, subset):
    return pd.util.hash_pandas_object(part.reset_index()[subset], index=False).sum()


@default_docs
@normalize_args
def hash_dataset(
    store: StoreInput | None = None,
    dataset_uuid: str | None = None,
    subset=None,
    group_key=None,
    table: str = SINGLE_TABLE,
    predicates: PredicatesType | None = None,
    factory: DatasetFactory | None = None,
) -> dd.Series:
    """Calculate a partition wise, or group wise, hash of the dataset.

    .. note::

        We do not guarantee the hash values to remain constant across versions.


    Example output::

        Assuming a dataset with two unique values in column `P` this gives

        >>> hash_dataset(factory=dataset_with_index_factory,group_key=["P"]).compute()
        ... P
        ... 1    11462879952839863487
        ... 2    12568779102514529673
        ... dtype: uint64

    Parameters
    ----------
    subset
        If provided, only take these columns into account when hashing the dataset
    group_key
        If provided, calculate hash per group instead of per partition
    """
    dataset_factory = _ensure_factory(
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
    )

    columns = subset
    if subset and group_key:
        columns = sorted(set(subset) | set(group_key))
    ddf = read_dataset_as_ddf(
        table=table,
        predicates=predicates,
        factory=dataset_factory,
        columns=columns,
        dates_as_object=True,
    )
    with dask.config.set(
        {"dataframe.convert-string": False, "dataframe.shuffle.method": "tasks"}
    ):
        subset = subset or ddf.columns.to_list()
        if not group_key:
            return ddf.map_partitions(
                _hash_partition, subset=subset, meta=(None, "uint64")
            )
        else:
            ddf2 = pack_payload(ddf, group_key=group_key)
            final = ddf2.shuffle(on=group_key).map_partitions(
                _unpack_hash,
                group_key=group_key,
                unpack_meta=ddf._meta,
                subset=subset,
                meta=(None, "uint64"),
            )

            return final
