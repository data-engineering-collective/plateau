=========
Changelog
=========

Plateau 4.5.0 (2025-07-24)
==========================

* Include pyarrow 17.0.0, 18.0.1, 19.0.1 and 20.0.0 in CI
* Support numpy 2.x
* Support pandas 3.x
* Compatibility with dask 2025.1.1
* Fixes for Python 3.13 support

Plateau 4.4.1 (2024-12-17)
==========================

* Port project to use the OSS QuantCo copier template (`copier template https://github.com/Quantco/copier-template-python-open-source`_) and (`pixi https://pixi.sh`_) as environment manager.


Plateau 4.4.0 (2024-05-23)
==========================

* Improve support for Dask's query planner.
* Support column projection when using ``dask-expr``'s ``from_map``.
* Include pyarrow 15.0.0 in CI
* Fix tests for dask 2024.1.1
* Account for breaking changes in nightly builds of Pandas which would have caused issues upon the release of Pandas 3.


Plateau 4.3.0 (2023-12-21)
==========================

* Add ``is distinct from`` predicate operation. This follows the behaviour of the SQL operation of the same name, i.e. it treats two null values as the same. This is also the current behaviour of the ``!=`` operation.
* Deprecates the current behaviour of the ``!=`` predicate operation. From 5.0.0 onwards, the new behaviour will be the same as its SQL counterpart, i.e. it will filter out null/missing values. Where the current behaviour of ``!=`` is needed, ``is distinct from`` should be used instead.
* Include pyarrow 14.0.1 in CI


Plateau 4.2.2 (2023-11-06)
==========================

* Include Python 3.12 in CI
* Migrate setup.cfg to pyproject.toml completely
* Port to ruff, remove isort, black, and flake8


Plateau 4.2.1 (2023-10-31)
==========================

* Add support for pandas 2.1
* Fix a bug to do with timestamp dtype conversion
* Add timestamp unit coercion as Plateau currently only supports nanosecond units on timestamps

Plateau 4.2.0 (2023-10-10)
==========================

* Support pandas 2
* Test pyarrow 12 and 13
* Prevent dask from casting all object dtypes to strings
* Remove tests for pyarrow<=3 as they fail with pandas>=2
* Adds support for dask 2023.9.2

Plateau 4.1.5 (2023-03-14)
==========================

* Enable dask column projection.
* Test pyarrow 11
* Remove usage of ``pkg_resources``.

Plateau 4.1.4 (2022-12-14)
==========================

* We no longer package the ``tests`` directory.
* Test pyarrow 10 and Python 3.11, pandas 1.5

Plateau 4.1.3 (2022-10-24)
==========================

* Patch to load partition with only nulls as categorical (#55)

Plateau 4.1.2 (2022-10-20)
==========================

* Removed upper-bound pin in pyarrow version dependency

Plateau 4.1.1 (2022-10-19)
==========================

* Fix usage of ``setuptools_scm``

Plateau 4.1.0 (2022-10-19)
==========================

* Support for ``pyarrow`` 5, 6, 7, 8, 9
* Support for ``numpy`` 1.23
* Support for ``dask>=2022.4.2``
* Support Python 3.8-3.10
* Replaced ``simplekv`` and ``storefact`` usages with ``minimalkv``

Plateau 4.0.4 (2022-03-17)
==========================

* Fork and rename the project as `plateau` (flat files, flat land).
* Fixed a bug where sometimes data partitions in a `plateau` dataset would be read in a non-deterministic order among Python executions (#23).

Kartothek 4.0.3 (2021-06-10)
============================

* Pin dask to not use 2021.5.1 and 2020.6.0 (#475)

Kartothek 4.0.2 (2021-06-07)
============================

* Fix a bug in ``MetaPartition._reconstruct_index_columns`` that would raise an ``IndexError`` when loading few columns of a dataset with many primary indices.


Kartothek 4.0.1 (2021-04-13)
============================

* Fixed dataset corruption after updates when table names other than "table" are used (#445).


Kartothek 4.0.0 (2021-03-17)
============================

This is a major release of kartothek with breaking API changes.

* Removal of complex user input (see gh427)
* Removal of multi table feature
* Removal of `kartothek.io.merge` module
* class ``kartothek.core.dataset.DatasetMetadata`` now has an attribute called `schema` which replaces the previous attribute `table_meta` and returns only a single schema
* All outputs which previously returned a sequence of dictionaries where each key-value pair would correspond to a table-data pair now returns only one :class:`pandas.DataFrame`
* All read pipelines will now automatically infer the table to read such that it is no longer necessary to provide `table` or `table_name` as an input argument
* All writing pipelines which previously supported a complex user input type now expose an argument `table_name` which can be used to continue usage of legacy datasets (i.e. datasets with an intrinsic, non-trivial table name). This usage is discouraged and we recommend users to migrate to a default table name (i.e. leave it None / `table`)
* All pipelines which previously accepted an argument `tables` to select the subset of tables to load no longer accept this keyword. Instead the to-be-loaded table will be inferred
* Trying to read a multi-tabled dataset will now cause an exception telling users that this is no longer supported with kartothek 4.0
* The dict schema for ``kartothek.core.dataset.DatasetMetadataBase.to_dict`` and ``kartothek.core.dataset.DatasetMetadata.from_dict`` changed replacing a dictionary in `table_meta` with the simple `schema`
* All pipeline arguments which previously accepted a dictionary of sequences to describe a table specific subset of columns now accept plain sequences (e.g. `columns`, `categoricals`)
* Remove the following list of deprecated arguments for io pipelines
  * label_filter
  * central_partition_metadata
  * load_dynamic_metadata
  * load_dataset_metadata
  * concat_partitions_on_primary_index
* Remove `output_dataset_uuid` and `df_serializer` from ``kartothek.io.eager.commit_dataset`` since these arguments didn't have any effect
* Remove `metadata`, `df_serializer`, `overwrite`, `metadata_merger` from ``kartothek.io.eager.write_single_partition``
* ``kartothek.io.eager.store_dataframes_as_dataset`` now requires a list as an input
* Default value for argument `date_as_object` is now universally set to ``True``. The behaviour for `False` will be deprecated and removed in the next major release
* No longer allow to pass `delete_scope` as a delayed object to ``kartothek.io.dask.dataframe.update_dataset_from_ddf``
* ``kartothek.io.dask.dataframe.update_dataset_from_ddf`` and :func:``kartothek.io.dask.dataframe.store_dataset_from_ddf`` now return a `dd.core.Scalar` object. This enables all `dask.DataFrame` graph optimizations by default.
* Remove argument `table_name` from ``kartothek.io.dask.dataframe.collect_dataset_metadata``


Version 3.20.0 (2021-03-15)
===========================

This will be the final release in the 3.X series. Please ensure your existing
codebase does not raise any DeprecationWarning from kartothek and migrate your
import paths ahead of time to the new ``kartothek.api`` modules to ensure a
smooth migration to 4.X.

* Introduce ``kartothek.api`` as the public definition of the API. See also :doc:`versioning`.
* Introduce `DatasetMetadataBase.schema` to prepare deprecation of `table_meta`
* ``kartothek.io.eager.read_dataset_as_dataframes`` and
  ``kartothek.io.iter.read_dataset_as_dataframes__iterator`` now correctly return
  categoricals as requested for misaligned categories.


Version 3.19.1 (2021-02-24)
===========================

* Allow ``pyarrow==3`` as a dependency.
* Fix a bug in ``kartothek.io_components.utils.align_categories`` for dataframes
  with missings and of non-categorical dtype.
* Fix an issue with the cube index validation introduced in v3.19.0 (#413).


Version 3.19.0 (2021-02-12)
===========================

* Fix an issue where updates on cubes or updates on datasets using
  dask.dataframe might not update all secondary indices, resulting in a corrupt
  state after the update
* Expose compression type and row group chunk size in Cube interface via optional
  parameter of type ``kartothek.serialization.ParquetSerializer``.
* Add retries to ``kartothek.serialization._parquet.ParquetSerializer.restore_dataframe``
  IOErrors on long running ktk + dask tasks have been observed. Until the root cause is fixed,
  the serialization is retried to gain more stability.

Version 3.18.0 (2021-01-25)
===========================

* Add ``cube.suppress_index_on`` to switch off the default index creation for dimension columns
* Fixed the import issue of zstd module for ``kartothek.core _zmsgpack``.
* Fix a bug in ``kartothek.io_components.read.dispatch_metapartitions_from_factory`` where
  `dispatch_by=[]` would be treated like `dispatch_by=None`, not merging all dataset partitions into
  a single partitions.

Version 3.17.3 (2020-12-04)
===========================

* Allow ``pyarrow==2`` as a dependency.

Version 3.17.2 (2020-12-01)
===========================

* #378 Improve logging information for potential buffer serialization errors


Version 3.17.1 (2020-11-24)
===========================

Bugfixes
^^^^^^^^

* Fix GitHub #375 by loosening checks of the supplied store argument


Version 3.17.0 (2020-11-23)
===========================

Improvements
^^^^^^^^^^^^
* Improve performance for "in" predicate literals using long object lists as values
* ``kartothek.io.eager.commit_dataset`` now allows to modify the user
  metadata without adding new data.

Bugfixes
^^^^^^^^
* Fix an issue where ``kartothek.io.dask.dataframe.collect_dataset_metadata`` would return
  improper rowgroup statistics
* Fix an issue where ``kartothek.io.dask.dataframe.collect_dataset_metadata`` would execute
  ``get_parquet_metadata`` at graph construction time
* Fix a bug in ``kartothek.io.eager_cube.remove_partitions`` where all partitions were removed
  instead of non at all.
* Fix a bug in ``kartothek.core.dataset.DatasetMetadataBase.get_indices_as_dataframe`` which would
  raise an ``IndexError`` if indices were empty or had not been loaded

Version 3.16.0 (2020-09-29)
===========================

New functionality
^^^^^^^^^^^^^^^^^
* Allow filtering of nans using "==", "!=" and "in" operators

Bugfixes
^^^^^^^^
* Fix a regression which would not allow the usage of non serializable stores even when using factories


Version 3.15.1 (2020-09-28)
===========================
* Fix a packaging issue where `typing_extensions` was not properly specified as
  a requirement for python versions below 3.8

Version 3.15.0 (2020-09-28)
===========================

New functionality
^^^^^^^^^^^^^^^^^
* Add ``kartothek.io.dask.dataframe.store_dataset_from_ddf`` to offer write
  support of a dask dataframe without update support. This forbids or explicitly
  allows overwrites and does not update existing datasets.
* The ``sort_partitions_by`` feature now supports multiple columns. While this
  has only marginal effect for predicate pushdown, it may be used to improve the
  parquet compression.
* ``build_cube_from_dataframe`` now supports the ``shuffle`` methods offered by
  ``kartothek.io.dask.dataframe.store_dataset_from_ddf`` and
  ``kartothek.io.dask.dataframe.update_dataset_from_ddf`` but writes the
  output in the cube format

Improvements
^^^^^^^^^^^^
* Reduce memory consumption during index write.
* Allow `simplekv` stores and `storefact` URLs to be passed explicitly as input for the `store` arguments

Version 3.14.0 (2020-08-27)
===========================

New functionality
^^^^^^^^^^^^^^^^^
* Add ``hash_dataset`` functionality

Improvements
^^^^^^^^^^^^

* Expand ``pandas`` version pin to include 1.1.X
* Expand ``pyarrow`` version pin to include 1.x
* Large addition to documentation for multi dataset handling (Kartothek Cubes)

Version 3.13.1 (2020-08-04)
===========================

* Fix evaluation of "OR"-connected predicates (#295)

Version 3.13.0 (2020-07-30)
===========================

Improvements
^^^^^^^^^^^^

* Update timestamp related code into Ktk Discover Cube functionality.
* Support backward compatibility to old cubes and fix for cli entry point.

Version 3.12.0 (2020-07-23)
===========================

New functionality
^^^^^^^^^^^^^^^^^

* Introduction of ``cube`` Functionality which is made with multiple Kartothek datasets.
* Basic Features - Extend, Query, Remove(Partitions),
  Delete (can delete entire datasets/cube), API, CLI, Core and IO features.
* Advanced Features - Multi-Dataset with Single Table, Explicit physical Partitions, Seed based join system.


Version 3.11.0 (2020-07-15)
===========================

New functionality
^^^^^^^^^^^^^^^^^

* Add ``kartothek.io_components.metapartition.MetaPartition.get_parquet_metadata`` and ``kartothek.io.dask.dataframe.collect_dataset_metadata``, enabling users to collect information about the Parquet metadata of a dataset (#306)

Bug fixes
^^^^^^^^^

* Performance of dataset update with ``delete_scope`` significantly improved for datasets with many partitions (#308)


Version 3.10.0 (2020-07-02)
===========================

Improvements
^^^^^^^^^^^^
* Dispatch performance improved for large datasets including metadata
* Introduction of ``dispatch_metadata`` kwarg to metapartitions read pipelines
  to allow for transition for future breaking release.

Bug fixes
^^^^^^^^^

* Ensure that the empty (sentinel) DataFrame used in `kartothek.io.eager.read_table``
  also has the correct behaviour when using the ``categoricals`` argument.


Breaking changes in ``io_components.read``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* The ``dispatch_metapartitions`` and ``dispatch_metapartitions_from_factory``
  will no longer attach index and metadata information to the created MP
  instances, unless explicitly requested.


Version 3.9.0 (2020-06-03)
==========================

Improvements
^^^^^^^^^^^^
* Arrow 0.17.X support
* Significant performance improvements for shuffle operations in
  ``kartothek.io.dask.dataframe.update_dataset_from_ddf``
  for large dask.DataFrames with many payload columns by using in-memory
  compression during the shuffle operation.
* Allow calling ``kartothek.io.dask.dataframe.update_dataset_from_ddf``
  without `partition_on` when `shuffle=True`.
* ``kartothek.io.dask.dataframe.read_dataset_as_ddf`` supports kwarg ``dispatch_by``
  to control the internal partitioning structure when creating a dataframe.
* ``kartothek.io.dask.dataframe.read_dataset_as_ddf`` and ``kartothek.io.dask.dataframe.update_dataset_from_ddf``
  now allow the keyword ``table`` to be optional, using the default SINGLE_TABLE identifier.
  (recommended since the multi table dataset support is in sunset).


Version 3.8.2 (2020-04-09)
==========================

Improvements
^^^^^^^^^^^^

* Read performance improved for, especially for partitioned datasets and queries with empty payload columns.

Bug fixes
^^^^^^^^^
* GH262: Raise an exception when trying to partition on a column with null values to prevent silent data loss
* Fix multiple index creation issues (cutting data, crashing) for ``uint`` data
* Fix index update issues for some types resulting in ``TypeError: Trying to update an index with different types...``
  messages.
* Fix issues where index creation with empty partitions can lead to ``ValueError: Trying to create non-typesafe index``


Version 3.8.1 (2020-03-20)
==========================

Improvements
^^^^^^^^^^^^

* Only fix column odering when restoring ``DataFrame`` if the ordering is incorrect.

Bug fixes
^^^^^^^^^
* GH248 Fix an issue causing a ValueError to be raised when using `dask_index_on` on non-integer columns
* GH255 Fix an issue causing the python interpreter to shut down when reading an
  empty file (see also https://issues.apache.org/jira/browse/ARROW-8142)

Version 3.8.0 (2020-03-12)
==========================

Improvements
^^^^^^^^^^^^

* Add keyword argument `dask_index_on` which reconstructs a dask index from an kartothek index when loading the dataset
* Add method ``kartothek.core.index.IndexBase.observed_values`` which returns an array of all observed values of the index column
* Updated and improved documentation w.r.t. guides and API documentation

Bug fixes
^^^^^^^^^
* GH227 Fix a Type error when loading categorical data in dask without
  specifying it explicitly
* No longer trigger the SettingWithCopyWarning when using bucketing
* GH228 Fix an issue where empty header creation from a pyarrow schema would not
  normalize the schema which causes schema violations during update.
* Fix an issue where ``kartothek.io.eager.create_empty_dataset_header``
  would not accept a store factory.


Version 3.7.0 (2020-02-12)
==========================

Improvements
^^^^^^^^^^^^

* Support for pyarrow 0.16.0
* Decrease scheduling overhead for dask based pipelines
* Performance improvements for categorical data when using pyarrow>=0.15.0
* Dask is now able to calculate better size estimates for the following classes:
    * ``kartothek.core.dataset.DatasetMetadata``
    * ``kartothek.core.factory.DatasetFactory``
    * ``kartothek.io_components.metapartition.MetaPartition``
    * ``kartothek.core.index.ExplicitSecondaryIndex``
    * ``kartothek.core.index.PartitionIndex``
    * ``kartothek.core.partition.Partition``
    * ``kartothek.core.common_metadata.SchemaWrapper``


Version 3.6.2 (2019-12-17)
==========================

Improvements
^^^^^^^^^^^^

* Add more explicit typing to ``kartothek.io.eager``.

Bug fixes
^^^^^^^^^
* Fix an issue where ``kartothek.io.dask.dataframe.update_dataset_from_ddf`` would create a column named "_KTK_HASH_BUCKET" in the dataset


Version 3.6.1 (2019-12-11)
==========================

Bug fixes
^^^^^^^^^
* Fix a regression introduced in 3.5.0 where predicates which allow multiple
  values for a field would generate duplicates

Version 3.6.0 (2019-12-03)
==========================

New functionality
^^^^^^^^^^^^^^^^^
- The partition on shuffle algorithm in ``kartothek.io.dask.dataframe.update_dataset_from_ddf`` now supports
  producing deterministic buckets based on hashed input data.

Bug fixes
^^^^^^^^^
- Fix addition of bogus index columns to Parquet files when using `sort_partitions_by`.
- Fix bug where ``partition_on`` in write path drops empty DataFrames and can lead to datasets without tables.


Version 3.5.1 (2019-10-25)
==========================
- Fix potential ``pyarrow.lib.ArrowNotImplementedError`` when trying to store or pickle empty
  ``kartothek.core.index.ExplicitSecondaryIndex`` objects
- Fix pickling of ``kartothek.core.index.ExplicitSecondaryIndex`` unloaded in
  `dispatch_metapartitions_from_factory`


Version 3.5.0 (2019-10-21)
==========================

New functionality
^^^^^^^^^^^^^^^^^
- Add support for pyarrow 0.15.0
- Additional functions in ``kartothek.serialization`` module for dealing with predicates
  * ``kartothek.serialization.check_predicates``
  * ``kartothek.serialization.filter_predicates_by_column``
  * ``kartothek.serialization.columns_in_predicates``
- Added available types for type annotation when dealing with predicates
  * ``kartothek.serialization.PredicatesType``
  * ``kartothek.serialization.ConjunctionType``
  * ``kartothek.serialization.LiteralType``
- Make ``kartothek.io.*read_table*`` methods use default table name if unspecified
- ``MetaPartition.parse_input_to_metapartition`` accepts dicts and list of tuples equivalents as ``obj`` input
- Added `secondary_indices` as a default argument to the `write` pipelines

Bug fixes
^^^^^^^^^
- Input to ``normalize_args`` is properly normalized to ``list``
- ``MetaPartition.load_dataframes`` now raises if table in ``columns`` argument doesn't exist
- require ``urlquote>=1.1.0`` (where ``urlquote.quoting`` was introduced)
- Improve performance for some cases where predicates are used with the `in` operator.
- Correctly preserve :class:``kartothek.core.index.ExplicitSecondaryIndex`` dtype when index is empty
- Fixed DeprecationWarning in pandas ``CategoricalDtype``
- Fixed broken docstring for `store_dataframes_as_dataset`
- Internal operations no longer perform schema validations. This will improve
  performance for batched partition operations (e.g. `partition_on`) but will
  defer the validation in case of inconsistencies to the final commit. Exception
  messages will be less verbose in these cases as before.
- Fix an issue where an empty dataframe of a partition in a multi-table dataset
  would raise a schema validation exception
- Fix an issue where the `dispatch_by` keyword would disable partition pruning
- Creating dataset with non existing columns as explicit index to raise a ValueError

Breaking changes
^^^^^^^^^^^^^^^^
- Remove support for pyarrow < 0.13.0
- Move the docs module from `io_components` to `core`


Version 3.4.0 (2019-09-17)
==========================
- Add support for pyarrow 0.14.1
- Use urlquote for faster quoting/unquoting


Version 3.3.0 (2019-08-15)
==========================
- Fix rejection of bool predicates in ``kartothek.serialization.filter_array_like`` when bool columns contains
  ``None``
- Streamline behavior of `store_dataset_from_ddf` when passing empty ddf.
- Fix an issue where a segmentation fault may be raised when comparing MetaPartition instances
- Expose a ``date_as_object`` flag in ``kartothek.core.index.as_flat_series``


Version 3.2.0 (2019-07-25)
==========================
- Fix gh:66 where predicate pushdown may evaluate false results if evaluated
  using improper types. The behavior now is to raise in these situations.
- Predicate pushdown and ``kartothek.serialization.filter_array_like`` will now properly handle pandas Categoricals.
- Add ``kartothek.io.dask.bag.read_dataset_as_dataframe_bag``
- Add ``kartothek.io.dask.bag.read_dataset_as_metapartitions_bag``


Version 3.1.1 (2019-07-12)
==========================

- make ``kartothek.io.dask.bag.build_dataset_indices__bag`` more efficient
- make ``kartothek.io.eager.build_dataset_indices`` more efficient
- fix pseudo-private ``kartothek.io_components.read.dispatch_metapartitions`` handling of
  ``concat_partitions_on_primary_index``
- fix internal errors if querying (e.g. via ``kartothek.io.eager.read_dataset_as_dataframes``) with
  ``datetime.date`` predicates that use the dataset index; this affects all code paths using
  ``kartothek.io_components.metapartition.MetaPartition.load_dataframes``


Version 3.1.0 (2019-07-10)
==========================

- fix ``getargspec`` ``DeprecationWarning``
- fix ``FutureWarning`` in ``filter_array_like``
- remove ``funcsigs`` requirement
- Implement reference ``io.eager`` implementation, adding the functions:

    - ``kartothek.io.eager.garbage_collect_dataset``
    - ``kartothek.io.eager.build_dataset_indices``
    - ``kartothek.io.eager.update_dataset_from_dataframes``

- fix ``_apply_partition_key_predicates`` ``FutureWarning``
- serialize ``kartothek.core.index.ExplicitSecondaryIndex`` to parquet
- improve messages for schema violation errors
- Ensure binary column names are read as type ``str``:

    - Ensure dataframe columns are of type ``str`` in ``kartothek.core.common_metadata.empty_dataframe_from_schema``
    - Testing: create ``kartothek.io.testing.read.test_binary_column_metadata`` which checks column names stored as
      ``bytes`` objects are read as type ``str``

- fix issue where it was possible to add an index to an existing dataset by using update functions and partition indices
  (https://github.com/JDASoftwareGroup/kartothek/issues/16).

- fix issue where unreferenced files were not being removed when deleting an entire dataset

- support nested ``kartothek.io_components.metapartition.MetaPartition``
  in ``kartothek.io_components.metapartition.MetaPartition.add_metapartition``.
  This fixes issue https://github.com/JDASoftwareGroup/kartothek/issues/40 .

- Add ``kartothek.io.dask.bag.build_dataset_indices__bag``

- Return `dask.bag.Item` object from ``kartothek.io.dask.bag.store_bag_as_dataset`` to avoid misoptimization

**Breaking:**

- categorical normalization was moved from ``kartothek.core.common_metadata.make_meta`` to
  ``kartothek.core.common_metadata.normalize_type``.
- ``kartothek.core.common_metadata.SchemaWrapper.origin`` is now a set of of strings instead of a single string
- ``Partition.from_v2_dict`` was removed, use ``kartothek.core.partition.Partition.from_dict`` instead


Version 3.0.0 (2019-05-02)
==========================

- Initial public release
