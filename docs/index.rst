================================================
plateau - manage tabular data in object stores
================================================

:Release: |release|
:Date: |today|


plateau is a Python library to manage (create, read, update, delete) large
amounts of tabular data in a blob store. It stores data as datasets, which
it presents as pandas DataFrames to the user. Datasets are a collection of
files with the same schema that reside in a blob store. plateau uses a
metadata definition to handle these datasets efficiently. For distributed
access and manipulation of datasets, plateau offers a `Dask
<https://dask.org>`_ interface (:mod:`plateau.io.dask`).

Storing data distributed over multiple files in a blob store (S3, ABS, GCS,
etc.) allows for a fast, cost-efficient and highly scalable data
infrastructure. A downside of storing data solely in an object store is that the
storages themselves give little to no guarantees beyond the consistency of a
single file. In particular, they cannot guarantee the consistency of your
dataset. If we demand a consistent state of our dataset at all times, we need to
track the state of the dataset. plateau frees us from having to do this
manually.

The :mod:`plateau.io` module provides building blocks to create and modify
these datasets in data pipelines. plateau handles I/O, tracks dataset
partitions and selects subsets of data transparently.

To get started, have a look at our :doc:`guide/getting_started` guide,
head to the description of the :doc:`spec/format_specification` or head straight to the API documentation :doc:`api`.

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   Getting Started <guide/getting_started>
   Partitioning <guide/partitioning>
   Mutating Datasets <guide/mutating_datasets>
   Dask indexing <guide/dask_indexing>
   Examples <guide/examples>


.. toctree::
   :maxdepth: 2
   :caption: Background
   :hidden:

   Specification <spec/format_specification>
   Type System <spec/type_system>
   DataFrame Serialization <spec/serialization>
   KeyValueStore Interface <spec/store_interface>
   Storage Layout <spec/storage_layout>
   Indexing <spec/indexing>
   Efficient Querying <spec/efficient_querying>
   Parallel Execution with Dask <spec/parallel_dask>


.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   Module Reference <_rst/modules>
   Versioning <versioning>
   Changelog <changes>
