.. _mutating_datasets:

Mutating Datasets
=================

It's possible to update existing data
by adding new physical partitions to them and deleting or replacing old partitions. plateau
provides update functions that generally have the prefix `update_dataset` in their names.
For example, :func:`~plateau.io.eager.update_dataset_from_dataframes` is the update
function for the ``eager`` backend.

To see updating in action, let's first set up a storage location and store
some data there with plateau.

.. ipython:: python
    :suppress:

    import numpy as np
    import pandas as pd
    from functools import partial
    from tempfile import TemporaryDirectory
    from minimalkv import get_store_from_url

    from plateau.api.dataset import store_dataframes_as_dataset

    dataset_dir = TemporaryDirectory()

    store_url = f"hfs://{dataset_dir.name}"

    df = pd.DataFrame(
        {
            "A": 1.0,
            "B": [
                pd.Timestamp("20130102"),
                pd.Timestamp("20130102"),
                pd.Timestamp("20130103"),
                pd.Timestamp("20130103"),
            ],
            "C": pd.Series(1, index=list(range(4)), dtype="float32"),
            "D": np.array([3] * 4, dtype="int32"),
            "E": pd.Categorical(["test", "train", "test", "train"]),
            "F": "foo",
        }
    )

.. ipython:: python
    :okwarning:

    dm = store_dataframes_as_dataset(
        store=store_url, dataset_uuid="partitioned_dataset", dfs=[df], partition_on="B"
    )
    sorted(dm.partitions.keys())


Appending Data
--------------

Now, we create ``another_df`` with the same schema as our initial dataframe
``df`` and update it using the ``eager`` backend by calling :func:`~plateau.io.eager.update_dataset_from_dataframes`:

.. ipython:: python
    :okwarning:

    from plateau.api.dataset import update_dataset_from_dataframes

    another_df = pd.DataFrame(
        {
            "A": 5.0,
            "B": [
                pd.Timestamp("20110103"),
                pd.Timestamp("20110103"),
                pd.Timestamp("20110104"),
                pd.Timestamp("20110104"),
            ],
            "C": pd.Series(2, index=list(range(4)), dtype="float32"),
            "D": np.array([6] * 4, dtype="int32"),
            "E": pd.Categorical(["prod", "dev", "prod", "dev"]),
            "F": "bar",
        }
    )

    dm = update_dataset_from_dataframes([another_df], store=store_url, dataset_uuid=dm.uuid)
    sorted(dm.partitions.keys())


Looking at ``dm.partitions``, we can see that another partition has
been added.

If we read the data again, we can see that the ``another_df`` has been appended to the
previous contents.

.. ipython:: python
    :okwarning:

    from plateau.api.dataset import read_table

    updated_df = read_table(dataset_uuid=dm.uuid, store=store_url)
    updated_df


The way dataset updates work is that new partitions are added to a dataset
as long as they have the same tables as the existing partitions.

Deleting Data
-------------

Adding data to an existing dataset is not the only functionality achievable within an update
operation, and it can also be used to remove data.
To do so we use the ``delete_scope`` keyword argument as shown in the example below:

.. ipython:: python
    :okwarning:

    dm = update_dataset_from_dataframes(
        None,
        store=store_url,
        dataset_uuid=dm.uuid,
        partition_on="B",
        delete_scope=[{"B": pd.Timestamp("20130102")}],
    )
    sorted(dm.partitions.keys())


As we can see, we specified using a dictionary that data where the column ``B`` has the
value ``pd.Timestamp("20130102")`` should be removed. Looking at the partitions after the update, we see that
the partition ``B=2013-01-02[...]`` has in fact been removed.

.. warning:: We defined ``delete_scope`` over a value of ``B``, which is the column that
    we partitioned on: ``delete_scope`` *only works on* partitioned columns.

    Thus, ``delete_scope`` *should only* be used on partitioned columns due to their one-to-one mapping;
    without the guarantee of one-to-one mappings, using ``delete_scope`` could have unwanted
    effects like accidentally removing data with different values.

    Attempting to use ``delete_scope`` *will also* work on datasets not previously partitioned
    on any column(s); however this is **not at all advised** since the effect will simply be to
    remove **all** previous partitions and replace them with the ones in the update.

    If the intention of the user is to delete the entire dataset, using :func:`plateau.io.eager.delete_dataset`
    would be a much better, cleaner and safer way to go about doing so.


When  using ``delete_scope``, multiple values for the same column cannot be defined as a
list but have to be specified instead as individual dictionaries, i.e.
``[{"E": ["test", "train"]}]`` will not work but ``[{"E": "test"}, {"E": "train"}]`` will.

.. ipython:: python
    :okwarning:

    duplicate_df = df.copy()
    duplicate_df.F = "bar"

    dm = store_dataframes_as_dataset(
        store_url,
        "another_partitioned_dataset",
        [df, duplicate_df],
        partition_on=["E", "F"],
    )
    sorted(dm.partitions.keys())


.. ipython:: python
    :okwarning:

    dm = update_dataset_from_dataframes(
        None,
        store=store_url,
        dataset_uuid=dm.uuid,
        partition_on=["E", "F"],
        delete_scope=[{"E": "train", "F": "foo"}, {"E": "test", "F": "bar"}],
    )

    sorted(dm.partitions.keys())  # `E=train/F=foo` and `E=test/F=bar` are deleted


Replacing Data
--------------

Finally, an update step can be used to perform the two steps above, i.e. deleting and appending
together in an atomic operation. This is done simply by specifying a dataset to be appended while also defining
a ``delete_scope`` over the partition. The following example illustrates how both can be performed
with one update:

.. ipython:: python
    :okwarning:

    df  # Column B includes 2 values for '2013-01-02' and another 2 for '2013-01-03'

    dm = store_dataframes_as_dataset(store_url, "replace_partition", [df], partition_on="B")
    sorted(dm.partitions.keys())  # two partitions, one for each value of `B`

    modified_df = another_df.copy()
    # set column E to have value 'train' for all rows in this dataframe
    modified_df.B = pd.Timestamp("20130103").as_unit("ns")

    dm = update_dataset_from_dataframes(
        [
            modified_df
        ],  # specify dataframe which has 'new' data for partition to be replaced
        store=store_url,
        dataset_uuid=dm.uuid,
        partition_on="B",  # don't forget to specify the partitioning column
        delete_scope=[
            {"B": pd.Timestamp("2013-01-03").as_unit("ns")}
        ],  # specify the partition to be deleted
    )
    sorted(dm.partitions.keys())

    read_table(dm.uuid, store_url)


As can be seen in the example above, the resultant dataframe from :func:`~plateau.io.eager.read_table`
consists of two rows corresponding to ``B=2013-01-02`` (from ``df``) and four rows corresponding to ``B=2013-01-03`` from ``modified_df``.
Thus, the original partition with the two rows corresponding to ``B=2013-01-03`` from ``df``
has been completely replaced.



Garbage collection
------------------

When plateau is executing an operation, it makes sure to not
commit changes to the dataset until the operation has been successfully completed. If a
write operation does not succeed for any reason, although there may be new files written
to storage, those files will not be used by the dataset as they will not be referenced in
the plateau metadata. Thus, when the user reads the dataset, no new data will
appear in the output.

Similarly, when deleting a partition, plateau only removes the reference of that file
from the metadata.

These temporary files will remain in storage until a plateau  garbage collection
function is called on the dataset.
If a dataset is updated on a regular basis, it may be useful to run garbage collection
periodically to decrease unnecessary storage use.

An example of garbage collection is shown below.
A little above, near the end of the delete section,
we removed two partitions for the dataset with uuid `replace_partition`.
The removed files remain in storage but are untracked by plateau.
When garbage collection is called, the files are removed.

.. ipython:: python

    from plateau.api.dataset import garbage_collect_dataset
    from minimalkv import get_store_from_url

    store = get_store_from_url(store_url)

    files_before = set(store.keys())

    garbage_collect_dataset(store=store, dataset_uuid=dm.uuid)

    files_before.difference(store.keys())  # Show files removed

.. _minimalkv: https://github.com/data-engineering-collective/minimalkv


Mutating indexed datasets
-------------------------

The mutating operation will update all indices that currently exist for the dataset. This even holds true in case the update function does not specify any or only partially the indices. Consider the following example

.. ipython:: python
    :okwarning:

    df = pd.DataFrame({"payload": range(10), "i1": 0, "i2": ["a"] * 5 + ["b"] * 5})
    dm = store_dataframes_as_dataset(
        store_url, "indexed_dataset", [df], secondary_indices=["i1", "i2"]
    )
    dm = dm.load_all_indices(store_url)
    dm.indices["i1"].observed_values()
    dm.indices["i2"].observed_values()

    new_df = pd.DataFrame({"payload": range(10), "i1": 1, "i2": "c"})

If we do not specify anything, plateau will infer the indices and update them correctly

.. ipython:: python
    :okwarning:

    dm = update_dataset_from_dataframes([new_df], store=store_url, dataset_uuid=dm.uuid)

    dm = dm.load_all_indices(store_url)
    dm.indices["i1"].observed_values()
    dm.indices["i2"].observed_values()


This is even true if only a subset is given

.. ipython:: python
    :okwarning:

    new_df = pd.DataFrame({"payload": range(10), "i1": 2, "i2": "d"})
    dm = update_dataset_from_dataframes(
        [new_df], store=store_url, dataset_uuid=dm.uuid, secondary_indices="i1"
    )

    dm = dm.load_all_indices(store_url)
    dm.indices["i1"].observed_values()
    dm.indices["i2"].observed_values()
