Examples
--------

Setup a store

.. ipython:: python

    from tempfile import TemporaryDirectory

    # You can, of course, also directly use S3, ABS or anything else
    # supported by :mod:`minimalkv`
    dataset_dir = TemporaryDirectory()
    store_url = f"hfs://{dataset_dir.name}"


.. ipython:: python
    :okwarning:

    import pandas as pd
    from plateau.api.dataset import read_table, store_dataframes_as_dataset

    df = pd.DataFrame({"Name": ["Paul", "Lisa"], "Age": [32, 29]})

    dataset_uuid = "my_list_of_friends"

    metadata = {
        "Name": "My list of friends",
        "Columns": {
            "Name": "First name of my friend",
            "Age": "honest age of my friend in years",
        },
    }

    store_dataframes_as_dataset(
        store=store_url, dataset_uuid=dataset_uuid, dfs=[df], metadata=metadata
    )

    # Load your data
    # By default the single dataframe is stored in the 'core' table
    df_from_store = read_table(store=store_url, dataset_uuid=dataset_uuid)
    df_from_store


Eager
`````

Write
~~~~~

.. ipython:: python
    :okwarning:

    import pandas as pd
    from plateau.api.dataset import store_dataframes_as_dataset

    #  Now, define the actual partitions. This list will, most of the time,
    # be the intermediate result of a previously executed pipeline which e.g. pulls
    # data from an external data source
    # In our particular case, we'll use manual input and define our partitions explicitly

    # We'll define two partitions which both have two tables
    input_list_of_partitions = [
        pd.DataFrame({"A": range(10)}),
        pd.DataFrame({"A": range(10, 20)}),
    ]

    # The pipeline will return a :class:`~plateau.core.dataset.DatasetMetadata` object
    #  which refers to the created dataset
    dataset = store_dataframes_as_dataset(
        dfs=input_list_of_partitions,
        store=store_url,
        dataset_uuid="MyFirstDataset",
        metadata={"dataset": "metadata"},  #  This is optional dataset metadata
        metadata_version=4,
    )
    dataset


Read
~~~~

.. ipython:: python

    import pandas as pd
    from plateau.api.dataset import read_dataset_as_dataframes

    #  Create the pipeline with a minimal set of configs
    list_of_partitions = read_dataset_as_dataframes(
        dataset_uuid="MyFirstDataset", store=store_url
    )

    # In case you were using the dataset created in the Write example
    for d1, d2 in zip(
        list_of_partitions,
        [
            pd.DataFrame({"A": range(10)}),
            pd.DataFrame({"A": range(10, 20)}),
        ],
    ):
        for k1, k2 in zip(d1, d2):
            assert k1 == k2


Iter
````
Write
~~~~~

.. ipython:: python
    :okwarning:

    import pandas as pd
    from plateau.api.dataset import store_dataframes_as_dataset__iter

    input_list_of_partitions = [
        pd.DataFrame({"A": range(10)}),
        pd.DataFrame({"A": range(10, 20)}),
    ]

    # The pipeline will return a :class:`~plateau.core.dataset.DatasetMetadata` object
    #  which refers to the created dataset
    dataset = store_dataframes_as_dataset__iter(
        input_list_of_partitions,
        store=store_url,
        dataset_uuid="MyFirstDatasetIter",
        metadata={"dataset": "metadata"},  #  This is optional dataset metadata
        metadata_version=4,
    )
    dataset

Read
~~~~

.. ipython:: python
    :okwarning:

    import pandas as pd
    from plateau.api.dataset import read_dataset_as_dataframes__iterator

    #  Create the pipeline with a minimal set of configs
    list_of_partitions = read_dataset_as_dataframes__iterator(
        dataset_uuid="MyFirstDatasetIter", store=store_url
    )
    # the iter backend returns a generator object. In our case we want to look at
    # all partitions at once
    list_of_partitions = list(list_of_partitions)

    # In case you were using the dataset created in the Write example
    for d1, d2 in zip(
        list_of_partitions,
        [
            pd.DataFrame({"A": range(10)}),
            pd.DataFrame({"A": range(10, 20)}),
        ],
    ):
        for k1, k2 in zip(d1, d2):
            assert k1 == k2

Dask
````

Write
~~~~~

.. ipython:: python
    :okwarning:

    import pandas as pd
    from plateau.api.dataset import store_delayed_as_dataset

    input_list_of_partitions = [
        pd.DataFrame({"A": range(10)}),
        pd.DataFrame({"A": range(10, 20)}),
    ]

    # This will return a :class:`~dask.delayed`. The figure below
    # show the generated task graph.
    task = store_delayed_as_dataset(
        input_list_of_partitions,
        store=store_url,
        dataset_uuid="MyFirstDatasetDask",
        metadata={"dataset": "metadata"},  #  This is optional dataset metadata
        metadata_version=4,
    )
    task.compute()

.. figure:: ./taskgraph.jpeg
    :scale: 40%
    :figclass: align-center

    Task graph for the above dataset store pipeline.

Read
~~~~

.. ipython:: python

    import dask
    import pandas as pd
    from plateau.api.dataset import read_dataset_as_delayed

    tasks = read_dataset_as_delayed(dataset_uuid="MyFirstDatasetDask", store=store_url)
    tasks
    dask.compute(tasks)
