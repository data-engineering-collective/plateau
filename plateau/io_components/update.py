"""This module contains logic to update an existing dataset.

Update means adding new partitions and deleting existing partitions.
plateau does not allow an update of the content of existing partitions.
"""


from plateau.core.index import PartitionIndex
from plateau.core.utils import ensure_store
from plateau.io_components.write import store_dataset_from_partitions


def _get_partitions(dataset, query_params):
    partitions = []
    for params in query_params:
        partitions += dataset.query(**params)

    return partitions


def update_dataset_from_partitions(
    partition_list,
    store_factory,
    dataset_uuid,
    ds_factory,
    delete_scope,
    metadata,
    metadata_merger,
):
    store = ensure_store(store_factory)

    if ds_factory:
        ds_factory = ds_factory.load_all_indices()
        remove_partitions = _get_partitions(ds_factory, delete_scope)

        index_columns = list(ds_factory.indices.keys())
        for column in index_columns:
            index = ds_factory.indices[column]
            if isinstance(index, PartitionIndex):
                del ds_factory.indices[column]
    else:
        # Dataset does not exist yet.
        remove_partitions = []

    new_dataset = store_dataset_from_partitions(
        partition_list=partition_list,
        store=store,
        dataset_uuid=dataset_uuid,
        dataset_metadata=metadata,
        metadata_merger=metadata_merger,
        update_dataset=ds_factory,
        remove_partitions=remove_partitions,
    )

    return new_dataset
