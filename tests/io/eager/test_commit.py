import datetime
from collections import OrderedDict

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from plateau.core.common_metadata import make_meta
from plateau.core.dataset import DatasetMetadata
from plateau.io.eager import (
    commit_dataset,
    create_empty_dataset_header,
    read_table,
    store_dataframes_as_dataset,
    write_single_partition,
)


def test_commit_dataset_from_metapartition(dataset_function, store):
    new_data = [
        pd.DataFrame(
            OrderedDict(
                [
                    ("P", [5]),
                    ("L", [5]),
                    ("TARGET", [5]),
                    ("DATE", [datetime.date(2016, 3, 23)]),
                ]
            )
        )
    ]
    new_partition = write_single_partition(
        store=store, dataset_uuid=dataset_function.uuid, data=new_data
    )
    pre_commit_dataset = DatasetMetadata.load_from_store(
        uuid=dataset_function.uuid, store=store
    )
    # Cannot assert equal since the metadata is differently ordered
    assert pre_commit_dataset == dataset_function

    updated_dataset = commit_dataset(
        store=store,
        dataset_uuid=dataset_function.uuid,
        new_partitions=new_partition,
        delete_scope=None,
        partition_on=None,
    )
    assert updated_dataset != dataset_function

    assert updated_dataset.uuid == dataset_function.uuid
    assert len(updated_dataset.partitions) == len(dataset_function.partitions) + 1

    # ensure that the new dataset is actually the one on disc
    loaded_dataset = DatasetMetadata.load_from_store(
        uuid=updated_dataset.uuid, store=store
    )
    assert loaded_dataset == updated_dataset

    # Read the data and check whether the rows above are included.
    # This checks whether all necessary information were updated in the header
    # (e.g. files attributes of the partitions)
    actual = read_table(store=store, dataset_uuid=dataset_function.uuid)
    df_expected = pd.DataFrame(
        OrderedDict(
            [
                (
                    "DATE",
                    [
                        datetime.date(2016, 3, 23),
                        datetime.date(2010, 1, 1),
                        datetime.date(2009, 12, 31),
                    ],
                ),
                ("L", [5, 1, 2]),
                ("P", [5, 1, 2]),
                ("TARGET", [5, 1, 2]),
            ]
        )
    )
    actual = actual.sort_values("DATE", ascending=False).reset_index(drop=True)

    assert_frame_equal(df_expected, actual)


# TODO: document changes for write_single_partition / commit workflow
def test_commit_dataset_from_nested_metapartition(store):
    """Check it is possible to use `commit_dataset` with nested metapartitions
    as input.

    Original issue: https://github.com/JDASoftwareGroup/plateau/issues/40
    """

    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [3, 4, 5, 6]})

    create_empty_dataset_header(
        store=store,
        dataset_uuid="uuid",
        schema=make_meta(df, "table", ["a"]),
        partition_on=["a"],
    )

    partitions = []
    for _ in range(2):
        partitions.append(
            write_single_partition(
                store=store,
                dataset_uuid="uuid",
                data=df,
                partition_on=["a"],
            )
        )

    partition_labels = {mp_.label for mp in partitions for mp_ in mp}
    dm = commit_dataset(
        store=store, dataset_uuid="uuid", new_partitions=partitions, partition_on=["a"]
    )
    assert dm.partitions.keys() == partition_labels


def test_initial_commit(store):
    dataset_uuid = "dataset_uuid"
    df = pd.DataFrame(OrderedDict([("P", [5]), ("L", [5]), ("TARGET", [5])]))
    dataset = create_empty_dataset_header(
        store=store,
        schema=make_meta(df, origin="1"),
        dataset_uuid=dataset_uuid,
        metadata_version=4,
    )
    assert dataset.explicit_partitions is False
    new_metapartition = write_single_partition(
        store=store, dataset_uuid=dataset.uuid, data=df
    )

    updated_dataset = commit_dataset(
        store=store,
        dataset_uuid=dataset.uuid,
        # FIXME: is this breaking and if so, is it expected?
        new_partitions=[new_metapartition],
        delete_scope=None,
        partition_on=None,
    )
    assert updated_dataset.explicit_partitions is True
    actual = read_table(store=store, dataset_uuid=updated_dataset.uuid)
    df_expected = pd.DataFrame(OrderedDict([("L", [5]), ("P", [5]), ("TARGET", [5])]))

    assert_frame_equal(df_expected, actual)


def test_commit_dataset_only_delete(store, metadata_version):
    partitions = [
        pd.DataFrame({"p": [1]}),
        pd.DataFrame({"p": [2]}),
    ]
    dataset = store_dataframes_as_dataset(
        dfs=partitions,
        store=lambda: store,
        metadata={"dataset": "metadata"},
        dataset_uuid="dataset_uuid",
        secondary_indices="p",
        metadata_version=metadata_version,
    )
    dataset = dataset.load_index("p", store)
    assert len(dataset.partitions) == 2

    delete_scope = [{"p": 1}]
    updated_dataset = commit_dataset(
        store=store,
        dataset_uuid=dataset.uuid,
        new_partitions=None,
        delete_scope=delete_scope,
        partition_on=None,
    )
    assert len(updated_dataset.partitions) == 1
    assert updated_dataset.explicit_partitions is True


def test_commit_dataset_delete_all(store, metadata_version):
    partitions = [pd.DataFrame({"p": [1]})]

    dataset = store_dataframes_as_dataset(
        dfs=partitions,
        store=lambda: store,
        metadata={"dataset": "metadata"},
        dataset_uuid="dataset_uuid",
        secondary_indices="p",
        metadata_version=metadata_version,
    )
    dataset = dataset.load_index("p", store)
    assert len(dataset.partitions) == 1

    delete_scope = [{"p": 1}]
    updated_dataset = commit_dataset(
        store=store,
        dataset_uuid=dataset.uuid,
        new_partitions=None,
        delete_scope=delete_scope,
        partition_on=None,
    )
    assert len(updated_dataset.partitions) == 0
    assert updated_dataset.explicit_partitions is True


def test_commit_dataset_add_metadata(store, metadata_version):
    df = pd.DataFrame({"p": [1]})
    dataset = store_dataframes_as_dataset(
        dfs=[df],
        store=lambda: store,
        metadata={"old": "metadata"},
        dataset_uuid="dataset_uuid",
        metadata_version=metadata_version,
    )

    commit_dataset(store=store, dataset_uuid=dataset.uuid, metadata={"new": "metadata"})
    restored_ds = DatasetMetadata.load_from_store(uuid="dataset_uuid", store=store)

    actual_metadata = restored_ds.metadata
    del actual_metadata["creation_time"]
    expected_meta = {"new": "metadata", "old": "metadata"}
    assert actual_metadata == expected_meta


def test_commit_dataset_raises_no_input(store):
    with pytest.raises(ValueError, match="Need to provide either"):
        commit_dataset(store=store, dataset_uuid="something")

    with pytest.raises(ValueError, match="Need to provide either"):
        commit_dataset(store=store, dataset_uuid="something", new_partitions=[])
    with pytest.raises(ValueError, match="Need to provide either"):
        commit_dataset(store=store, dataset_uuid="something", metadata={})
    with pytest.raises(ValueError, match="Need to provide either"):
        commit_dataset(store=store, dataset_uuid="something", delete_scope=[])
