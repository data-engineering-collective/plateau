# pylint: disable=E1101


import pandas as pd
import pytest
from packaging import version

from plateau.core.dataset import DatasetMetadata
from plateau.core.index import ExplicitSecondaryIndex
from plateau.core.testing import TIME_TO_FREEZE_ISO
from plateau.io_components.metapartition import MetaPartition
from plateau.io_components.read import dispatch_metapartitions
from plateau.io_components.write import (
    raise_if_dataset_exists,
    store_dataset_from_partitions,
)


def test_store_dataset_from_partitions(meta_partitions_files_only, store, frozen_time):
    dataset = store_dataset_from_partitions(
        partition_list=meta_partitions_files_only,
        dataset_uuid="dataset_uuid",
        store=store,
        dataset_metadata={"some": "metadata"},
    )

    expected_metadata = {"some": "metadata", "creation_time": TIME_TO_FREEZE_ISO}

    assert dataset.metadata == expected_metadata
    assert sorted(dataset.partitions.values(), key=lambda x: x.label) == sorted(
        (mp.partition for mp in meta_partitions_files_only), key=lambda x: x.label
    )
    assert dataset.uuid == "dataset_uuid"

    store_files = list(store.keys())
    # Dataset metadata: 1 file
    expected_number_files = 1
    # common metadata for v4 datasets
    expected_number_files += 1
    assert len(store_files) == expected_number_files

    # Ensure the dataset can be loaded properly
    stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store)
    assert dataset == stored_dataset


def test_store_dataset_from_partitions_update(store, metadata_version, frozen_time):
    mp1 = MetaPartition(
        label="cluster_1",
        data=pd.DataFrame({"p": [1]}),
        file="1.parquet",
        indices={"p": ExplicitSecondaryIndex("p", index_dct={1: ["cluster_1"]})},
        metadata_version=metadata_version,
    )
    mp2 = MetaPartition(
        label="cluster_2",
        data=pd.DataFrame({"p": [2]}),
        file="2.parquet",
        indices={"p": ExplicitSecondaryIndex("p", index_dct={2: ["cluster_2"]})},
        metadata_version=metadata_version,
    )
    dataset = store_dataset_from_partitions(
        partition_list=[mp1, mp2],
        dataset_uuid="dataset_uuid",
        store=store,
        dataset_metadata={"dataset": "metadata"},
    )
    dataset = dataset.load_index("p", store)

    mp3 = MetaPartition(
        label="cluster_3",
        data=pd.DataFrame({"p": [3]}),
        file="3.parquet",
        indices={"p": ExplicitSecondaryIndex("p", index_dct={3: ["cluster_3"]})},
        metadata_version=metadata_version,
    )

    dataset_updated = store_dataset_from_partitions(
        partition_list=[mp3],
        dataset_uuid="dataset_uuid",
        store=store,
        dataset_metadata={"extra": "metadata"},
        update_dataset=dataset,
        remove_partitions=["cluster_1"],
    )
    dataset_updated = dataset_updated.load_index("p", store)
    expected_metadata = {"dataset": "metadata", "extra": "metadata"}

    expected_metadata["creation_time"] = TIME_TO_FREEZE_ISO

    assert dataset_updated.metadata == expected_metadata
    assert list(dataset.partitions) == ["cluster_1", "cluster_2"]
    assert list(dataset_updated.partitions) == ["cluster_2", "cluster_3"]
    assert dataset_updated.partitions["cluster_3"] == mp3.partition
    assert dataset_updated.uuid == "dataset_uuid"

    store_files = list(store.keys())
    # 1 dataset metadata file and 1 index file
    # note: the update writes a new index file but due to frozen_time this gets
    # the same name as the previous one and overwrites it.
    expected_number_files = 2
    # common metadata for v4 datasets (1 table)
    expected_number_files += 1
    assert len(store_files) == expected_number_files

    assert dataset.indices["p"].index_dct == {1: ["cluster_1"], 2: ["cluster_2"]}
    assert dataset_updated.indices["p"].index_dct == {
        2: ["cluster_2"],
        3: ["cluster_3"],
    }

    # Ensure the dataset can be loaded properly
    stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store)
    stored_dataset = stored_dataset.load_index("p", store)
    assert dataset_updated == stored_dataset


def test_raise_if_dataset_exists(store_factory, dataset_function):
    raise_if_dataset_exists(dataset_uuid="ThisDoesNotExist", store=store_factory)
    with pytest.raises(RuntimeError):
        raise_if_dataset_exists(dataset_uuid=dataset_function.uuid, store=store_factory)


@pytest.mark.skipif(
    version.parse(pd.__version__) < version.parse("2"),
    reason="Timestamp unit coercion is only relevant in pandas >= 2",
)
def test_coerce_schema_timestamp_units(store):
    date = pd.Timestamp(2000, 1, 1)

    mps_original = [
        MetaPartition(label="one", data=pd.DataFrame({"a": date, "b": [date]})),
        MetaPartition(
            label="two",
            data=pd.DataFrame({"a": date.as_unit("ns"), "b": [date.as_unit("ns")]}),
        ),
    ]

    mps = (
        mp.store_dataframes(store, dataset_uuid="dataset_uuid") for mp in mps_original
    )

    # Expect this not to fail even though the metapartitions have different
    # timestamp units, because all units should be coerced to nanoseconds.
    dataset = store_dataset_from_partitions(
        partition_list=mps,
        dataset_uuid="dataset_uuid",
        store=store,
        dataset_metadata={"some": "metadata"},
    )

    # Ensure the dataset can be loaded properly
    stored_dataset = DatasetMetadata.load_from_store("dataset_uuid", store)
    assert dataset == stored_dataset

    mps = dispatch_metapartitions("dataset_uuid", store)
    mps_loaded = (mp.load_dataframes(store) for mp in mps)

    # Ensure the values and dtypes of the loaded datasets are correct
    for mp in mps_loaded:
        assert mp.data["a"].dtype == "datetime64[ns]"
        assert mp.data["b"].dtype == "datetime64[ns]"
        assert mp.data["a"][0] == date
        assert mp.data["b"][0] == date
