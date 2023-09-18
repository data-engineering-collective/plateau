import pickle

import dask
import dask.dataframe as dd
import pandas as pd
import pytest

from plateau.io.dask.dataframe import update_dataset_from_ddf
from plateau.io.testing.update import *  # noqa


@pytest.fixture
def bound_update_dataset():
    return _update_dataset


def _id(part):
    if isinstance(part, pd.DataFrame):
        return part
    else:
        return part[0]


def _update_dataset(partitions, *args, **kwargs):
    # TODO: Simplify once parse_input_to_metapartition is removed / obsolete

    with dask.config.set({"dataframe.convert-string": False}):
        if isinstance(partitions, pd.DataFrame):
            partitions = dd.from_pandas(partitions, npartitions=1)
        elif partitions is not None:
            delayed_partitions = [dask.delayed(_id)(part) for part in partitions]
            partitions = dd.from_delayed(delayed_partitions)
        else:
            partitions = None

    # Replace `table_name` with `table` keyword argument to enable shared test code
    # via `bound_update_dataset` fixture
    if "table_name" in kwargs:
        kwargs["table"] = kwargs["table_name"]
        del kwargs["table_name"]

    ddf = update_dataset_from_ddf(partitions, *args, **kwargs)

    s = pickle.dumps(ddf, pickle.HIGHEST_PROTOCOL)
    ddf = pickle.loads(s)

    return ddf.compute()


def _return_none():
    return None


@pytest.mark.parametrize("shuffle", [True, False])
def test_update_dataset_from_ddf_empty(store_factory, shuffle):
    with pytest.raises(ValueError) as exc_info:
        update_dataset_from_ddf(
            dask.dataframe.from_delayed([], meta=(("a", int),)),
            store_factory,
            dataset_uuid="output_dataset_uuid",
            table="core",
            shuffle=shuffle,
            partition_on=["a"],
        ).compute()
    assert str(exc_info.value) in [
        "Cannot store empty datasets",  # dask <= 2021.5.0
        "Cannot store empty datasets, partition_list must not be empty if in store mode.",  # dask > 2021.5.0 + shuffle == True
        "No data left to save outside partition columns",  # dask > 2021.5.0 + shuffle == False
    ]
