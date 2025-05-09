import pickle
from functools import partial

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from dask.dataframe.utils import assert_eq as assert_dask_eq
from packaging import version
from pandas import testing as pdt
from pandas.testing import assert_frame_equal

from plateau.core.testing import get_dataframe_not_nested
from plateau.io.dask.dataframe import read_dataset_as_ddf
from plateau.io.eager import store_dataframes_as_dataset
from plateau.io.testing.read import *  # noqa
from plateau.io_components.metapartition import SINGLE_TABLE

PANDAS_LT_3 = version.parse(pd.__version__) < version.parse("3.0.0.dev")


@pytest.fixture()
def output_type():
    return "table"


def _read_as_ddf(
    dataset_uuid,
    store,
    factory=None,
    categoricals=None,
    tables=None,
    dataset_has_index=False,
    **kwargs,
):
    table = tables or SINGLE_TABLE

    ddf = read_dataset_as_ddf(
        dataset_uuid=dataset_uuid,
        store=store,
        factory=factory,
        categoricals=categoricals,
        table=table,
        **kwargs,
    )
    if categoricals:
        assert ddf._meta.dtypes["P"] == pd.api.types.CategoricalDtype(
            categories=["__UNKNOWN_CATEGORIES__"], ordered=False
        )
        if dataset_has_index:
            assert ddf._meta.dtypes["L"] == pd.api.types.CategoricalDtype(
                categories=[1, 2], ordered=False
            )
        else:
            assert ddf._meta.dtypes["L"] == pd.api.types.CategoricalDtype(
                categories=["__UNKNOWN_CATEGORIES__"], ordered=False
            )

    s = pickle.dumps(ddf, pickle.HIGHEST_PROTOCOL)
    ddf = pickle.loads(s)

    ddf = ddf.compute().reset_index(drop=True)

    def extract_dataframe(ix):
        df = ddf.iloc[[ix]].copy()
        for col in df.columns:
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = df[col].cat.remove_unused_categories()
        return df.reset_index(drop=True)

    return [extract_dataframe(ix) for ix in ddf.index]


@pytest.fixture()
def bound_load_dataframes():
    return _read_as_ddf


def test_load_dataframe_categoricals_with_index(dataset_with_index_factory):
    func = partial(_read_as_ddf, dataset_has_index=True)
    test_read_dataset_as_dataframes(  # noqa: F405
        dataset_factory=dataset_with_index_factory,
        dataset=dataset_with_index_factory,
        store_session_factory=dataset_with_index_factory.store_factory,
        use_dataset_factory=True,
        bound_load_dataframes=func,
        use_categoricals=True,
        output_type="table",
        dates_as_object=False,
    )


def test_read_ddf_from_categorical_partition(store_factory):
    df = pd.DataFrame({"x": ["a"]}).astype({"x": "category"})
    store_dataframes_as_dataset(
        dfs=[df], dataset_uuid="dataset_uuid", store=store_factory
    )
    ddf = read_dataset_as_ddf(
        dataset_uuid="dataset_uuid", store=store_factory, table="table"
    )
    df_expected = pd.DataFrame({"x": ["a"]})
    df_actual = ddf.compute(scheduler="sync")
    pdt.assert_frame_equal(df_expected, df_actual)

    ddf = read_dataset_as_ddf(
        dataset_uuid="dataset_uuid",
        store=store_factory,
        categoricals=["x"],
        table="table",
    )
    df_actual = ddf.compute(scheduler="sync")
    pdt.assert_frame_equal(df, df_actual)


@pytest.mark.parametrize("index_type", ["primary", "secondary"])
def test_reconstruct_dask_index(store_factory, index_type, monkeypatch):
    dataset_uuid = "dataset_uuid"
    colA = "ColumnA"
    colB = "ColumnB"
    df1 = pd.DataFrame({colA: [1, 2], colB: ["x", "y"]})
    df2 = pd.DataFrame({colA: [3, 4], colB: ["x", "y"]})
    df_chunks = np.array_split(pd.concat([df1, df2]).reset_index(drop=True), 4)
    if not PANDAS_LT_3:
        # Workaround for https://github.com/numpy/numpy/issues/24889.
        df_chunks = [
            pd.DataFrame(data=arr, columns=df1.columns).astype(df1.dtypes)
            for arr in df_chunks
        ]
    with dask.config.set(
        {"dataframe.convert-string": False, "dataframe.shuffle.method": "tasks"}
    ):
        ddf_expected = dd.from_map(lambda x: x, df_chunks).set_index(
            colA, divisions=[1, 2, 3, 4, 4]
        )
        ddf_expected_simple = dd.from_pandas(
            pd.concat([df1, df2]), npartitions=2
        ).set_index(colA)

    if index_type == "secondary":
        secondary_indices = colA
        partition_on = None
    else:
        secondary_indices = None
        partition_on = colA

    store_dataframes_as_dataset(
        store=store_factory,
        dataset_uuid=dataset_uuid,
        dfs=[df1, df2],
        secondary_indices=secondary_indices,
        partition_on=partition_on,
    )

    ddf = read_dataset_as_ddf(
        dataset_uuid=dataset_uuid,
        store=store_factory,
        table="table",
        dask_index_on=colA,
    )

    assert ddf_expected.npartitions == 4
    assert len(ddf_expected.divisions) == 5
    assert ddf_expected.divisions == (1, 2, 3, 4, 4)
    assert ddf.index.name == colA

    assert ddf.npartitions == 4
    assert len(ddf.divisions) == 5
    assert ddf.divisions == (1, 2, 3, 4, 4)

    assert_dask_eq(ddf_expected, ddf, scheduler="distributed")
    assert_dask_eq(
        ddf_expected_simple, ddf, check_divisions=False, scheduler="distributed"
    )

    assert_frame_equal(ddf_expected.compute(), ddf.compute())
    assert_frame_equal(ddf_expected_simple.compute(), ddf.compute())


@pytest.fixture()
def setup_reconstruct_dask_index_types(store_factory, df_not_nested):
    indices = list(df_not_nested.columns)
    indices.remove("null")
    return store_dataframes_as_dataset(
        store=store_factory,
        dataset_uuid="dataset_uuid",
        dfs=[df_not_nested],
        secondary_indices=indices,
    )


@pytest.mark.parametrize("col", get_dataframe_not_nested().columns)
def test_reconstruct_dask_index_types(
    store_factory, setup_reconstruct_dask_index_types, col
):
    if col == "null":
        pytest.xfail(reason="Cannot index null column")
    ddf = read_dataset_as_ddf(
        dataset_uuid=setup_reconstruct_dask_index_types.uuid,
        store=store_factory,
        table="table",
        dask_index_on=col,
    )
    assert ddf.known_divisions
    assert ddf.index.name == col


def test_reconstruct_dask_index_sorting(store_factory, monkeypatch):
    dataset_uuid = "dataset_uuid"
    colA = "ColumnA"
    colB = "ColumnB"

    df = pd.DataFrame(
        {colA: np.random.randint(high=100000, low=-100000, size=(50,)), colB: 0}
    )
    store_dataframes_as_dataset(
        store=store_factory, dataset_uuid=dataset_uuid, dfs=[df], partition_on=colA
    )
    ddf = read_dataset_as_ddf(
        dataset_uuid=dataset_uuid,
        store=store_factory,
        table="table",
        dask_index_on=colA,
    )

    assert all(
        ddf.map_partitions(lambda df: df.index.min()).compute().values
        == ddf.divisions[:-1]
    )


def test_reconstruct_dask_index_raise_no_index(store_factory):
    dataset_uuid = "dataset_uuid"
    colA = "ColumnA"
    df1 = pd.DataFrame({colA: [1, 2]})
    store_dataframes_as_dataset(
        store=store_factory, dataset_uuid=dataset_uuid, dfs=[df1]
    )
    with pytest.raises(
        RuntimeError,
        match=r"Requested index: \['ColumnA'\] but available index columns: \[\]",
    ):
        read_dataset_as_ddf(
            dataset_uuid=dataset_uuid,
            store=store_factory,
            table="table",
            dask_index_on=colA,
        )


def test_column_projection(store_factory, monkeypatch):
    dataset_uuid = "dataset_uuid"
    df1 = pd.DataFrame({"colA": [1, 2], "colB": ["x", "y"]})
    fake_called = False

    class FakeParquet:
        @classmethod
        def restore_dataframe(cls, store, key, filter_query, columns, *args, **kwargs):
            nonlocal fake_called
            fake_called = True
            assert columns == ["colA"]
            return df1[columns]

    from plateau.serialization import DataFrameSerializer

    monkeypatch.setitem(
        DataFrameSerializer._serializers,
        ".parquet",
        FakeParquet,
    )
    store_dataframes_as_dataset(
        store=store_factory, dataset_uuid=dataset_uuid, dfs=[df1]
    )
    ddf_auto = read_dataset_as_ddf(
        dataset_uuid=dataset_uuid,
        store=store_factory,
    )["colA"]
    ddf_manual = read_dataset_as_ddf(
        dataset_uuid=dataset_uuid, store=store_factory, columns=["colA"]
    )["colA"]
    assert_dask_eq(ddf_auto, ddf_manual)
    assert fake_called


def test_dask_index_on_non_string_raises(store_factory):
    dataset_uuid = "dataset_uuid"
    colA = 1
    df1 = pd.DataFrame({colA: [1, 2]})
    store_dataframes_as_dataset(
        store=store_factory, dataset_uuid=dataset_uuid, dfs=[df1]
    )
    with pytest.raises(
        TypeError,
        match=f"The parameter `dask_index_on` must be a string but got {type(colA)}",
    ):
        read_dataset_as_ddf(
            dataset_uuid=dataset_uuid,
            store=store_factory,
            table="table",
            dask_index_on=colA,
        )


def test_dask_dispatch_by_raises_if_index_on_not_none(store_factory):
    dataset_uuid = "dataset_uuid"
    colA = "ColumnA"
    df1 = pd.DataFrame({colA: [1, 2]})
    store_dataframes_as_dataset(
        store=store_factory, dataset_uuid=dataset_uuid, dfs=[df1]
    )
    with pytest.raises(
        ValueError,
        match="`read_dataset_as_ddf` got parameters `dask_index_on` and `dispatch_by`. "
        "Note that `dispatch_by` can only be used if `dask_index_on` is None.",
    ):
        read_dataset_as_ddf(
            dataset_uuid=dataset_uuid,
            store=store_factory,
            table="table",
            dask_index_on=colA,
            dispatch_by=[colA],
        )
