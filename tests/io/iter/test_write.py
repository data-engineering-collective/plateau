import pytest

from plateau.io.iter import store_dataframes_as_dataset__iter
from plateau.io.testing.write import *  # noqa


def _store_dataframes(df_list, *args, **kwargs):
    df_generator = (x for x in df_list)
    return store_dataframes_as_dataset__iter(df_generator, *args, **kwargs)


@pytest.fixture()
def bound_store_dataframes():
    return _store_dataframes
