import pytest

from plateau.io.eager import garbage_collect_dataset
from plateau.io.testing.gc import *  # noqa: F403, F4


@pytest.fixture()
def garbage_collect_callable():
    return garbage_collect_dataset
