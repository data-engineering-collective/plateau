import pytest

from plateau.io.eager import build_dataset_indices
from plateau.io.testing.index import *  # noqa: F403, F4


@pytest.fixture()
def bound_build_dataset_indices():
    return build_dataset_indices
