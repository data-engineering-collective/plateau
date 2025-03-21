import pytest
from minimalkv import get_store_from_url


@pytest.fixture
def store():
    """Create a temporary store for testing."""
    store = get_store_from_url("memory://")
    return store


@pytest.fixture
def store_factory():
    """Create a factory that returns a fresh store for each call."""
    return lambda: get_store_from_url("memory://")


@pytest.fixture
def mock_uuid(monkeypatch):
    """Mock UUID generation to return a predictable value."""
    import plateau.core.uuid as uuid_module

    monkeypatch.setattr(uuid_module, "gen_uuid", lambda: "auto_dataset_uuid")


@pytest.fixture
def metadata_version():
    """Return the metadata version to use in tests."""
    return 4


@pytest.fixture
def metadata_storage_format():
    """Return the metadata storage format to use in tests."""
    return "msgpack"


@pytest.fixture
def frozen_time(monkeypatch):
    """Freeze time to a fixed value for testing."""
    import datetime

    frozen = datetime.datetime(2010, 1, 1, 12, 0, 0)
    monkeypatch.setattr(datetime, "datetime", lambda: frozen)
    return frozen
