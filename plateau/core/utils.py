import pickle
from functools import partial
from typing import Any, cast

from minimalkv import KeyValueStore, get_store_from_url

from plateau.core.naming import MAX_METADATA_VERSION, MIN_METADATA_VERSION
from plateau.core.typing import StoreFactory, StoreInput

__all__ = ("ensure_store", "lazy_store")


def _verify_metadata_version(metadata_version: int) -> None:
    """This is factored out to be an easier target for mocking."""
    if metadata_version < MIN_METADATA_VERSION:
        raise NotImplementedError(
            f"Minimal supported metadata version is 4. You requested {metadata_version} instead."
        )
    elif metadata_version > MAX_METADATA_VERSION:
        raise NotImplementedError(
            f"Future metadata version `{metadata_version}` encountered."
        )


verify_metadata_version = _verify_metadata_version


def ensure_string_type(obj: bytes | str) -> str:
    """Parse object passed to the function to `str`.

    If the object is of type `bytes`, it is decoded, otherwise a generic string representation of the object is
    returned.

    Parameters
    ----------
    obj
        object which is to be parsed to `str`
    """
    if isinstance(obj, bytes):
        return obj.decode()
    else:
        return str(obj)


def _is_minimalkv_key_value_store(obj: Any) -> bool:
    """Check whether ``obj`` is the ``minimalkv.KeyValueStore``-like class.

    minimalkv uses duck-typing, e.g. for decorators. Therefore, avoid
    `isinstance(store, KeyValueStore)`, as it would be unreliable.
    Instead, only roughly verify that `store` looks like a
    KeyValueStore.
    """
    return hasattr(obj, "iter_prefixes")


def ensure_store(store: StoreInput) -> KeyValueStore:
    """Convert the ``store`` argument to a ``KeyValueStore``, without pickle
    test."""
    # This function is often used in an eager context where we may allow
    # non-serializable stores, so skip the pickle test.
    if _is_minimalkv_key_value_store(store):
        return cast(KeyValueStore, store)
    return lazy_store(store)()


def _identity(store: KeyValueStore) -> KeyValueStore:
    """Helper function for `lazy_store`."""
    return store


def lazy_store(store: StoreInput) -> StoreFactory:
    """Create a store factory from the input. Acceptable inputs are.

    * Storefact store url
    * Callable[[], KeyValueStore]
    * KeyValueStore

    If a KeyValueStore is provided, it is verified that the store is serializable
    (i.e. that pickle.dumps does not raise an exception).
    """
    if callable(store):
        return cast(StoreFactory, store)
    elif isinstance(store, str):
        ret_val = partial(get_store_from_url, store)
        ret_val = cast(StoreFactory, ret_val)  # type: ignore
        return ret_val
    else:
        if not _is_minimalkv_key_value_store(store):
            raise TypeError(
                f"Provided incompatible store type. Got {type(store)} but expected {StoreInput}."
            )

        try:
            pickle.dumps(store, pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            raise TypeError(
                """KeyValueStore not serializable.
Please consult https://plateau.readthedocs.io/en/stable/spec/store_interface.html for more information."""
            ) from exc
        return partial(_identity, store)
