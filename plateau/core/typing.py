from collections.abc import Callable
from typing import Union

from minimalkv import KeyValueStore

StoreFactory = Callable[[], KeyValueStore]
StoreInput = Union[str, KeyValueStore, StoreFactory]
