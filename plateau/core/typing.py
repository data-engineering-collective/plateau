from collections.abc import Callable

from minimalkv import KeyValueStore

StoreFactory = Callable[[], KeyValueStore]
StoreInput = str | KeyValueStore | StoreFactory
