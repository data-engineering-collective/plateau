import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except Exception:  # pragma: no cover
    __version__ = "unknown"
