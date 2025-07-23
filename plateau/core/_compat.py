import simplejson
from packaging.version import parse as parse_version


def pandas_infer_string():
    import pandas as pd

    # This option is available since pandas 1.5.0
    return pd.get_option("future.infer_string") or parse_version(
        pd.__version__
    ) >= parse_version("3.0.0")


def arrow_uses_large_string():
    """Check if the current Arrow version uses large_string for unicode."""
    import pyarrow as pa
    from packaging.version import parse as parse_version

    arrow_version = parse_version(pa.__version__)
    return arrow_version >= parse_version("20.0.0")


def load_json(buf, **kwargs):
    """Compatibility function to load JSON from str/bytes/unicode.

    For Python 2.7 json.loads accepts str and unicode. Python 3.4 only
    accepts str whereas 3.5+ accept bytes and str.
    """
    if isinstance(buf, bytes):
        return simplejson.loads(buf.decode("utf-8"), **kwargs)
    else:
        return simplejson.loads(buf, **kwargs)
