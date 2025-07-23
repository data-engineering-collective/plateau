import pandas as pd
import simplejson
from packaging.version import parse as parse_version

PANDAS_3 = parse_version(pd.__version__).major >= 3


def pandas_infer_string():
    # This option is available since pandas 1.5.0
    return (
        pd.get_option("future.infer_string") or parse_version(pd.__version__).major >= 3
    )


def load_json(buf, **kwargs):
    """Compatibility function to load JSON from str/bytes/unicode.

    For Python 2.7 json.loads accepts str and unicode. Python 3.4 only
    accepts str whereas 3.5+ accept bytes and str.
    """
    if isinstance(buf, bytes):
        return simplejson.loads(buf.decode("utf-8"), **kwargs)
    else:
        return simplejson.loads(buf, **kwargs)
