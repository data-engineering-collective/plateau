import pandas as pd
import pyarrow as pa
import simplejson
from packaging.version import parse as parse_version
from pandas.errors import OptionError

PANDAS_3 = parse_version(pd.__version__).major >= 3

ARROW_GE_20 = parse_version(pa.__version__).major >= 20


def pandas_infer_string() -> bool:
    if parse_version(pd.__version__).major >= 3:
        # In pandas 3, infer_string is always True
        return True
    try:
        return pd.get_option("future.infer_string")
    except OptionError:
        return False


def load_json(buf, **kwargs):
    """Compatibility function to load JSON from str/bytes/unicode.

    For Python 2.7 json.loads accepts str and unicode. Python 3.4 only
    accepts str whereas 3.5+ accept bytes and str.
    """
    if isinstance(buf, bytes):
        return simplejson.loads(buf.decode("utf-8"), **kwargs)
    else:
        return simplejson.loads(buf, **kwargs)
