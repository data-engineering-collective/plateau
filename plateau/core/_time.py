"""Functions to get the current time.

plateau modules that need to get the current time should do so only via
this module. This allows test to monkeypatch the methods in this module
to fake certain fixed times.
"""

import datetime
import sys


def datetime_now():
    """Get the current time as datimetime object.

    Same as datetime.datetime.now
    """
    return datetime.datetime.now()


def datetime_utcnow():
    """Get the current time as datimetime object.

    Same as datetime.datetime.utcnow
    """
    if sys.version_info < (3, 11):
        return datetime.datetime.utcnow()
    return datetime.datetime.now(datetime.UTC)
