import os
import uuid

import pandas.testing as pdt
import pytest
from minimalkv import get_store_from_url

from plateau.core.testing import get_dataframe_alltypes
from plateau.serialization import ParquetSerializer

KNOWN_ARROW_VERSIONS = [
    "4.0.1",
    "5.0.0",
    "6.0.1",
    "7.0.0",
    "8.0.1",
    "9.0.0",
    "10.0.1",
    "11.0.0",
    "12.0.0",
    "13.0.0",
    "14.0.1",
    "14.0.2",
    "15.0.0",
    "15.0.2",
    "16.1.0",
    "17.0.0",
    "18.1.0",
    "19.0.0",
    "19.0.1",
    "20.0.0",
]


@pytest.fixture(params=KNOWN_ARROW_VERSIONS)
def arrow_version(request):
    yield request.param


@pytest.fixture
def reference_store():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "..",
        "reference-data",
        "arrow-compat",
    )
    return get_store_from_url(f"hfs://{path}")


def test_current_arrow_version_tested():
    """Ensure that we do not forget to generate the reference file."""
    import pyarrow as pa
    from packaging.version import parse

    version = parse(pa.__version__)
    is_stable = not version.is_devrelease and not version.is_prerelease
    if is_stable:
        assert pa.__version__ in KNOWN_ARROW_VERSIONS


def test_arrow_compat(arrow_version, reference_store, mocker):
    """Test if reading/writing across the supported arrow versions is actually
    compatible.

    Generate new reference files by going to the `reference-data/arrow-
    compat` directory and executing `generate_reference.py` or
    `batch_generate_reference.sh`.
    """

    uuid_hook = mocker.patch("plateau.core.uuid._uuid_hook_object")
    uuid_hook.return_value = uuid.UUID(
        bytes=b"\x82\xd6\xc1\x06Z\x08\x11\xe9\x85eJ\x00\x07\xf8\n\x10"
    )

    orig = get_dataframe_alltypes()
    restored = ParquetSerializer().restore_dataframe(
        store=reference_store, key=arrow_version + ".parquet", date_as_object=True
    )

    pdt.assert_frame_equal(orig, restored)
