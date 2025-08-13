import subprocess

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "plateau",
        "plateau.serialization",
        "plateau.io",
        "plateau.io_components",
        "plateau.utils",
        "plateau.api",
        "plateau.io.dask",
        "plateau.io.testing",
        "plateau.io.eager",
        "plateau.io.iter",
    ],
)
def test_imports(module):
    subprocess.run(["python", "-c", f"import {module}"], check=True)
