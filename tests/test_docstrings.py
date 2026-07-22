"""Run every module's docstring examples as doctests."""

import doctest
import importlib

import numpy as np
import pytest

MODULES = [
    "core",
    "utils",
    "geometry",
    "beamforming",
    "impairments",
    "wideband",
    "polarization",
    "coordinates",
    "export",
    "visualization",
]


@pytest.fixture(autouse=True)
def numpy_legacy_repr():
    """NumPy 2 prints scalars as np.float64(16.0); the docstring examples
    use the classic repr. legacy='1.25' restores it for the doctest run.

    NumPy 1.x must not receive the option at all: it warns instead of
    raising and stores the invalid value, corrupting later printing.
    """
    if int(np.__version__.split(".")[0]) >= 2:
        np.set_printoptions(legacy="1.25")
        yield
        np.set_printoptions(legacy=False)
    else:
        yield


@pytest.mark.parametrize("module_name", MODULES)
def test_module_doctests(module_name):
    module = importlib.import_module(f"phased_array.{module_name}")
    results = doctest.testmod(
        module,
        verbose=False,
        report=True,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )
    assert results.failed == 0, (
        f"{results.failed} doctest failure(s) in phased_array.{module_name}"
    )
