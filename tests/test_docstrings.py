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
    use the classic repr. legacy='1.25' restores it for the doctest run."""
    try:
        np.set_printoptions(legacy="1.25")
    except (TypeError, ValueError):
        pass  # NumPy < 2 already uses the classic repr
    yield
    np.set_printoptions(legacy=False)


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
