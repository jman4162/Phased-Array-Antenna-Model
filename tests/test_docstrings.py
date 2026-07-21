"""Run every module's docstring examples as doctests."""

import doctest
import importlib

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
