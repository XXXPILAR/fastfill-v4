"""Keep the existing synthetic geometry and evaluation contract checks in the normal test command."""
import runpy
import sys
import warnings

import pytest


@pytest.mark.parametrize("module", [
    "fastfill.adapters.structured3d", "fastfill.adapters.hssd200",
    "fastfill.adapters.multiscan", "fastfill.adapters.interiorgs",
    "fastfill.evaluate", "fastfill.split", "fastfill.anchors",
])
def test_existing_contracts(module, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*found in sys.modules.*", category=RuntimeWarning)
        runpy.run_module(module, run_name="__main__")
