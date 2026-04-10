"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def uma_calc():
    """UMA-small with OMol head, loaded once per test session."""
    try:
        from omol_ts_bench.core.calculator import get_uma_calculator
        return get_uma_calculator()
    except ImportError:
        pytest.skip("fairchem-core not installed")
