"""Tests for the GSM stub."""

from __future__ import annotations

import pytest
from ase import Atoms

from omol_ts_bench.core.gsm import run_de_gsm


def test_run_de_gsm_raises_not_implemented():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    with pytest.raises(NotImplementedError):
        run_de_gsm(atoms, atoms, calc=None)
