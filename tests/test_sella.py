"""Tests for run_sella using a mocked Sella optimizer (no UMA required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from omol_ts_bench.core.sella import SellaResult, run_sella


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_atoms():
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])


def _make_calc(energy=-1.0):
    """Minimal ASE-compatible mock calculator."""
    calc = MagicMock()
    calc.get_potential_energy.return_value = energy
    calc.get_forces.return_value = np.zeros((2, 3))
    # ASE calls calc.calculate() internally in some paths; make it a no-op.
    calc.calculate.return_value = None
    calc.results = {"energy": energy, "forces": np.zeros((2, 3))}
    return calc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("sella.Sella")
def test_run_sella_returns_sella_result(MockSella):
    mock_opt = MockSella.return_value
    mock_opt.run.return_value = True
    mock_opt.nsteps = 5
    mock_opt.H = None

    atoms = _make_atoms()
    calc = _make_calc()
    atoms.calc = calc

    result = run_sella(atoms, calc)
    assert isinstance(result, SellaResult)


@patch("sella.Sella")
def test_run_sella_converged(MockSella):
    mock_opt = MockSella.return_value
    mock_opt.run.return_value = True
    mock_opt.nsteps = 10
    mock_opt.H = None

    atoms = _make_atoms()
    calc = _make_calc(energy=-2.5)
    atoms.calc = calc

    result = run_sella(atoms, calc)
    assert result.converged is True


@patch("sella.Sella")
def test_run_sella_not_converged(MockSella):
    mock_opt = MockSella.return_value
    mock_opt.run.return_value = False
    mock_opt.nsteps = 500
    mock_opt.H = None

    atoms = _make_atoms()
    calc = _make_calc()
    atoms.calc = calc

    result = run_sella(atoms, calc)
    assert result.converged is False


@patch("sella.Sella")
def test_run_sella_ts_optimized_is_atoms(MockSella):
    mock_opt = MockSella.return_value
    mock_opt.run.return_value = True
    mock_opt.nsteps = 7
    mock_opt.H = None

    atoms = _make_atoms()
    calc = _make_calc()
    atoms.calc = calc

    result = run_sella(atoms, calc)
    assert result.ts_optimized is not None
    assert len(result.ts_optimized) == len(atoms)


@patch("sella.Sella")
def test_run_sella_n_steps(MockSella):
    mock_opt = MockSella.return_value
    mock_opt.run.return_value = True
    mock_opt.nsteps = 42
    mock_opt.H = None

    atoms = _make_atoms()
    calc = _make_calc()
    atoms.calc = calc

    result = run_sella(atoms, calc)
    assert result.n_steps == 42


@patch("sella.Sella")
def test_run_sella_exception_returns_unconverged(MockSella):
    mock_opt = MockSella.return_value
    mock_opt.run.side_effect = RuntimeError("optimizer blew up")

    atoms = _make_atoms()
    calc = _make_calc()
    atoms.calc = calc

    result = run_sella(atoms, calc)
    assert result.converged is False
    assert result.ts_optimized is None
    assert result.ts_energy is None


@patch("sella.Sella")
def test_run_sella_kwargs_forwarded(MockSella):
    """fmax, max_steps, order, delta, gamma should reach the Sella constructor."""
    mock_opt = MockSella.return_value
    mock_opt.run.return_value = True
    mock_opt.nsteps = 1
    mock_opt.H = None

    atoms = _make_atoms()
    calc = _make_calc()
    atoms.calc = calc

    run_sella(atoms, calc, fmax=0.01, max_steps=10, order=1, delta=0.005, gamma=0.3)

    _, kwargs = MockSella.call_args
    assert kwargs["order"] == 1
    assert kwargs["delta0"] == 0.005
    assert kwargs["gamma"] == 0.3

    _, run_kwargs = mock_opt.run.call_args
    assert run_kwargs["fmax"] == 0.01
    assert run_kwargs["steps"] == 10
