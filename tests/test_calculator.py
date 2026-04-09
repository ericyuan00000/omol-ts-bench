"""Tests for the UMA calculator."""

import numpy as np
import pytest
from ase import Atoms


@pytest.fixture
def uma_calc():
    """Get UMA calculator, skip if fairchem not available."""
    try:
        from omol_ts_bench.core.calculator import get_uma_calculator
        return get_uma_calculator()
    except ImportError:
        pytest.skip("fairchem-core not installed")


def test_uma_energy(uma_calc):
    """UMA should return a finite energy for H2."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = uma_calc
    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)


def test_uma_forces_shape(uma_calc):
    """UMA forces should have shape (n_atoms, 3)."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = uma_calc
    forces = atoms.get_forces()
    assert forces.shape == (2, 3)


def test_uma_forces_finite(uma_calc):
    """UMA forces should be finite."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = uma_calc
    forces = atoms.get_forces()
    assert np.all(np.isfinite(forces))


def test_uma_organometallic(uma_calc):
    """UMA should handle a simple organometallic system (Fe-CO)."""
    atoms = Atoms(
        "FeCO",
        positions=[[0, 0, 0], [0, 0, 1.8], [0, 0, 2.95]],
    )
    atoms.info["charge"] = 0
    atoms.info["spin"] = 3
    atoms.calc = uma_calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert np.isfinite(energy)
    assert forces.shape == (3, 3)
    assert np.all(np.isfinite(forces))
