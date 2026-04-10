"""Tests for the UMA calculator (uma-s-1p2, omol head).

All tests use the session-scoped `uma_calc` fixture from conftest.py so the
model is loaded only once per pytest run.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _h2(d=0.74, charge=0, spin=1):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, d]])
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    return atoms


def _feco(charge=0, spin=3):
    atoms = Atoms("FeCO", positions=[[0, 0, 0], [0, 0, 1.8], [0, 0, 2.95]])
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    return atoms


# ---------------------------------------------------------------------------
# Basic energy / force sanity
# ---------------------------------------------------------------------------

def test_uma_energy_finite(uma_calc):
    atoms = _h2()
    atoms.calc = uma_calc
    assert np.isfinite(atoms.get_potential_energy())


def test_uma_forces_shape(uma_calc):
    atoms = _h2()
    atoms.calc = uma_calc
    assert atoms.get_forces().shape == (2, 3)


def test_uma_forces_finite(uma_calc):
    atoms = _h2()
    atoms.calc = uma_calc
    assert np.all(np.isfinite(atoms.get_forces()))


def test_uma_organometallic_energy_finite(uma_calc):
    atoms = _feco()
    atoms.calc = uma_calc
    assert np.isfinite(atoms.get_potential_energy())


def test_uma_organometallic_forces_shape(uma_calc):
    atoms = _feco()
    atoms.calc = uma_calc
    assert atoms.get_forces().shape == (3, 3)


def test_uma_organometallic_forces_finite(uma_calc):
    atoms = _feco()
    atoms.calc = uma_calc
    assert np.all(np.isfinite(atoms.get_forces()))


# ---------------------------------------------------------------------------
# PES shape: energy should vary meaningfully with geometry
# ---------------------------------------------------------------------------

def test_uma_energy_increases_when_compressed(uma_calc):
    """Compressing H2 well below equilibrium should raise energy."""
    atoms_eq = _h2(d=0.74)
    atoms_eq.calc = uma_calc
    e_eq = atoms_eq.get_potential_energy()

    atoms_comp = _h2(d=0.50)
    atoms_comp.calc = uma_calc
    e_comp = atoms_comp.get_potential_energy()

    assert e_comp > e_eq


def test_uma_energy_increases_when_stretched(uma_calc):
    """Stretching H2 well beyond equilibrium should raise energy."""
    atoms_eq = _h2(d=0.74)
    atoms_eq.calc = uma_calc
    e_eq = atoms_eq.get_potential_energy()

    atoms_str = _h2(d=3.0)
    atoms_str.calc = uma_calc
    e_str = atoms_str.get_potential_energy()

    assert e_str > e_eq


# ---------------------------------------------------------------------------
# Force–energy consistency (finite difference)
# ---------------------------------------------------------------------------

def test_uma_forces_consistent_with_energy(uma_calc):
    """Analytic forces should agree with finite-difference gradient to ~1%."""
    atoms = _h2(d=0.74)
    atoms.calc = uma_calc
    f_analytic = atoms.get_forces()

    dx = 1e-3
    # Perturb atom 1 along z (the bond axis)
    atoms_p = _h2(d=0.74 + dx)
    atoms_p.calc = uma_calc
    ep = atoms_p.get_potential_energy()

    atoms_m = _h2(d=0.74 - dx)
    atoms_m.calc = uma_calc
    em = atoms_m.get_potential_energy()

    f_numerical = -(ep - em) / (2 * dx)
    # atom 1 z-component
    np.testing.assert_allclose(f_analytic[1, 2], f_numerical, rtol=0.02)


# ---------------------------------------------------------------------------
# Symmetry invariances
# ---------------------------------------------------------------------------

def test_uma_translation_invariance(uma_calc):
    atoms = _feco()
    atoms.calc = uma_calc
    e1 = atoms.get_potential_energy()

    atoms2 = atoms.copy()
    atoms2.positions += [10.0, -5.0, 3.0]
    atoms2.calc = uma_calc
    e2 = atoms2.get_potential_energy()

    np.testing.assert_allclose(e1, e2, atol=1e-5)


def test_uma_rotation_invariance(uma_calc):
    atoms = _feco()
    atoms.calc = uma_calc
    e1 = atoms.get_potential_energy()

    atoms2 = atoms.copy()
    atoms2.rotate(90, "x", center="COP")
    atoms2.calc = uma_calc
    e2 = atoms2.get_potential_energy()

    np.testing.assert_allclose(e1, e2, atol=1e-5)


# ---------------------------------------------------------------------------
# Charge and spin sensitivity
# ---------------------------------------------------------------------------

def test_uma_charge_changes_energy(uma_calc):
    """H2 and H2+ should have different energies."""
    neutral = _h2(charge=0, spin=1)
    neutral.calc = uma_calc
    e_neutral = neutral.get_potential_energy()

    cation = _h2(charge=1, spin=2)
    cation.calc = uma_calc
    e_cation = cation.get_potential_energy()

    assert not np.isclose(e_neutral, e_cation, rtol=1e-3)


def test_uma_spin_changes_energy(uma_calc):
    """Different spin multiplicities should give different energies."""
    low_spin = _h2(charge=0, spin=1)
    low_spin.calc = uma_calc
    e_ls = low_spin.get_potential_energy()

    high_spin = _h2(charge=0, spin=3)
    high_spin.calc = uma_calc
    e_hs = high_spin.get_potential_energy()

    assert not np.isclose(e_ls, e_hs, rtol=1e-3)


# ---------------------------------------------------------------------------
# Calculator reuse across multiple atoms objects
# ---------------------------------------------------------------------------

def test_uma_calc_reusable(uma_calc):
    """The same calculator instance should handle multiple Atoms objects."""
    a1 = _h2(d=0.74)
    a1.calc = uma_calc
    e1 = a1.get_potential_energy()

    a2 = _feco()
    a2.calc = uma_calc
    e2 = a2.get_potential_energy()

    a3 = _h2(d=1.20)
    a3.calc = uma_calc
    e3 = a3.get_potential_energy()

    assert np.isfinite(e1)
    assert np.isfinite(e2)
    assert np.isfinite(e3)
    assert e1 != e3  # different geometries → different energies


# ---------------------------------------------------------------------------
# Diverse organometallic systems (OMol head coverage)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("symbol,positions,charge,spin", [
    ("Pd-CO",
     [[0, 0, 0], [0, 0, 1.9], [0, 0, 3.05]],
     0, 1),
    ("Rh-H",
     [[0, 0, 0], [0, 0, 1.6]],
     0, 2),
    ("Ni-C2H2",
     [[0, 0, 0], [0, 0, 1.9], [0, 0, 3.2], [0, 0.9, 3.8], [0, -0.9, 3.8]],
     0, 1),
])
def test_uma_organometallic_systems(uma_calc, symbol, positions, charge, spin):
    species = symbol.replace("-", "").replace("C2H2", "CCHH")
    # Derive chemical symbols from positions length
    sym_map = {"PdCO": "PdCO", "RhH": "RhH", "NiCCHH": "NiCCHH"}
    # Build from symbol string directly
    elem_str = {
        "Pd-CO": "PdCO",
        "Rh-H": "RhH",
        "Ni-C2H2": "NiCCHH",
    }[symbol]

    atoms = Atoms(elem_str, positions=positions)
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    atoms.calc = uma_calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert np.isfinite(energy), f"{symbol}: energy is not finite"
    assert forces.shape == (len(atoms), 3), f"{symbol}: wrong forces shape"
    assert np.all(np.isfinite(forces)), f"{symbol}: forces contain non-finite values"
