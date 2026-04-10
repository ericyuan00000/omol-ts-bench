"""Tests for run_neb using LennardJones (no UMA required)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones

from omol_ts_bench.core.neb import NEBResult, run_neb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ar2_endpoints():
    """Ar2: bonded (r=1.2 Å) → dissociated (r=3.5 Å). LJ has a barrier here."""
    reactant = Atoms("Ar2", positions=[[0, 0, 0], [0, 0, 1.2]])
    product = Atoms("Ar2", positions=[[0, 0, 0], [0, 0, 3.5]])
    return reactant, product


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_neb_returns_neb_result(ar2_endpoints):
    reactant, product = ar2_endpoints
    result = run_neb(reactant, product, LennardJones(), n_images=3,
                     max_steps=50, climb=False, interpolation="linear")
    assert isinstance(result, NEBResult)


def test_neb_converged(ar2_endpoints):
    reactant, product = ar2_endpoints
    result = run_neb(reactant, product, LennardJones(), n_images=3,
                     max_steps=100, climb=False, interpolation="linear")
    assert result.converged


def test_neb_ts_guess_is_atoms(ar2_endpoints):
    reactant, product = ar2_endpoints
    result = run_neb(reactant, product, LennardJones(), n_images=3,
                     max_steps=50, climb=False, interpolation="linear")
    assert result.ts_guess is not None
    assert len(result.ts_guess) == len(reactant)


def test_neb_image_count(ar2_endpoints):
    reactant, product = ar2_endpoints
    n_images = 4
    result = run_neb(reactant, product, LennardJones(), n_images=n_images,
                     max_steps=50, climb=False, interpolation="linear")
    # images includes endpoints
    assert len(result.images) == n_images + 2


def test_neb_ts_energy_is_highest(ar2_endpoints):
    """The TS guess should be the highest-energy image."""
    reactant, product = ar2_endpoints
    result = run_neb(reactant, product, LennardJones(), n_images=3,
                     max_steps=50, climb=False, interpolation="linear")
    assert result.ts_energy is not None
    assert result.ts_energy == pytest.approx(max(result.energies), abs=1e-10)


def test_neb_barrier_forward_positive(ar2_endpoints):
    """Forward barrier = E_ts - E_reactant, should be > 0 for LJ dissociation."""
    reactant, product = ar2_endpoints
    result = run_neb(reactant, product, LennardJones(), n_images=3,
                     max_steps=50, climb=False, interpolation="linear")
    assert result.barrier_forward is not None
    assert result.barrier_forward > 0


def test_neb_n_steps_positive(ar2_endpoints):
    reactant, product = ar2_endpoints
    result = run_neb(reactant, product, LennardJones(), n_images=3,
                     max_steps=50, climb=False, interpolation="linear")
    assert result.n_steps > 0


def test_neb_max_steps_limit(ar2_endpoints):
    """With max_steps=1, NEB should not converge but still return a result."""
    reactant, product = ar2_endpoints
    result = run_neb(reactant, product, LennardJones(), n_images=3,
                     max_steps=1, climb=False, interpolation="linear")
    assert isinstance(result, NEBResult)
    # May or may not converge in 1 step, but should not raise
    assert result.ts_energy is not None


def test_neb_energies_finite(ar2_endpoints):
    reactant, product = ar2_endpoints
    result = run_neb(reactant, product, LennardJones(), n_images=3,
                     max_steps=50, climb=False, interpolation="linear")
    assert all(np.isfinite(e) for e in result.energies)
