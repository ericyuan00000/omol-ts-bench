"""Tests for the validation module."""

import pytest
from ase import Atoms

from omol_ts_bench.core.validate import compare_atoms, validate_irc_endpoints


def test_compare_identical_molecules():
    """Identical molecules should match."""
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0, 0.93, -0.24]])
    assert compare_atoms(atoms, atoms) is True


def test_compare_translated_molecules():
    """Translated copies should have the same connectivity."""
    atoms1 = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0, 0.93, -0.24]])
    atoms2 = Atoms("H2O", positions=[[5, 5, 5], [5, 5, 5.96], [5, 5.93, 4.76]])
    assert compare_atoms(atoms1, atoms2) is True


def test_compare_different_connectivity():
    """Structures with different bonding should not match."""
    # H2O: O-H bonds at ~0.96 A
    water = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0, 0.93, -0.24]])
    # H...O...H: atoms far apart, no bonds
    separated = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 5.0], [0, 5.0, 0]])
    assert compare_atoms(water, separated) is False


def test_validate_irc_correct_assignment():
    """Validation should succeed when IRC endpoints match reactant/product."""
    reactant = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0, 0.93, -0.24]])
    product = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0, 0.93, -0.24]])
    # IRC endpoints are just shifted copies
    irc_fwd = Atoms("H2O", positions=[[1, 0, 0], [1, 0, 0.96], [1, 0.93, -0.24]])
    irc_rev = Atoms("H2O", positions=[[2, 0, 0], [2, 0, 0.96], [2, 0.93, -0.24]])

    result = validate_irc_endpoints(reactant, product, irc_fwd, irc_rev)
    assert result.both_match is True
