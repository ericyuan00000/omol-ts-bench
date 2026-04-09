"""Validation of IRC endpoints against intended reactants/products using OpenBabel.

Compares molecular connectivity (neighbor lists) to determine whether
two structures represent the same molecule. This approach checks whether
the bond connectivity matrices are identical.

NOTE: The current implementation uses a simple connectivity matrix
comparison that works for organic molecules. For organometallic systems,
a more sophisticated comparison (e.g., accounting for metal coordination
environments, haptic bonding, etc.) may be needed. This is a placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms


@dataclass
class ValidationResult:
    """Result of comparing IRC endpoints to intended reactant/product."""

    reactant_match: bool
    """Whether one IRC endpoint matches the intended reactant."""
    product_match: bool
    """Whether one IRC endpoint matches the intended product."""
    both_match: bool
    """Whether both endpoints match (reaction is correctly connected)."""


def atoms_to_obmol(atoms: Atoms):
    """Convert ASE Atoms to an OpenBabel OBMol.

    Parameters
    ----------
    atoms
        ASE Atoms object.

    Returns
    -------
    openbabel.OBMol
        OpenBabel molecule with connectivity perceived from geometry.
    """
    from openbabel import openbabel as ob

    mol = ob.OBMol()

    for atom in atoms:
        ob_atom = mol.NewAtom()
        ob_atom.SetAtomicNum(int(atom.number))
        pos = atom.position
        ob_atom.SetVector(float(pos[0]), float(pos[1]), float(pos[2]))

    mol.ConnectTheDots()
    mol.PerceiveBondOrders()

    return mol


def get_neighbor_list(molecule) -> np.ndarray:
    """Get the connectivity matrix from an OpenBabel molecule.

    Parameters
    ----------
    molecule
        OpenBabel OBMol object.

    Returns
    -------
    np.ndarray
        Symmetric binary connectivity matrix of shape (n_atoms, n_atoms).
    """
    from openbabel import openbabel

    n = molecule.NumAtoms()
    connectivity = np.zeros((n, n), dtype=int)
    for bond in openbabel.OBMolBondIter(molecule):
        connectivity[bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1] = 1
        connectivity[bond.GetEndAtomIdx() - 1, bond.GetBeginAtomIdx() - 1] = 1
    return connectivity


def compare_molecules(molecule1, molecule2) -> bool:
    """Compare two OpenBabel molecules by connectivity.

    Parameters
    ----------
    molecule1
        First OpenBabel OBMol.
    molecule2
        Second OpenBabel OBMol.

    Returns
    -------
    bool
        True if the connectivity matrices are identical.
    """
    return bool((get_neighbor_list(molecule1) == get_neighbor_list(molecule2)).all())


def compare_atoms(atoms1: Atoms, atoms2: Atoms) -> bool:
    """Compare two ASE Atoms objects by molecular connectivity.

    Converts both to OpenBabel molecules, perceives bonds from
    geometry, and checks whether the connectivity matrices match.

    Parameters
    ----------
    atoms1
        First structure.
    atoms2
        Second structure.

    Returns
    -------
    bool
        True if the connectivity matrices are identical.
    """
    mol1 = atoms_to_obmol(atoms1)
    mol2 = atoms_to_obmol(atoms2)
    return compare_molecules(mol1, mol2)


def validate_irc_endpoints(
    reactant: Atoms,
    product: Atoms,
    irc_forward: Atoms,
    irc_reverse: Atoms,
) -> ValidationResult:
    """Validate that IRC endpoints match intended reactant and product.

    Compares molecular connectivity matrices. Tries both assignments
    (forward=reactant, reverse=product) and vice versa, since the
    IRC direction is arbitrary.

    Parameters
    ----------
    reactant
        Intended reactant geometry.
    product
        Intended product geometry.
    irc_forward
        IRC endpoint in forward direction.
    irc_reverse
        IRC endpoint in reverse direction.

    Returns
    -------
    ValidationResult
        Validation result indicating which endpoints matched.
    """
    mol_r = atoms_to_obmol(reactant)
    mol_p = atoms_to_obmol(product)
    mol_fwd = atoms_to_obmol(irc_forward)
    mol_rev = atoms_to_obmol(irc_reverse)

    # Try assignment 1: forward=reactant, reverse=product
    assign1_r = compare_molecules(mol_fwd, mol_r)
    assign1_p = compare_molecules(mol_rev, mol_p)

    # Try assignment 2: forward=product, reverse=reactant
    assign2_r = compare_molecules(mol_rev, mol_r)
    assign2_p = compare_molecules(mol_fwd, mol_p)

    if (assign1_r and assign1_p) or (assign2_r and assign2_p):
        return ValidationResult(
            reactant_match=True,
            product_match=True,
            both_match=True,
        )

    return ValidationResult(
        reactant_match=assign1_r or assign2_r,
        product_match=assign1_p or assign2_p,
        both_match=False,
    )
