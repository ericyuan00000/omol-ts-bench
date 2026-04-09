"""Growing String Method (GSM) transition state search via pyGSM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator


@dataclass
class GSMResult:
    """Result of a GSM calculation."""

    converged: bool
    ts_guess: Atoms | None
    """Highest-energy node (TS guess)."""
    ts_energy: float | None
    """Energy of the TS guess in eV."""
    barrier_forward: float | None
    barrier_reverse: float | None
    n_nodes: int = 0


def run_de_gsm(
    reactant: Atoms,
    product: Atoms,
    calc: Calculator,
    n_nodes: int = 11,
    max_steps: int = 500,
    conv_tol: float = 0.05,
) -> GSMResult:
    """Run double-ended Growing String Method.

    Parameters
    ----------
    reactant
        Reactant endpoint geometry.
    product
        Product endpoint geometry.
    calc
        ASE calculator for energy/force evaluations.
    n_nodes
        Number of nodes along the string.
    max_steps
        Maximum optimization steps.
    conv_tol
        Convergence tolerance in eV/A.

    Returns
    -------
    GSMResult
        Result dataclass containing TS guess and energetics.
    """
    # NOTE: pyGSM integration is calculator-specific.
    # This is a placeholder for the ASE-pyGSM bridge.
    # Users should implement the appropriate pyGSM Lot (Level of Theory)
    # wrapper for their calculator.
    raise NotImplementedError(
        "pyGSM integration requires a calculator-specific Lot wrapper. "
        "See the pyGSM documentation for details on implementing a custom Lot class. "
        "An ASE-based Lot adapter is planned for a future release."
    )
