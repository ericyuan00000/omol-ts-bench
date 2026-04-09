"""Sella saddle-point optimization for TS refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator


@dataclass
class SellaResult:
    """Result of a Sella TS optimization."""

    converged: bool
    ts_optimized: Atoms | None
    """Optimized transition state geometry."""
    ts_energy: float | None
    """Energy of the optimized TS in eV."""
    imaginary_freq: float | None
    """Imaginary frequency in cm^-1 (negative value)."""
    n_steps: int = 0


def run_sella(
    ts_guess: Atoms,
    calc: Calculator,
    fmax: float = 0.02,
    max_steps: int = 500,
    order: int = 1,
    delta: float = 0.01,
    gamma: float = 0.4,
) -> SellaResult:
    """Refine a TS guess using Sella's saddle-point optimizer.

    Parameters
    ----------
    ts_guess
        Initial TS geometry (e.g., from NEB or GSM).
    calc
        ASE calculator for energy/force evaluations.
    fmax
        Force convergence criterion in eV/A.
    max_steps
        Maximum number of optimization steps.
    order
        Saddle point order (1 for transition states).
    delta
        Finite difference step for Hessian update.
    gamma
        Trust radius parameter.

    Returns
    -------
    SellaResult
        Result dataclass containing optimized TS and vibrational info.
    """
    from sella import Sella

    atoms = ts_guess.copy()
    atoms.calc = calc

    opt = Sella(atoms, order=order, delta0=delta, gamma=gamma)

    try:
        converged = opt.run(fmax=fmax, steps=max_steps)
    except Exception:
        return SellaResult(
            converged=False, ts_optimized=None, ts_energy=None,
            imaginary_freq=None,
        )

    ts_energy = atoms.get_potential_energy()

    # Extract imaginary frequency from Sella's internal Hessian
    imaginary_freq = None
    if hasattr(opt, "H") and opt.H is not None:
        import numpy as np
        eigenvalues = np.linalg.eigvalsh(opt.H.todense() if hasattr(opt.H, "todense") else opt.H)
        neg_eigs = eigenvalues[eigenvalues < 0]
        if len(neg_eigs) > 0:
            # Convert to wavenumber (approximate)
            imaginary_freq = float(neg_eigs[0])

    return SellaResult(
        converged=converged,
        ts_optimized=atoms.copy(),
        ts_energy=ts_energy,
        imaginary_freq=imaginary_freq,
        n_steps=opt.nsteps,
    )
