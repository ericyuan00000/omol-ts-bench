"""Intrinsic Reaction Coordinate (IRC) calculations using Sella."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator


@dataclass
class IRCResult:
    """Result of an IRC calculation."""

    converged_forward: bool
    converged_reverse: bool
    endpoint_forward: Atoms | None
    """IRC endpoint in the forward direction."""
    endpoint_reverse: Atoms | None
    """IRC endpoint in the reverse direction."""
    path_forward: list[Atoms] | None
    """Full IRC path in forward direction."""
    path_reverse: list[Atoms] | None
    """Full IRC path in reverse direction."""
    n_steps_forward: int = 0
    n_steps_reverse: int = 0


def run_irc(
    ts: Atoms,
    calc: Calculator,
    dx: float = 0.1,
    max_steps: int = 200,
    fmax: float = 0.05,
    direction: str = "both",
    irctol: float = 0.01,
    eta: float = 1e-4,
    gamma: float = 0.1,
    trajectory_prefix: str | None = None,
) -> IRCResult:
    """Run IRC from an optimized transition state using Sella.

    Uses Sella's IRC implementation, which follows the minimum energy
    path via the stabilized Euler predictor-corrector method.

    Parameters
    ----------
    ts
        Optimized transition state geometry.
    calc
        ASE calculator for energy/force evaluations.
    dx
        IRC step size in mass-weighted coordinates (amu^(1/2) * Angstrom).
    max_steps
        Maximum number of IRC steps per direction.
    fmax
        Force convergence criterion in eV/A.
    direction
        "both", "forward", or "reverse".
    irctol
        Tolerance for IRC corrector convergence.
    eta
        Regularization parameter for the RFO step.
    gamma
        Trust radius parameter.
    trajectory_prefix
        If provided, save trajectories to
        ``{prefix}_forward.traj`` / ``{prefix}_reverse.traj``.

    Returns
    -------
    IRCResult
        Result dataclass containing IRC endpoints and paths.
    """
    from ase.io import Trajectory
    from sella import IRC

    directions_to_run = ["forward", "reverse"] if direction == "both" else [direction]

    results = {
        "forward": {"converged": False, "endpoint": None, "path": None, "n_steps": 0},
        "reverse": {"converged": False, "endpoint": None, "path": None, "n_steps": 0},
    }

    for d in directions_to_run:
        atoms = ts.copy()
        atoms.calc = calc

        traj_file = f"{trajectory_prefix}_{d}.traj" if trajectory_prefix else None

        irc = IRC(
            atoms,
            trajectory=traj_file,
            dx=dx,
            irctol=irctol,
            eta=eta,
            gamma=gamma,
            logfile="-",
        )

        try:
            converged = irc.run(fmax=fmax, steps=max_steps, direction=d)
        except Exception:
            converged = False

        results[d]["converged"] = converged
        results[d]["endpoint"] = atoms.copy()
        results[d]["n_steps"] = irc.nsteps

        # Read back trajectory if saved
        if traj_file:
            try:
                results[d]["path"] = list(Trajectory(traj_file))
            except Exception:
                results[d]["path"] = None

    return IRCResult(
        converged_forward=results["forward"]["converged"],
        converged_reverse=results["reverse"]["converged"],
        endpoint_forward=results["forward"]["endpoint"],
        endpoint_reverse=results["reverse"]["endpoint"],
        path_forward=results["forward"]["path"],
        path_reverse=results["reverse"]["path"],
        n_steps_forward=results["forward"]["n_steps"],
        n_steps_reverse=results["reverse"]["n_steps"],
    )
