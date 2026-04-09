"""NEB (Nudged Elastic Band) transition state search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator


@dataclass
class NEBResult:
    """Result of a NEB calculation."""

    converged: bool
    ts_guess: Atoms | None
    """Highest-energy image (TS guess)."""
    ts_energy: float | None
    """Energy of the TS guess in eV."""
    barrier_forward: float | None
    """Forward barrier in eV."""
    barrier_reverse: float | None
    """Reverse barrier in eV."""
    n_steps: int = 0
    images: list[Atoms] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)


def run_neb(
    reactant: Atoms,
    product: Atoms,
    calc: Calculator,
    n_images: int = 10,
    fmax: float = 0.05,
    max_steps: int = 500,
    climb: bool = True,
    method: str = "aseneb",
    interpolation: str = "idpp",
    optimizer: str = "BFGS",
    optimizer_kwargs: dict | None = None,
) -> NEBResult:
    """Run a NEB calculation between two endpoints.

    Parameters
    ----------
    reactant
        Reactant endpoint geometry.
    product
        Product endpoint geometry.
    calc
        ASE calculator to use for energy/force evaluations.
    n_images
        Number of intermediate images.
    fmax
        Force convergence criterion in eV/A.
    max_steps
        Maximum number of optimizer steps.
    climb
        Whether to use climbing-image NEB.
    method
        NEB method: "aseneb" or "improvedtangent".
    interpolation
        Interpolation method: "linear" or "idpp".
    optimizer
        ASE optimizer class name.
    optimizer_kwargs
        Additional keyword arguments for the optimizer.

    Returns
    -------
    NEBResult
        Result dataclass containing TS guess and energetics.
    """
    import ase.optimize
    import numpy as np
    from ase.mep import NEB

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    # Build images
    images = [reactant.copy()]
    for _ in range(n_images):
        images.append(reactant.copy())
    images.append(product.copy())

    # Attach calculators to intermediate images
    for image in images[1:-1]:
        image.calc = calc

    # Also need calculators on endpoints for energy evaluation
    images[0].calc = calc
    images[-1].calc = calc

    # Set up NEB
    neb = NEB(images, climb=climb, method=method)
    neb.interpolate(method=interpolation)

    # Run optimization
    opt_class = getattr(ase.optimize, optimizer)
    opt = opt_class(neb, **optimizer_kwargs)

    try:
        converged = opt.run(fmax=fmax, steps=max_steps)
    except Exception:
        return NEBResult(converged=False, ts_guess=None, ts_energy=None,
                         barrier_forward=None, barrier_reverse=None)

    # Extract energies
    energies = [image.get_potential_energy() for image in images]
    imax = int(np.argmax(energies))

    return NEBResult(
        converged=converged,
        ts_guess=images[imax].copy(),
        ts_energy=energies[imax],
        barrier_forward=energies[imax] - energies[0],
        barrier_reverse=energies[imax] - energies[-1],
        n_steps=opt.nsteps,
        images=[img.copy() for img in images],
        energies=energies,
    )
