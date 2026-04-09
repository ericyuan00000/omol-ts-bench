"""Interpolation methods for generating initial path between endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms


def linear_interpolation(
    reactant: Atoms,
    product: Atoms,
    n_images: int = 10,
) -> list[Atoms]:
    """Generate linearly interpolated images between reactant and product.

    Parameters
    ----------
    reactant
        Reactant endpoint geometry.
    product
        Product endpoint geometry.
    n_images
        Number of intermediate images (excluding endpoints).

    Returns
    -------
    list[Atoms]
        List of images including endpoints: [reactant, ..., product].
    """
    from ase.mep import NEB

    images = [reactant.copy()]
    for _ in range(n_images):
        images.append(reactant.copy())
    images.append(product.copy())

    neb = NEB(images)
    neb.interpolate()
    return images


def idpp_interpolation(
    reactant: Atoms,
    product: Atoms,
    n_images: int = 10,
) -> list[Atoms]:
    """Generate IDPP-interpolated images between reactant and product.

    Image Dependent Pair Potential (IDPP) interpolation provides a better
    initial path than linear interpolation by avoiding atomic clashes.

    Parameters
    ----------
    reactant
        Reactant endpoint geometry.
    product
        Product endpoint geometry.
    n_images
        Number of intermediate images (excluding endpoints).

    Returns
    -------
    list[Atoms]
        List of images including endpoints: [reactant, ..., product].
    """
    from ase.mep import NEB

    images = [reactant.copy()]
    for _ in range(n_images):
        images.append(reactant.copy())
    images.append(product.copy())

    neb = NEB(images)
    neb.interpolate(method="idpp")
    return images
