"""Tests for interpolation methods."""

import numpy as np
import pytest
from ase import Atoms

from omol_ts_bench.core.interpolation import linear_interpolation


def test_linear_interpolation_endpoints():
    """Linear interpolation should preserve endpoints."""
    reactant = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    product = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.5]])

    images = linear_interpolation(reactant, product, n_images=5)

    assert len(images) == 7  # 5 + 2 endpoints
    np.testing.assert_allclose(images[0].positions, reactant.positions, atol=1e-10)
    np.testing.assert_allclose(images[-1].positions, product.positions, atol=1e-10)


def test_linear_interpolation_monotonic():
    """Intermediate images should interpolate monotonically."""
    reactant = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    product = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.5]])

    images = linear_interpolation(reactant, product, n_images=5)

    # H-H distance should increase monotonically
    distances = [img.get_distance(0, 1) for img in images]
    for i in range(len(distances) - 1):
        assert distances[i] <= distances[i + 1]
