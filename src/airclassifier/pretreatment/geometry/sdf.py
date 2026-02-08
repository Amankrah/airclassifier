"""
Signed Distance Fields
======================

SDFs for oven boundaries, used for identifying material vs air cells
and applying boundary conditions in the physics solvers.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def oven_sdf(
    grid_shape: Tuple[int, int, int],
    cell_sizes: Tuple[float, float, float],
    electrode_gap_m: float,
    bed_depth_m: float,
    belt_stack_m: float,
) -> np.ndarray:
    """Compute signed distance field for the oven domain.

    Negative inside the material bed, positive outside.

    Args:
        grid_shape: (nx, ny, nz).
        cell_sizes: (dx, dy, dz) in metres.
        electrode_gap_m: Current electrode gap.
        bed_depth_m: Material bed depth.
        belt_stack_m: Total thickness of belt + wear strips + top sheet.

    Returns:
        np.ndarray of shape grid_shape, dtype float32.
    """
    # TODO: Implement SDF computation
    raise NotImplementedError
