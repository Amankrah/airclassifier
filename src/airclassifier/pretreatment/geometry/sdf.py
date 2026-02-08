"""
Signed Distance Fields
======================

SDFs for oven boundaries, used for identifying material vs air cells
and applying boundary conditions in the physics solvers.

Negative inside the material bed, positive outside (air/belt).
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
    The distance is measured to the nearest bed boundary surface.

    Layout (Y-axis, bottom to top)::

        y = 0           lower electrode
        y = belt_stack   top of belt/wear-strip/top-sheet
        y = belt_stack + bed_depth   top of material bed
        y = gap          upper electrode

    Args:
        grid_shape: (nx, ny, nz).
        cell_sizes: (dx, dy, dz) in metres.
        electrode_gap_m: Current electrode gap.
        bed_depth_m: Material bed depth.
        belt_stack_m: Belt + wear strip + top sheet thickness.

    Returns:
        np.ndarray of shape grid_shape, dtype float32.
        Negative inside material, positive outside.
    """
    nx, ny, nz = grid_shape
    dx, dy, dz = cell_sizes

    sdf = np.zeros(grid_shape, dtype=np.float32)

    y_bed_bottom = belt_stack_m
    y_bed_top = belt_stack_m + bed_depth_m

    for j in range(ny):
        y = (j + 0.5) * (electrode_gap_m / ny)

        if y < y_bed_bottom:
            # Below bed — in belt layer, positive distance
            d = y_bed_bottom - y
        elif y > y_bed_top:
            # Above bed — in air gap, positive distance
            d = y - y_bed_top
        else:
            # Inside bed — negative distance to nearest surface
            d = -min(y - y_bed_bottom, y_bed_top - y)

        sdf[:, j, :] = d

    return sdf
