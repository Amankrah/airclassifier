"""
3D Field Visualization Helpers
==============================

Utilities for rendering pretreatment simulation fields in the
Air Classifier Designer's PyVista viewport.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def fields_to_pyvista_grid(
    grid_shape: Tuple[int, int, int],
    cell_sizes: Tuple[float, float, float],
    T: Optional[np.ndarray] = None,
    M: Optional[np.ndarray] = None,
    P_v: Optional[np.ndarray] = None,
):
    """Convert 3D field arrays to a PyVista UniformGrid for visualization.

    Args:
        grid_shape: (nx, ny, nz).
        cell_sizes: (dx, dy, dz) in metres.
        T: Temperature field [degC].
        M: Moisture field [wet basis fraction].
        P_v: RF power density [W/m^3].

    Returns:
        pyvista.UniformGrid with scalar arrays attached.
    """
    try:
        import pyvista as pv
    except ImportError:
        raise RuntimeError("PyVista is required for visualization")

    nx, ny, nz = grid_shape
    dx, dy, dz = cell_sizes

    grid = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=(dx, dy, dz),
    )

    if T is not None:
        grid.cell_data["Temperature [C]"] = T.flatten(order="F")
    if M is not None:
        grid.cell_data["Moisture [wb]"] = M.flatten(order="F")
    if P_v is not None:
        grid.cell_data["RF Power [W/m3]"] = P_v.flatten(order="F")

    return grid


def create_bed_slice(
    grid,
    axis: str = "y",
    origin: Optional[float] = None,
):
    """Create a cross-section slice through the material bed.

    Args:
        grid: PyVista grid from fields_to_pyvista_grid.
        axis: Slice normal axis ("x", "y", or "z").
        origin: Position along the axis. None = center.

    Returns:
        pyvista.PolyData slice.
    """
    try:
        import pyvista as pv
    except ImportError:
        raise RuntimeError("PyVista is required")

    bounds = grid.bounds
    if origin is None:
        if axis == "x":
            origin = (bounds[0] + bounds[1]) / 2
        elif axis == "y":
            origin = (bounds[2] + bounds[3]) / 2
        else:
            origin = (bounds[4] + bounds[5]) / 2

    normal = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    point = {"x": (origin, 0, 0), "y": (0, origin, 0), "z": (0, 0, origin)}[axis]

    return grid.slice(normal=normal, origin=point)
