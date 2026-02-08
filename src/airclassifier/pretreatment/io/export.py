"""
Data Export
===========

Export simulation results in VTK, CSV, and NumPy formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np


def export_vtk(
    filepath: str,
    grid_shape: tuple,
    cell_sizes: tuple,
    fields: Dict[str, np.ndarray],
):
    """Export 3D fields as a VTK structured grid.

    Args:
        filepath: Output path (.vts or .vtk).
        grid_shape: (nx, ny, nz).
        cell_sizes: (dx, dy, dz) in metres.
        fields: Dict mapping field names to 3D arrays.
    """
    # TODO: Implement VTK export using vtk or pyvista
    raise NotImplementedError


def export_csv_timeseries(
    filepath: str,
    time_series: Dict[str, list],
):
    """Export time-series KPIs as CSV.

    Args:
        filepath: Output path (.csv).
        time_series: Dict mapping column names to lists of values.
    """
    # TODO: Implement CSV export
    raise NotImplementedError


def export_numpy_snapshot(
    directory: str,
    time_s: float,
    T: np.ndarray,
    M: np.ndarray,
    P_v: Optional[np.ndarray] = None,
):
    """Save field snapshots as .npy files.

    Args:
        directory: Output directory.
        time_s: Simulation time [s].
        T: Temperature field.
        M: Moisture field.
        P_v: Optional RF power density field.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"t{time_s:.1f}"
    np.save(out / f"T_{tag}.npy", T)
    np.save(out / f"M_{tag}.npy", M)
    if P_v is not None:
        np.save(out / f"Pv_{tag}.npy", P_v)
