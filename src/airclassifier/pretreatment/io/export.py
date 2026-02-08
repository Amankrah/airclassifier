"""
Data Export
===========

Export simulation results in VTK, CSV, and NumPy formats.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np


def export_vtk(
    filepath: str,
    grid_shape: tuple,
    cell_sizes: tuple,
    fields: Dict[str, np.ndarray],
):
    """Export 3D fields as a VTK structured grid (.vts).

    Uses PyVista if available, otherwise falls back to manual VTK
    legacy format (.vtk).

    Args:
        filepath: Output path (.vts or .vtk).
        grid_shape: (nx, ny, nz).
        cell_sizes: (dx, dy, dz) in metres.
        fields: Dict mapping field names to 3D arrays.
    """
    nx, ny, nz = grid_shape
    dx, dy, dz = cell_sizes

    try:
        import pyvista as pv

        grid = pv.ImageData(
            dimensions=(nx + 1, ny + 1, nz + 1),
            spacing=(dx, dy, dz),
            origin=(0.0, 0.0, 0.0),
        )
        for name, arr in fields.items():
            grid.cell_data[name] = arr.ravel(order="F")
        grid.save(filepath)

    except ImportError:
        # Fallback: write VTK legacy structured points
        with open(filepath, "w") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Pretreatment simulation export\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")
            f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
            f.write(f"ORIGIN 0.0 0.0 0.0\n")
            f.write(f"SPACING {dx} {dy} {dz}\n")
            f.write(f"POINT_DATA {nx * ny * nz}\n")
            for name, arr in fields.items():
                f.write(f"SCALARS {name} float\n")
                f.write("LOOKUP_TABLE default\n")
                for val in arr.ravel(order="F"):
                    f.write(f"{val:.6g}\n")


def export_csv_timeseries(
    filepath: str,
    time_series: Dict[str, list],
):
    """Export time-series KPIs as CSV.

    Args:
        filepath: Output path (.csv).
        time_series: Dict mapping column names to lists of values.
    """
    if not time_series:
        return

    columns = list(time_series.keys())
    n_rows = len(next(iter(time_series.values())))

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i in range(n_rows):
            row = [time_series[col][i] for col in columns]
            writer.writerow(row)


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
