"""
Milling Data Export
===================

Export functions for milling simulation results.
Supports CSV, JSON, and VTK formats.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..physics import MillingStepState
from ..simulator import MillingResult


def export_history_csv(
    history: List[MillingStepState],
    path: Path,
):
    """Export simulation history to CSV.

    Args:
        history: List of step states
        path: Output file path
    """
    if not history:
        return

    fieldnames = [
        "time_s",
        "rotor_theta_rad",
        "num_particles",
        "num_fed",
        "num_discharged",
        "holdup_kg",
        "feed_rate_kg_per_s",
        "discharge_rate_kg_per_s",
        "num_impacts",
        "total_impact_energy_j",
        "mean_impact_energy_j",
        "num_breakage_events",
        "mean_size_reduction",
        "num_fragments_created",
        "num_passed_screen",
        "screen_passage_rate",
        "d10_m",
        "d50_m",
        "d90_m",
        "power_kw",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for state in history:
            row = asdict(state)
            writer.writerow(row)


def export_psd_csv(
    size_classes: np.ndarray,
    mass_fractions: np.ndarray,
    path: Path,
):
    """Export particle size distribution to CSV.

    Args:
        size_classes: Size class boundaries [m]
        mass_fractions: Mass fractions per class
        path: Output file path
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["size_lower_m", "size_upper_m", "size_mid_um", "mass_fraction"])

        n = len(mass_fractions)
        for i in range(n):
            lower = size_classes[i]
            upper = size_classes[i + 1] if i + 1 < len(size_classes) else lower * 1.5
            mid_um = np.sqrt(lower * upper) * 1e6
            writer.writerow([lower, upper, mid_um, mass_fractions[i]])


def export_result_json(
    result: MillingResult,
    path: Path,
):
    """Export milling result to JSON.

    Args:
        result: Milling result
        path: Output file path
    """
    data = {
        "summary": {
            "d10_um": result.d10_um,
            "d50_um": result.d50_um,
            "d90_um": result.d90_um,
            "throughput_kg_per_hr": result.throughput_kg_per_hr,
            "mean_residence_time_s": result.mean_residence_time_s,
            "specific_energy_kwh_per_t": result.specific_energy_kwh_per_t,
            "mean_power_kw": result.mean_power_kw,
            "max_power_kw": result.max_power_kw,
            "psd_total_mass_kg": result.psd_total_mass_kg,
        },
        "psd": {
            "size_classes_m": result.psd_size_classes_m.tolist(),
            "mass_fractions": result.psd_mass_fractions.tolist(),
        },
        "history_length": len(result.history),
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def export_particles_vtk(
    positions: np.ndarray,
    sizes: np.ndarray,
    path: Path,
    velocities: Optional[np.ndarray] = None,
):
    """Export particle positions to VTK format.

    Args:
        positions: Particle positions [n, 3]
        sizes: Particle sizes [n]
        path: Output file path (.vtp)
        velocities: Optional particle velocities [n, 3]
    """
    n = len(positions)

    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="1.0">\n')
        f.write('<PolyData>\n')
        f.write(f'<Piece NumberOfPoints="{n}" NumberOfVerts="{n}">\n')

        # Points
        f.write('<Points>\n')
        f.write('<DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        for pos in positions:
            f.write(f"{pos[0]} {pos[1]} {pos[2]}\n")
        f.write('</DataArray>\n')
        f.write('</Points>\n')

        # Vertices
        f.write('<Verts>\n')
        f.write('<DataArray type="Int32" Name="connectivity" format="ascii">\n')
        for i in range(n):
            f.write(f"{i}\n")
        f.write('</DataArray>\n')
        f.write('<DataArray type="Int32" Name="offsets" format="ascii">\n')
        for i in range(1, n + 1):
            f.write(f"{i}\n")
        f.write('</DataArray>\n')
        f.write('</Verts>\n')

        # Point data
        f.write('<PointData>\n')

        # Sizes
        f.write('<DataArray type="Float32" Name="size" format="ascii">\n')
        for s in sizes:
            f.write(f"{s}\n")
        f.write('</DataArray>\n')

        # Velocities (optional)
        if velocities is not None:
            f.write('<DataArray type="Float32" Name="velocity" NumberOfComponents="3" format="ascii">\n')
            for vel in velocities:
                f.write(f"{vel[0]} {vel[1]} {vel[2]}\n")
            f.write('</DataArray>\n')

        f.write('</PointData>\n')

        f.write('</Piece>\n')
        f.write('</PolyData>\n')
        f.write('</VTKFile>\n')
