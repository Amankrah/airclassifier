"""
Rotor Geometry
==============

Hammer mill rotor assembly: main shaft with support discs.
The rotor rotates around the X axis, carrying the hammers.

Physical components:
    - Main shaft (cylinder along X)
    - Support discs (annular rings at regular intervals)
    - Hammer pin holes (conceptual - hammers pivot on pins through discs)

Coordinate system:
    X = along shaft axis (rotor length direction)
    Y = vertical (up)
    Z = lateral
    Origin at shaft centerline, X=0 at drive end
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import cylinder_mesh, disc_mesh, concat_meshes

if TYPE_CHECKING:
    from ...config import MillConfig


@dataclass
class RotorParams:
    """Rotor geometry parameters.

    All dimensions in meters. Positions are relative to the
    rotor coordinate system (shaft centerline at Y=Z=0).
    """

    # --- Shaft ---
    shaft_radius_m: float = 0.03                  # Main shaft radius
    shaft_length_m: float = 0.40                  # Total shaft length (includes bearing journals)
    active_length_m: float = 0.30                 # Length with hammers (RF zone)

    # --- Support discs ---
    disc_count: int = 5                           # Number of support discs
    disc_outer_radius_m: float = 0.08             # Disc outer radius
    disc_thickness_m: float = 0.010               # Disc thickness
    disc_start_x_m: float = 0.05                  # X position of first disc

    # --- Bearing journals (shaft extensions) ---
    journal_length_m: float = 0.04                # Length of each bearing journal
    journal_radius_m: float = 0.025               # Journal radius (slightly smaller than shaft)

    # --- Derived positions ---
    @property
    def disc_spacing_m(self) -> float:
        """Spacing between disc centers."""
        if self.disc_count <= 1:
            return 0.0
        return self.active_length_m / (self.disc_count - 1)

    @property
    def disc_positions_x(self) -> Tuple[float, ...]:
        """X positions of disc centers."""
        if self.disc_count == 1:
            return (self.disc_start_x_m + self.active_length_m / 2.0,)
        return tuple(
            self.disc_start_x_m + i * self.disc_spacing_m
            for i in range(self.disc_count)
        )

    @classmethod
    def from_mill_config(cls, config: "MillConfig") -> "RotorParams":
        """Create rotor params from mill configuration.

        Follows the parameter chain:
            MillConfig -> RotorParams
        """
        return cls(
            shaft_radius_m=config.shaft_diameter_m / 2.0,
            shaft_length_m=config.rotor_length_m + 0.10,  # Add bearing journals
            active_length_m=config.rotor_length_m,
            disc_count=config.hammer_rows + 1,  # One more disc than hammer rows
            disc_outer_radius_m=config.rotor_diameter_m / 2.0,
        )


class RotorGeometry:
    """Generates hammer mill rotor meshes.

    The rotor is the rotating assembly that carries the hammers.
    It consists of a central shaft with support discs.

    Animation:
        The rotor rotates around the X axis at angular velocity omega.
        The GUI applies rotation by angle theta = omega * t.
    """

    def __init__(self, params: Optional[RotorParams] = None):
        self.params = params or RotorParams()
        self._cached_verts: Optional[np.ndarray] = None
        self._cached_tris: Optional[np.ndarray] = None

    def generate_mesh(
        self,
        resolution: int = 24,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the complete rotor mesh.

        Args:
            resolution: Number of segments for cylindrical surfaces.

        Returns:
            (vertices, triangles, metadata)
        """
        p = self.params
        parts = []

        # --- Main shaft ---
        # Shaft runs along X from 0 to shaft_length
        shaft_verts, shaft_tris = cylinder_mesh(
            center=(0.0, 0.0, 0.0),
            radius=p.shaft_radius_m,
            height=p.shaft_length_m,
            resolution=resolution,
            axis="x",
        )
        parts.append((shaft_verts, shaft_tris))

        # --- Support discs ---
        for x_pos in p.disc_positions_x:
            disc_verts, disc_tris = disc_mesh(
                center=(x_pos, 0.0, 0.0),
                inner_radius=p.shaft_radius_m,
                outer_radius=p.disc_outer_radius_m,
                thickness=p.disc_thickness_m,
                resolution=resolution,
                axis="x",
            )
            parts.append((disc_verts, disc_tris))

        # --- Bearing journals (thinner shaft extensions) ---
        # Drive end journal (X < 0)
        journal_drive = cylinder_mesh(
            center=(-p.journal_length_m, 0.0, 0.0),
            radius=p.journal_radius_m,
            height=p.journal_length_m,
            resolution=resolution,
            axis="x",
        )
        parts.append(journal_drive)

        # Free end journal (X > shaft_length)
        journal_free = cylinder_mesh(
            center=(p.shaft_length_m, 0.0, 0.0),
            radius=p.journal_radius_m,
            height=p.journal_length_m,
            resolution=resolution,
            axis="x",
        )
        parts.append(journal_free)

        self._cached_verts, self._cached_tris = concat_meshes(parts)

        metadata = {
            "type": "rotor",
            "animation_type": "rotate",
            "animation_axis": "x",
            "pivot": (0.0, 0.0, 0.0),
            "shaft_radius_m": p.shaft_radius_m,
            "shaft_length_m": p.shaft_length_m,
            "disc_count": p.disc_count,
            "disc_positions_x": p.disc_positions_x,
            "disc_outer_radius_m": p.disc_outer_radius_m,
        }

        return self._cached_verts, self._cached_tris, metadata

    @property
    def ports(self) -> Dict[str, any]:
        """Rotor has no external connection ports."""
        return {}
