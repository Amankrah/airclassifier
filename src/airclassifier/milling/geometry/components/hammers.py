"""
Hammers Geometry
================

Hammer mill swinging hammers that impact the feed material.
Hammers are attached to pins that pass through the rotor discs
and can swing freely (or be locked at fixed angles).

Physical components:
    - Hammer bodies (rectangular plates)
    - Hammer pins (not rendered - conceptual pivot points)

Coordinate system:
    X = along rotor axis
    Y = vertical (up)
    Z = lateral
    Hammers rotate with the rotor around X axis

Animation:
    Hammers rotate with the rotor. At high speed they are thrown
    outward by centrifugal force. For visualization, we show them
    at their extended position (radially outward from shaft).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import box_mesh, concat_meshes, rotate_mesh_around_x

if TYPE_CHECKING:
    from ...config import MillConfig
    from .rotor import RotorParams


@dataclass
class HammerParams:
    """Hammer geometry parameters.

    Describes the swinging hammers attached to the rotor.
    """

    # --- Hammer dimensions ---
    hammer_length_m: float = 0.10                 # From pivot to tip
    hammer_width_m: float = 0.05                  # Width along rotor axis
    hammer_thickness_m: float = 0.008             # Plate thickness

    # --- Arrangement ---
    hammer_rows: int = 4                          # Rows along rotor length
    hammers_per_row: int = 4                      # Hammers per row (evenly spaced angularly)

    # --- Pivot geometry ---
    pivot_radius_m: float = 0.06                  # Radius from shaft to hammer pivot
    pivot_offset_m: float = 0.005                 # Gap between pivot and hammer body

    # --- Row positions (X coordinates) ---
    row_start_x_m: float = 0.08                   # X position of first hammer row
    row_spacing_m: float = 0.06                   # Spacing between rows

    # --- Swing angle (for visualization) ---
    # At operating speed, hammers are thrown outward
    swing_angle_deg: float = 0.0                  # 0 = fully extended radially

    @property
    def row_positions_x(self) -> Tuple[float, ...]:
        """X positions of hammer rows."""
        return tuple(
            self.row_start_x_m + i * self.row_spacing_m
            for i in range(self.hammer_rows)
        )

    @property
    def angular_positions_rad(self) -> Tuple[float, ...]:
        """Angular positions of hammers in each row [rad]."""
        return tuple(
            i * 2.0 * math.pi / self.hammers_per_row
            for i in range(self.hammers_per_row)
        )

    @property
    def tip_radius_m(self) -> float:
        """Radius from shaft center to hammer tip."""
        return self.pivot_radius_m + self.pivot_offset_m + self.hammer_length_m

    @classmethod
    def from_mill_config(cls, config: "MillConfig") -> "HammerParams":
        """Create hammer params from mill configuration."""
        # Compute row spacing to fit within active length
        active_length = config.rotor_length_m
        num_rows = config.hammer_rows
        margin = 0.04  # 4cm margin at each end
        usable_length = active_length - 2 * margin
        spacing = usable_length / max(num_rows - 1, 1) if num_rows > 1 else 0.0

        return cls(
            hammer_length_m=config.hammer_length_m,
            hammer_width_m=config.hammer_width_m,
            hammer_thickness_m=config.hammer_thickness_m,
            hammer_rows=config.hammer_rows,
            hammers_per_row=config.hammers_per_row,
            pivot_radius_m=config.rotor_diameter_m / 2.0 - 0.02,
            row_start_x_m=margin + 0.05,  # Offset into housing
            row_spacing_m=spacing,
        )

    @classmethod
    def from_rotor(cls, rotor_params: "RotorParams", config: "MillConfig") -> "HammerParams":
        """Create hammer params aligned to rotor geometry."""
        params = cls.from_mill_config(config)
        # Adjust pivot radius to be just outside rotor disc radius
        params.pivot_radius_m = rotor_params.disc_outer_radius_m - 0.01
        return params


class HammerGeometry:
    """Generates hammer mill hammer meshes.

    Creates individual hammer bodies positioned around the rotor.
    Each hammer is a rectangular plate extending radially outward
    from its pivot point.

    Animation:
        Hammers rotate with the rotor around the X axis.
        The animation_type is "rotate" with same theta as rotor.
    """

    def __init__(self, params: Optional[HammerParams] = None):
        self.params = params or HammerParams()
        self._cached_verts: Optional[np.ndarray] = None
        self._cached_tris: Optional[np.ndarray] = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate all hammer meshes.

        Returns:
            (vertices, triangles, metadata)
        """
        p = self.params
        parts = []

        for row_idx, x_pos in enumerate(p.row_positions_x):
            for ham_idx, angle_rad in enumerate(p.angular_positions_rad):
                # Create hammer at Y+ position (radially outward at angle=0)
                # Then rotate to its angular position

                # Hammer body: origin at pivot, extends in +Y direction
                # Local coords: hammer body from pivot_offset to pivot_offset + length
                ham_verts, ham_tris = self._create_single_hammer(p, x_pos)

                # Rotate hammer to its angular position around X axis
                if angle_rad != 0.0:
                    ham_verts, ham_tris = rotate_mesh_around_x(
                        ham_verts, ham_tris,
                        angle_rad,
                        pivot=(x_pos, 0.0, 0.0)
                    )

                parts.append((ham_verts, ham_tris))

        self._cached_verts, self._cached_tris = concat_meshes(parts)

        metadata = {
            "type": "hammers",
            "animation_type": "rotate",
            "animation_axis": "x",
            "pivot": (0.0, 0.0, 0.0),
            "hammer_rows": p.hammer_rows,
            "hammers_per_row": p.hammers_per_row,
            "total_hammers": p.hammer_rows * p.hammers_per_row,
            "tip_radius_m": p.tip_radius_m,
            "row_positions_x": p.row_positions_x,
        }

        return self._cached_verts, self._cached_tris, metadata

    def _create_single_hammer(
        self,
        p: HammerParams,
        x_center: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a single hammer body mesh.

        The hammer is created at angle=0 (extending in +Y direction).
        """
        # Hammer extends from pivot_radius + offset outward
        y_start = p.pivot_radius_m + p.pivot_offset_m
        y_end = y_start + p.hammer_length_m

        # Box centered on x_center, extending in Y from y_start to y_end
        x0 = x_center - p.hammer_width_m / 2.0
        y0 = y_start
        z0 = -p.hammer_thickness_m / 2.0

        return box_mesh(x0, y0, z0, p.hammer_width_m, p.hammer_length_m, p.hammer_thickness_m)

    def get_hammer_positions(self) -> List[Tuple[float, float, float, float]]:
        """Get positions and angles of all hammers.

        Returns:
            List of (x, pivot_y, pivot_z, angle_rad) for each hammer.
        """
        p = self.params
        positions = []
        for x_pos in p.row_positions_x:
            for angle_rad in p.angular_positions_rad:
                # Pivot point in world coords
                pivot_y = p.pivot_radius_m * math.cos(angle_rad)
                pivot_z = p.pivot_radius_m * math.sin(angle_rad)
                positions.append((x_pos, pivot_y, pivot_z, angle_rad))
        return positions

    @property
    def ports(self) -> Dict[str, any]:
        """Hammers have no external connection ports."""
        return {}
