"""
Hammers Geometry
================

Hammer mill swinging hammers that impact the feed material.
Hammers are attached to pins that pass through the rotor discs
and can swing freely (or be locked at fixed angles).

Physical components:
    - Hammer bodies (rectangular plates with pivot boss)
    - Hammer pins (rods through rotor discs holding the hammers)

Coordinate system:
    X = along rotor axis
    Y = vertical (up)
    Z = lateral
    Hammers rotate with the rotor around X axis

Real-world anatomy:
    Hammer pins pass through holes in the rotor discs.  Hammers
    are threaded onto the pins between adjacent disc pairs.  At
    operating speed centrifugal force throws the hammers outward.
    Consecutive rows are angularly staggered for uniform coverage.

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

from ..mesh_utils import box_mesh, cylinder_mesh, concat_meshes, rotate_mesh_around_x

if TYPE_CHECKING:
    from ...config import MillConfig
    from .rotor import RotorParams


@dataclass
class HammerParams:
    """Hammer geometry parameters.

    Describes the swinging hammers, their pin attachments, and
    pivot boss geometry for realistic visualization.
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

    # --- Hammer pins (rods through rotor discs) ---
    pin_radius_m: float = 0.010                   # Pin rod radius
    pin_start_x_m: float = 0.04                   # Pin start X (first disc)
    pin_end_x_m: float = 0.30                     # Pin end X (last disc)

    # --- Pivot boss (mounting eye around pin hole) ---
    boss_radius_m: float = 0.018                  # Boss outer radius (> pin_radius)
    boss_length_m: float = 0.05                   # Boss length along X (~ hammer width)

    # --- Row stagger (angular offset between consecutive rows) ---
    stagger_angle_rad: float = 0.0                # 0 = no stagger

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

        pivot_r = config.rotor_diameter_m / 2.0 - 0.02

        # Pin proportional to hammer thickness
        pin_r = max(0.008, config.hammer_thickness_m * 1.2)
        boss_r = pin_r * 1.8

        # Stagger: half the angular spacing between hammers
        stagger = math.pi / config.hammers_per_row

        return cls(
            hammer_length_m=config.hammer_length_m,
            hammer_width_m=config.hammer_width_m,
            hammer_thickness_m=config.hammer_thickness_m,
            hammer_rows=config.hammer_rows,
            hammers_per_row=config.hammers_per_row,
            pivot_radius_m=pivot_r,
            row_start_x_m=margin + 0.05,
            row_spacing_m=spacing,
            pin_radius_m=pin_r,
            boss_radius_m=boss_r,
            boss_length_m=config.hammer_width_m,
            stagger_angle_rad=stagger,
        )

    @classmethod
    def from_rotor(cls, rotor_params: "RotorParams", config: "MillConfig") -> "HammerParams":
        """Create hammer params aligned to rotor geometry."""
        params = cls.from_mill_config(config)
        # Adjust pivot radius to be just outside rotor disc radius
        params.pivot_radius_m = rotor_params.disc_outer_radius_m - 0.01
        # Pins span from first disc to last disc
        disc_xs = rotor_params.disc_positions_x
        if disc_xs:
            params.pin_start_x_m = disc_xs[0]
            params.pin_end_x_m = disc_xs[-1]
        return params


class HammerGeometry:
    """Generates hammer mill hammer meshes.

    Creates individual hammer bodies with pivot bosses and hammer
    pins positioned around the rotor for realistic visualization.

    Animation:
        Hammers rotate with the rotor around the X axis.
        The animation_type is "rotate" with same theta as rotor.
    """

    def __init__(self, params: Optional[HammerParams] = None):
        self.params = params or HammerParams()
        self._cached_verts: Optional[np.ndarray] = None
        self._cached_tris: Optional[np.ndarray] = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate all hammer meshes (combined).

        Returns:
            (vertices, triangles, metadata)
        """
        parts_dict = self.generate_mesh_parts()
        all_parts = list(parts_dict.values())
        self._cached_verts, self._cached_tris = concat_meshes(all_parts)

        p = self.params
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

    def generate_mesh_parts(
        self,
        resolution: int = 12,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Generate hammer meshes as separate parts for color-coded visualization.

        Returns:
            Dict mapping part name to (vertices, triangles):
            - hammers: Hammer body plates with pivot bosses
            - hammer_pins: Pin rods through rotor discs
        """
        p = self.params
        hammer_parts: List[Tuple[np.ndarray, np.ndarray]] = []

        for row_idx, x_pos in enumerate(p.row_positions_x):
            # Stagger: each row offset by row_idx * stagger_angle
            row_stagger = row_idx * p.stagger_angle_rad

            for base_angle in p.angular_positions_rad:
                angle_rad = base_angle + row_stagger

                # Create hammer at Y+ position (angle=0), then rotate
                ham_verts, ham_tris = self._create_single_hammer(
                    p, x_pos, resolution
                )

                # Rotate hammer to its angular position around X axis
                if angle_rad != 0.0:
                    ham_verts, ham_tris = rotate_mesh_around_x(
                        ham_verts, ham_tris,
                        angle_rad,
                        pivot=(x_pos, 0.0, 0.0),
                    )

                hammer_parts.append((ham_verts, ham_tris))

        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        out["hammers"] = concat_meshes(hammer_parts)

        # Hammer pins (rods through rotor discs)
        out["hammer_pins"] = self._create_hammer_pins(p, resolution)

        return out

    def _create_single_hammer(
        self,
        p: HammerParams,
        x_center: float,
        resolution: int = 12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a single hammer body mesh with pivot boss.

        The hammer is created at angle=0 (extending in +Y direction).
        The boss is a short cylinder at the pivot point on the pin axis.
        """
        parts: List[Tuple[np.ndarray, np.ndarray]] = []

        # Hammer extends from pivot_radius + offset outward
        y_start = p.pivot_radius_m + p.pivot_offset_m

        # Main body plate
        x0 = x_center - p.hammer_width_m / 2.0
        y0 = y_start
        z0 = -p.hammer_thickness_m / 2.0
        parts.append(box_mesh(
            x0, y0, z0,
            p.hammer_width_m, p.hammer_length_m, p.hammer_thickness_m,
        ))

        # Pivot boss (short cylinder at pivot point, axis along X)
        boss = cylinder_mesh(
            center=(x_center - p.boss_length_m / 2, p.pivot_radius_m, 0.0),
            radius=p.boss_radius_m,
            height=p.boss_length_m,
            resolution=resolution,
            axis="x",
        )
        parts.append(boss)

        return concat_meshes(parts)

    def _create_hammer_pins(
        self,
        p: HammerParams,
        resolution: int = 12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create hammer pin cylinders.

        Pins run along X through the rotor disc holes at the pivot
        radius.  One pin per angular position, spanning from the
        first disc to the last disc.
        """
        pin_length = p.pin_end_x_m - p.pin_start_x_m
        if pin_length <= 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

        parts: List[Tuple[np.ndarray, np.ndarray]] = []

        for angle_rad in p.angular_positions_rad:
            py = p.pivot_radius_m * math.cos(angle_rad)
            pz = p.pivot_radius_m * math.sin(angle_rad)

            pin = cylinder_mesh(
                center=(p.pin_start_x_m, py, pz),
                radius=p.pin_radius_m,
                height=pin_length,
                resolution=resolution,
                axis="x",
            )
            parts.append(pin)

        return concat_meshes(parts)

    def get_hammer_positions(self) -> List[Tuple[float, float, float, float]]:
        """Get positions and angles of all hammers.

        Returns:
            List of (x, pivot_y, pivot_z, angle_rad) for each hammer.
        """
        p = self.params
        positions = []
        for row_idx, x_pos in enumerate(p.row_positions_x):
            row_stagger = row_idx * p.stagger_angle_rad
            for angle_rad in p.angular_positions_rad:
                total_angle = angle_rad + row_stagger
                # Pivot point in world coords
                pivot_y = p.pivot_radius_m * math.cos(total_angle)
                pivot_z = p.pivot_radius_m * math.sin(total_angle)
                positions.append((x_pos, pivot_y, pivot_z, total_angle))
        return positions

    @property
    def ports(self) -> Dict[str, any]:
        """Hammers have no external connection ports."""
        return {}
