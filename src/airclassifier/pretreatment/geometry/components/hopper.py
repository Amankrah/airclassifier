"""
Infeed Hopper Geometry
======================

Material infeed hopper with slanted feed plate and sizing gate
for controlling bed depth on the conveyor belt.

Based on real GP-15 machine dimensions:
- Slant length: ~70cm (0.70m)
- Width: ~35cm (0.35m)
- Opening height: ~25cm (0.255m)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import concat_meshes

if TYPE_CHECKING:
    from airclassifier.geometry.connection_ports import ConnectionPort


@dataclass
class InfeedHopperParams:
    """Infeed hopper parameters based on real GP-15 machine."""

    # From machine image measurements
    slant_length: float = 0.70      # [m] Length of slanted feed plate (69.7cm)
    width: float = 0.35             # [m] Width across belt direction Z (35.3cm)
    opening_height: float = 0.255   # [m] Height at belt entry (25.5cm)

    # Slant angle (from vertical, typical ~50-60 degrees)
    slant_angle_deg: float = 55.0   # degrees from vertical

    # Material thickness
    plate_thickness: float = 0.003  # [m] Stainless steel plate ~3mm

    # Sizing gate at bottom
    sizing_gate_height: float = 0.05   # [m] Adjustable gap for bed depth
    sizing_gate_thickness: float = 0.005

    # Side rail height (vertical portion at bottom)
    side_rail_height: float = 0.10  # [m] Vertical side rails

    # Position (bottom front corner where material exits)
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_machine(cls, config, material=None) -> "InfeedHopperParams":
        """Create params from MachineConfig."""
        bed_depth = 0.05 if material is None else material.bed_depth_m
        return cls(
            width=min(config.belt_width_m * 0.5, 0.40),  # Max 40cm or half belt width
            sizing_gate_height=bed_depth + 0.01,  # Slightly larger than bed depth
        )

    @property
    def slant_angle_rad(self) -> float:
        """Slant angle in radians."""
        return math.radians(self.slant_angle_deg)

    @property
    def horizontal_extent(self) -> float:
        """Horizontal distance (X) from bottom to top of slant."""
        return self.slant_length * math.sin(self.slant_angle_rad)

    @property
    def vertical_extent(self) -> float:
        """Vertical distance (Y) from bottom to top of slant."""
        return self.slant_length * math.cos(self.slant_angle_rad)

    @property
    def total_height(self) -> float:
        """Total height from base to top."""
        return self.side_rail_height + self.vertical_extent


class InfeedHopperGeometry:
    """Realistic infeed hopper based on GP-15 machine images.

    The hopper has:
    - Slanted feed plate (back wall) at ~55° from vertical
    - Triangular side walls
    - Vertical side rails at the bottom
    - Sizing gate to control bed depth
    - Open bottom discharging onto conveyor belt

    Coordinate system (positioned at discharge point):
    - X: Conveyor direction (material flows in +X)
    - Y: Vertical (0 = belt surface)
    - Z: Across belt width
    """

    def __init__(self, params: Optional[InfeedHopperParams] = None):
        self.params = params or InfeedHopperParams()
        self._vertices: Optional[np.ndarray] = None
        self._triangles: Optional[np.ndarray] = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate realistic hopper mesh.

        Returns:
            (vertices, triangles, metadata)
        """
        if self._vertices is not None:
            return self._vertices, self._triangles, {"type": "infeed_hopper"}

        p = self.params
        cx, cy, cz = p.center
        t = p.plate_thickness

        # Key geometry points
        # Bottom of hopper (at belt level)
        rail_h = p.side_rail_height
        # Top of slant
        slant_dx = p.horizontal_extent  # How far back (-X) the top is
        slant_dy = p.vertical_extent    # How high the top is

        parts = []

        # === Slanted back plate ===
        # This is the main feed surface - a flat plate at an angle
        back_plate = self._create_slanted_plate(
            cx - slant_dx, cy + rail_h + slant_dy,  # Top back corner
            cx, cy + rail_h,                         # Bottom front corner
            cz, p.width, t
        )
        parts.append(back_plate)

        # === Side walls (triangular + rectangular) ===
        # Left side (-Z)
        left_side = self._create_side_wall(
            cx, cy, cz,
            slant_dx, slant_dy, rail_h, t
        )
        parts.append(left_side)

        # Right side (+Z)
        right_side = self._create_side_wall(
            cx, cy, cz + p.width - t,
            slant_dx, slant_dy, rail_h, t
        )
        parts.append(right_side)

        # === Bottom side rails (vertical portions) ===
        # These extend from belt level up to where the slant starts
        # Left rail
        left_rail = self._box_mesh(
            cx - t, cy, cz,
            t, rail_h, t
        )
        parts.append(left_rail)

        # Right rail
        right_rail = self._box_mesh(
            cx - t, cy, cz + p.width - t,
            t, rail_h, t
        )
        parts.append(right_rail)

        # === Sizing gate (adjustable plate at exit) ===
        # Positioned at +X side to control bed depth
        gate_y = cy + p.sizing_gate_height
        gate = self._box_mesh(
            cx, gate_y, cz,
            p.sizing_gate_thickness, p.opening_height - p.sizing_gate_height, p.width
        )
        parts.append(gate)

        # === Front lip (small plate at bottom front) ===
        front_lip = self._box_mesh(
            cx, cy, cz,
            t, 0.02, p.width  # Small 2cm lip
        )
        parts.append(front_lip)

        self._vertices, self._triangles = concat_meshes(parts)
        return self._vertices, self._triangles, {"type": "infeed_hopper"}

    def _create_slanted_plate(
        self,
        x_top: float, y_top: float,
        x_bot: float, y_bot: float,
        z_start: float, width: float, thickness: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a slanted rectangular plate.

        The plate goes from (x_top, y_top) to (x_bot, y_bot) at both
        z_start and z_start + width.
        """
        # 8 vertices for a slanted box
        verts = np.array([
            # Bottom edge (y_bot)
            [x_bot, y_bot, z_start],
            [x_bot, y_bot, z_start + width],
            # Top edge (y_top)
            [x_top, y_top, z_start],
            [x_top, y_top, z_start + width],
            # Same but offset by thickness in the normal direction
            # Normal points roughly in +X, +Y direction
            [x_bot + thickness, y_bot, z_start],
            [x_bot + thickness, y_bot, z_start + width],
            [x_top + thickness, y_top, z_start],
            [x_top + thickness, y_top, z_start + width],
        ], dtype=np.float32)

        # Triangles for front, back, and edges
        tris = np.array([
            # Front face (outer surface)
            [0, 1, 3], [0, 3, 2],
            # Back face (inner surface)
            [4, 6, 7], [4, 7, 5],
            # Bottom edge
            [0, 4, 5], [0, 5, 1],
            # Top edge
            [2, 3, 7], [2, 7, 6],
            # Left edge
            [0, 2, 6], [0, 6, 4],
            # Right edge
            [1, 5, 7], [1, 7, 3],
        ], dtype=np.int32)

        return verts, tris

    def _create_side_wall(
        self,
        cx: float, cy: float, cz: float,
        slant_dx: float, slant_dy: float,
        rail_height: float, thickness: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a side wall with triangular top portion.

        Shape: Rectangle at bottom (rail) + triangle above (slant area)
        """
        # Points defining the side profile (in X-Y plane at z=cz)
        # Starting from bottom-front, going clockwise
        p0 = (cx, cy)                                    # Bottom front
        p1 = (cx - thickness, cy)                        # Bottom back (small offset)
        p2 = (cx - thickness, cy + rail_height)          # Rail top back
        p3 = (cx - slant_dx, cy + rail_height + slant_dy)  # Slant top
        p4 = (cx, cy + rail_height)                      # Rail top front

        # Create vertices for both z=cz and z=cz+thickness faces
        verts = np.array([
            # Inner face (z = cz)
            [p0[0], p0[1], cz],  # 0
            [p1[0], p1[1], cz],  # 1
            [p2[0], p2[1], cz],  # 2
            [p3[0], p3[1], cz],  # 3
            [p4[0], p4[1], cz],  # 4
            # Outer face (z = cz + thickness)
            [p0[0], p0[1], cz + thickness],  # 5
            [p1[0], p1[1], cz + thickness],  # 6
            [p2[0], p2[1], cz + thickness],  # 7
            [p3[0], p3[1], cz + thickness],  # 8
            [p4[0], p4[1], cz + thickness],  # 9
        ], dtype=np.float32)

        # Triangles
        tris = np.array([
            # Inner face (pentagon split into triangles)
            [0, 1, 2], [0, 2, 4], [2, 3, 4],
            # Outer face
            [5, 7, 6], [5, 9, 7], [7, 9, 8],
            # Bottom edge (0-1 to 5-6)
            [0, 5, 6], [0, 6, 1],
            # Back edge (1-2-3 to 6-7-8)
            [1, 6, 7], [1, 7, 2],
            [2, 7, 8], [2, 8, 3],
            # Top edge (3-4 to 8-9) - diagonal
            [3, 8, 9], [3, 9, 4],
            # Front edge (4-0 to 9-5)
            [4, 9, 5], [4, 5, 0],
        ], dtype=np.int32)

        return verts, tris

    def _box_mesh(
        self,
        x0: float, y0: float, z0: float,
        lx: float, ly: float, lz: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple axis-aligned box mesh."""
        verts = np.array([
            [x0, y0, z0],
            [x0 + lx, y0, z0],
            [x0 + lx, y0 + ly, z0],
            [x0, y0 + ly, z0],
            [x0, y0, z0 + lz],
            [x0 + lx, y0, z0 + lz],
            [x0 + lx, y0 + ly, z0 + lz],
            [x0, y0 + ly, z0 + lz],
        ], dtype=np.float32)

        tris = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ], dtype=np.int32)

        return verts, tris

    @property
    def ports(self) -> Dict[str, "ConnectionPort"]:
        """Connection ports for material flow.

        - discharge: Where material exits onto belt (bottom, +X direction)
        - inlet: Open top for material loading
        """
        from airclassifier.geometry.connection_ports import ConnectionPort, PortType

        p = self.params
        cx, cy, cz = p.center

        return {
            'discharge': ConnectionPort(
                position=(cx, cy + p.sizing_gate_height / 2, cz + p.width / 2),
                direction=(1.0, 0.0, 0.0),  # Material flows in +X onto belt
                width=p.width,
                height=p.sizing_gate_height,
                port_type=PortType.GRAVITY,
                name="hopper_discharge",
            ),
            'inlet': ConnectionPort(
                position=(
                    cx - p.horizontal_extent / 2,
                    cy + p.total_height,
                    cz + p.width / 2
                ),
                direction=(0.0, 1.0, 0.0),  # Open top
                width=p.horizontal_extent,
                height=p.width,
                port_type=PortType.GRAVITY,
                name="hopper_inlet",
            ),
        }
