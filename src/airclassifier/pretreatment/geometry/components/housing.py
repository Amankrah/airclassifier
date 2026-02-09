"""
Machine Housing Geometry
========================

Outer cabinet walls enclosing the machine internals.
The housing provides structural support and safety enclosure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import box_mesh, concat_meshes

if TYPE_CHECKING:
    from airclassifier.geometry.connection_ports import ConnectionPort


@dataclass
class HousingParams:
    """Outer cabinet housing parameters."""

    # Machine envelope dimensions
    length: float = 5.5         # [m] Machine length (X direction)
    width: float = 2.9          # [m] Machine width (Z direction)
    height: float = 2.2         # [m] Machine height (Y direction)

    # Wall and floor thickness
    wall_thickness: float = 0.04
    floor_thickness: float = 0.05

    # Base height (from floor to bottom of main housing)
    base_height: float = 0.85   # Conveyor height

    # Position (corner at origin)
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_machine(cls, config) -> "HousingParams":
        """Create params from MachineConfig."""
        return cls(
            length=config.machine_length_m,
            width=config.machine_width_m,
            height=config.machine_height_m,
        )


class HousingGeometry:
    """Outer cabinet walls and floor.

    The housing is a semi-transparent enclosure showing:
    - Bottom platform at conveyor height
    - Top ceiling panel
    - Side walls (left and right)
    - Open at front and back for tunnel access
    """

    def __init__(self, params: Optional[HousingParams] = None):
        self.params = params or HousingParams()
        self._vertices: Optional[np.ndarray] = None
        self._triangles: Optional[np.ndarray] = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate housing mesh (floor, ceiling, side walls).

        Returns:
            (vertices, triangles, metadata)
        """
        if self._vertices is not None:
            return self._vertices, self._triangles, {"type": "housing"}

        p = self.params
        cx, cy, cz = p.center
        L = p.length
        W = p.width
        H = p.height
        t = p.wall_thickness
        y_base = p.base_height

        parts = []

        # Floor platform (at base height)
        floor = box_mesh(cx, cy + y_base - p.floor_thickness, cz,
                        L, p.floor_thickness, W)
        parts.append(floor)

        # Ceiling
        ceiling = box_mesh(cx, cy + H - p.floor_thickness, cz,
                          L, p.floor_thickness, W)
        parts.append(ceiling)

        # Left wall (-Z side)
        left_wall = box_mesh(cx, cy + y_base, cz,
                            L, H - y_base - p.floor_thickness, t)
        parts.append(left_wall)

        # Right wall (+Z side)
        right_wall = box_mesh(cx, cy + y_base, cz + W - t,
                             L, H - y_base - p.floor_thickness, t)
        parts.append(right_wall)

        self._vertices, self._triangles = concat_meshes(parts)
        return self._vertices, self._triangles, {"type": "housing"}

    @property
    def ports(self) -> Dict[str, "ConnectionPort"]:
        """Housing provides extraction port on top for EMU."""
        from airclassifier.geometry.connection_ports import ConnectionPort, PortType

        p = self.params
        cx, cy, cz = p.center

        return {
            'top_extraction': ConnectionPort(
                position=(cx + p.length / 2, cy + p.height, cz + p.width / 2),
                direction=(0.0, 1.0, 0.0),
                diameter=0.25,  # Standard extraction duct
                port_type=PortType.CIRCULAR,
                name="housing_extraction_port",
            ),
        }
