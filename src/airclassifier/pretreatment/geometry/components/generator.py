"""
RF Generator Geometry
=====================

RF oscillator cabinet with cooling vent grid pattern.
The generator houses the triode valve oscillator and power supply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import box_mesh, concat_meshes

if TYPE_CHECKING:
    from airclassifier.geometry.connection_ports import ConnectionPort


@dataclass
class GeneratorParams:
    """RF generator cabinet parameters."""

    # Cabinet dimensions (in machine coordinates: X=depth, Y=height, Z=width)
    width: float = 0.8          # [m] Cabinet width (Z direction)
    height: float = 1.8         # [m] Cabinet height (Y direction)
    depth: float = 0.6          # [m] Cabinet depth (X direction)
    wall_thickness: float = 0.02

    # Cooling vents (grid pattern on sides)
    vent_rows: int = 8
    vent_cols: int = 4
    vent_slot_width: float = 0.12
    vent_slot_height: float = 0.015
    vent_spacing_x: float = 0.03
    vent_spacing_y: float = 0.06

    # RF output connection point
    rf_output_diameter: float = 0.05  # Feed strip bundle
    rf_output_height: float = 1.3     # Height from base

    # Position in machine coordinates
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_machine(cls, config) -> "GeneratorParams":
        """Create params from MachineConfig."""
        return cls()  # Use defaults; position set by assembly


class GeneratorGeometry:
    """RF oscillator cabinet geometry.

    The generator is a cabinet with cooling vent slots on the sides.
    In the machine images, the generator shows a distinctive grid
    pattern of ventilation openings.
    """

    def __init__(self, params: Optional[GeneratorParams] = None):
        self.params = params or GeneratorParams()
        self._vertices: Optional[np.ndarray] = None
        self._triangles: Optional[np.ndarray] = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate cabinet mesh with cooling vents.

        Returns:
            (vertices, triangles, metadata)
        """
        if self._vertices is not None:
            return self._vertices, self._triangles, {"type": "generator"}

        p = self.params
        cx, cy, cz = p.center

        # Main cabinet body (simple box for now)
        parts = []

        # Cabinet is positioned with corner at (cx, cy, cz)
        # Full outer box
        main_box = box_mesh(cx, cy, cz, p.depth, p.height, p.width)
        parts.append(main_box)

        # Add vent detail strips (simplified representation)
        # Vents are horizontal slots on the +Z side (facing into machine)
        vent_start_y = cy + 0.3
        vent_start_x = cx + 0.1

        for row in range(min(p.vent_rows, 6)):
            for col in range(min(p.vent_cols, 3)):
                vx = vent_start_x + col * (p.vent_slot_width + p.vent_spacing_x)
                vy = vent_start_y + row * (p.vent_slot_height + p.vent_spacing_y)
                # Small raised strip to indicate vent
                vent = box_mesh(
                    vx, vy, cz + p.width - 0.005,
                    p.vent_slot_width * 0.8, p.vent_slot_height, 0.01
                )
                parts.append(vent)

        self._vertices, self._triangles = concat_meshes(parts)
        return self._vertices, self._triangles, {"type": "generator"}

    @property
    def ports(self) -> Dict[str, "ConnectionPort"]:
        """Connection ports for RF output.

        The RF output connects to the electrode assembly via
        copper feed strips.
        """
        from airclassifier.geometry.connection_ports import ConnectionPort, PortType

        p = self.params
        cx, cy, cz = p.center

        return {
            'rf_output': ConnectionPort(
                position=(cx + p.depth, cy + p.rf_output_height, cz + p.width / 2),
                direction=(1.0, 0.0, 0.0),  # Points into machine (+X)
                diameter=p.rf_output_diameter,
                port_type=PortType.SLIP,
                name="rf_output",
            ),
        }
