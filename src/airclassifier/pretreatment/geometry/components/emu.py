"""
EMU (Environment Management Unit) Geometry
===========================================

The EMU manages air circulation within the oven:
- Heater banks for warming inlet air
- Blower fans for air circulation
- Extraction duct for removing moist air
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import box_mesh, cylinder_mesh, concat_meshes

if TYPE_CHECKING:
    from airclassifier.geometry.connection_ports import ConnectionPort


@dataclass
class EMUParams:
    """Environment Management Unit parameters."""

    # Extraction duct
    duct_diameter: float = 0.25     # [m]
    duct_height: float = 0.4        # [m] Height above connection point
    duct_resolution: int = 16

    # Heater bank boxes (mounted on sides)
    heater_box_width: float = 0.4
    heater_box_height: float = 0.3
    heater_box_depth: float = 0.2
    num_heater_banks: int = 2
    heater_spacing: float = 0.6     # Spacing between heater banks

    # Position (duct base center)
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_machine(cls, config) -> "EMUParams":
        """Create params from MachineConfig."""
        return cls(
            duct_diameter=config.extraction_duct_diameter_m,
            num_heater_banks=config.heater_bank_count,
        )


class EMUGeometry:
    """EMU with extraction duct and heater banks.

    The extraction duct is mounted on top of the oven housing
    and removes moisture-laden air. Heater banks are positioned
    on the sides to warm incoming air.
    """

    def __init__(self, params: Optional[EMUParams] = None):
        self.params = params or EMUParams()
        self._vertices: Optional[np.ndarray] = None
        self._triangles: Optional[np.ndarray] = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate EMU geometry: duct cylinder + heater boxes.

        Returns:
            (vertices, triangles, metadata)
        """
        if self._vertices is not None:
            return self._vertices, self._triangles, {"type": "emu"}

        p = self.params
        cx, cy, cz = p.center

        parts = []

        # Extraction duct (vertical cylinder)
        duct = cylinder_mesh(
            center=(cx, cy, cz),
            radius=p.duct_diameter / 2,
            height=p.duct_height,
            resolution=p.duct_resolution,
            axis="y",
        )
        parts.append(duct)

        # Heater bank boxes (positioned symmetrically around duct)
        for i in range(p.num_heater_banks):
            offset = (i - (p.num_heater_banks - 1) / 2) * p.heater_spacing
            hx = cx + offset - p.heater_box_width / 2
            hy = cy  # At base level
            hz = cz + p.duct_diameter / 2 + 0.05  # Behind the duct

            heater = box_mesh(
                hx, hy, hz,
                p.heater_box_width, p.heater_box_height, p.heater_box_depth
            )
            parts.append(heater)

        self._vertices, self._triangles = concat_meshes(parts)
        return self._vertices, self._triangles, {"type": "emu"}

    @property
    def ports(self) -> Dict[str, "ConnectionPort"]:
        """Duct inlet connection to oven extraction port."""
        from airclassifier.geometry.connection_ports import ConnectionPort, PortType

        p = self.params
        cx, cy, cz = p.center

        return {
            'duct_inlet': ConnectionPort(
                position=(cx, cy, cz),
                direction=(0.0, -1.0, 0.0),  # Points down into oven
                diameter=p.duct_diameter,
                port_type=PortType.CIRCULAR,
                name="emu_duct_inlet",
            ),
            'exhaust': ConnectionPort(
                position=(cx, cy + p.duct_height, cz),
                direction=(0.0, 1.0, 0.0),  # Points up (exhaust)
                diameter=p.duct_diameter,
                port_type=PortType.CIRCULAR,
                name="emu_exhaust",
            ),
        }
