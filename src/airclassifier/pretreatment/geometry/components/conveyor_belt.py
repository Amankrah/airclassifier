"""
Conveyor Belt Geometry
======================

Belt geometry, material bed mesh, and advection domain.

This is a refactored version of the original conveyor.py with added
ConnectionPort support for assembly alignment.

The conveyor moves material continuously through the oven along the
positive X-axis. The belt rides on Teflon wear strips above the lower
electrode. Material is deposited at the infeed and exits at the outfeed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import box_mesh

if TYPE_CHECKING:
    from airclassifier.geometry.connection_ports import ConnectionPort
    from ...config import MachineConfig


@dataclass
class ConveyorBeltParams:
    """Conveyor belt geometry parameters."""

    belt_width_m: float = 0.8          # [m] Belt width (Z direction)
    belt_length_m: float = 1.5         # [m] Length within oven (X direction)
    belt_thickness_m: float = 0.002
    wear_strip_thickness_m: float = 0.001
    top_sheet_thickness_m: float = 0.0005

    # Position (belt start corner)
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_machine(cls, config: "MachineConfig") -> "ConveyorBeltParams":
        """Create params from MachineConfig."""
        return cls(
            belt_width_m=config.belt_width_m,
            belt_length_m=config.oven_length_m,
            belt_thickness_m=config.belt_thickness_m,
            wear_strip_thickness_m=config.wear_strip_thickness_m,
            top_sheet_thickness_m=config.top_sheet_thickness_m,
        )

    @property
    def belt_stack_thickness_m(self) -> float:
        """Total belt stack thickness."""
        return (self.belt_thickness_m +
                self.wear_strip_thickness_m +
                self.top_sheet_thickness_m)


class ConveyorBeltGeometry:
    """Generates the conveyor belt and material bed meshes.

    This class extends the original ConveyorGeometry with
    ConnectionPort support for hopper and material flow.
    """

    def __init__(self, params: Optional[ConveyorBeltParams] = None):
        self.params = params or ConveyorBeltParams()
        self._belt_vertices: Optional[np.ndarray] = None
        self._belt_triangles: Optional[np.ndarray] = None
        self._bed_vertices: Optional[np.ndarray] = None
        self._bed_triangles: Optional[np.ndarray] = None

    def generate_belt_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate belt surface mesh.

        Returns:
            (vertices, triangles, metadata)
        """
        if self._belt_vertices is not None:
            return self._belt_vertices, self._belt_triangles, {"type": "belt", "material": "PTFE"}

        p = self.params
        y = p.wear_strip_thickness_m + p.top_sheet_thickness_m

        self._belt_vertices, self._belt_triangles = box_mesh(
            0.0, y, 0.0,
            p.belt_length_m, p.belt_thickness_m, p.belt_width_m
        )

        return self._belt_vertices, self._belt_triangles, {"type": "belt", "material": "PTFE"}

    def generate_bed_mesh(
        self, bed_depth_m: float,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate material bed mesh (rectangular slab on belt).

        Args:
            bed_depth_m: Material bed depth [m].

        Returns:
            (vertices, triangles, metadata)
        """
        p = self.params
        y_base = p.belt_stack_thickness_m

        verts, tris = box_mesh(
            0.0, y_base, 0.0,
            p.belt_length_m, bed_depth_m, p.belt_width_m
        )

        return verts, tris, {"type": "material_bed"}

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate combined belt mesh (for compatibility).

        Returns:
            (vertices, triangles, metadata)
        """
        return self.generate_belt_mesh()

    @property
    def ports(self) -> Dict[str, "ConnectionPort"]:
        """Connection ports for material flow.

        Ports:
        - material_inlet: Where material enters at x=0 (from hopper)
        - material_outlet: Where material exits at x=length
        """
        from airclassifier.geometry.connection_ports import ConnectionPort, PortType

        p = self.params
        y_top = p.belt_stack_thickness_m + 0.05  # Above belt surface

        return {
            'material_inlet': ConnectionPort(
                position=(0.0, y_top, p.belt_width_m / 2),
                direction=(-1.0, 0.0, 0.0),  # Material enters from infeed
                width=p.belt_width_m,
                height=0.1,  # Approximate bed height
                port_type=PortType.GRAVITY,
                name="belt_material_inlet",
            ),
            'material_outlet': ConnectionPort(
                position=(p.belt_length_m, y_top, p.belt_width_m / 2),
                direction=(1.0, 0.0, 0.0),  # Material exits to outfeed
                width=p.belt_width_m,
                height=0.1,
                port_type=PortType.GRAVITY,
                name="belt_material_outlet",
            ),
        }


# Backward compatibility aliases
ConveyorParams = ConveyorBeltParams
ConveyorGeometry = ConveyorBeltGeometry
