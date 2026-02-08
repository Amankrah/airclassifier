"""
Oven Chamber Geometry
=====================

Generates the oven / applicator mesh: the rectangular chamber containing
the electrode system, conveyor belt, and material bed.

This is a refactored version of the original oven.py with added
ConnectionPort support for assembly alignment.

The oven is the primary simulation domain. Geometry includes:
- Upper electrode assembly (2 perforated plates + frame)
- Lower electrode trays (beneath belt)
- Side walls with viewing windows
- Infeed and outfeed openings

All dimensions sourced from MachineConfig.
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
class OvenChamberParams:
    """Parameters controlling oven chamber mesh generation."""

    length: float = 1.5            # [m] Oven length along conveyor (X)
    width: float = 0.8             # [m] Belt width (Z direction)
    height: float = 0.30           # [m] Max electrode gap (Y direction)
    wall_thickness: float = 0.005  # [m]
    resolution: int = 32           # Cells per longest dimension

    # Tunnel connection dimensions
    tunnel_opening_height: float = 0.35  # [m] Tunnel port height
    tunnel_opening_width: float = 0.9    # [m] Tunnel port width (> belt width)

    # EMU extraction port
    extraction_port_diameter: float = 0.25

    # Position (corner at origin in local coordinates)
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_machine(cls, config: "MachineConfig") -> "OvenChamberParams":
        """Create params from MachineConfig."""
        return cls(
            length=config.oven_length_m,
            width=config.belt_width_m,
            height=config.electrode_gap_max_m,
            extraction_port_diameter=config.extraction_duct_diameter_m,
        )


class OvenChamberGeometry:
    """Generates the oven chamber mesh and simulation grid.

    This class extends the original OvenGeometry with ConnectionPort
    support for tunnel and EMU connections.

    Usage::

        oven = OvenChamberGeometry(OvenChamberParams.from_machine(config))
        vertices, indices, meta = oven.generate_mesh()
        grid_shape = oven.get_grid_shape()
    """

    def __init__(self, params: Optional[OvenChamberParams] = None):
        self.params = params or OvenChamberParams()
        self._vertices: Optional[np.ndarray] = None
        self._triangles: Optional[np.ndarray] = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate oven wall mesh as a wireframe box.

        The oven is a rectangular chamber with infeed (x=0) and outfeed
        (x=L) openings. Walls are on the +/-Z sides and the +Y (top)
        face. The bottom (y=0) is the lower electrode tray.

        Returns:
            (vertices, triangles, metadata) triangle mesh.
        """
        if self._vertices is not None:
            return self._vertices, self._triangles, self._get_metadata()

        p = self.params
        L = p.length
        H = p.height
        W = p.width

        # Simple box mesh for the oven walls
        self._vertices, self._triangles = box_mesh(0, 0, 0, L, H, W)

        return self._vertices, self._triangles, self._get_metadata()

    def _get_metadata(self) -> dict:
        """Return mesh metadata."""
        p = self.params
        return {
            "type": "oven_chamber",
            "length_m": p.length,
            "width_m": p.width,
            "height_m": p.height,
        }

    def get_grid_shape(self) -> Tuple[int, int, int]:
        """Return (nx, ny, nz) for the simulation grid."""
        p = self.params
        r = p.resolution
        aspect_xz = p.length / p.width
        aspect_yz = p.height / p.width
        nz = r
        nx = max(4, int(r * aspect_xz))
        ny = max(4, int(r * aspect_yz))
        return (nx, ny, nz)

    def get_cell_sizes(self) -> Tuple[float, float, float]:
        """Return (dx, dy, dz) cell sizes in metres."""
        nx, ny, nz = self.get_grid_shape()
        return (
            self.params.length / nx,
            self.params.height / ny,
            self.params.width / nz,
        )

    def build_material_mask(
        self,
        electrode_gap_m: float,
        bed_depth_m: float,
        belt_stack_m: float = 0.0035,
    ) -> np.ndarray:
        """Build a 3D int array tagging each cell by zone.

        Zone IDs:
            0 - air gap (above bed)
            1 - material bed
            2 - belt / wear-strip / top-sheet stack

        The vertical layout within the electrode gap (Y-axis) is::

            y = gap      upper electrode
            y = d_bed    top of material bed
            y = d_belt   top of belt stack
            y = 0        lower electrode (ground)

        Args:
            electrode_gap_m: Current electrode gap [m].
            bed_depth_m: Material bed depth [m].
            belt_stack_m: Belt + wear strip + top sheet thickness [m].

        Returns:
            np.ndarray of shape (nx, ny, nz), dtype int32.
        """
        nx, ny, nz = self.get_grid_shape()
        dy = electrode_gap_m / ny

        mask = np.zeros((nx, ny, nz), dtype=np.int32)

        for j in range(ny):
            y_centre = (j + 0.5) * dy          # centre of cell j
            if y_centre < belt_stack_m:
                mask[:, j, :] = 2               # belt layer
            elif y_centre < belt_stack_m + bed_depth_m:
                mask[:, j, :] = 1               # material bed
            # else: 0 - air gap (default)

        return mask

    @property
    def ports(self) -> Dict[str, "ConnectionPort"]:
        """Connection ports for tunnels and EMU.

        Ports:
        - inlet: Infeed tunnel connection at x=0
        - outlet: Outfeed tunnel connection at x=length
        - extraction: EMU duct connection at top center
        """
        from airclassifier.geometry.connection_ports import ConnectionPort, PortType

        p = self.params

        return {
            'inlet': ConnectionPort(
                position=(0.0, p.tunnel_opening_height / 2, p.width / 2),
                direction=(-1.0, 0.0, 0.0),  # Points toward infeed tunnel
                width=p.tunnel_opening_width,
                height=p.tunnel_opening_height,
                port_type=PortType.RECTANGULAR,
                name="oven_inlet",
            ),
            'outlet': ConnectionPort(
                position=(p.length, p.tunnel_opening_height / 2, p.width / 2),
                direction=(1.0, 0.0, 0.0),  # Points toward outfeed tunnel
                width=p.tunnel_opening_width,
                height=p.tunnel_opening_height,
                port_type=PortType.RECTANGULAR,
                name="oven_outlet",
            ),
            'extraction': ConnectionPort(
                position=(p.length / 2, p.height, p.width / 2),
                direction=(0.0, 1.0, 0.0),  # Points up
                diameter=p.extraction_port_diameter,
                port_type=PortType.CIRCULAR,
                name="oven_extraction",
            ),
        }


# Backward compatibility aliases
OvenGeometryParams = OvenChamberParams
OvenGeometry = OvenChamberGeometry
