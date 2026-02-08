"""
Oven Chamber Geometry
=====================

Generates the oven / applicator mesh: the rectangular chamber containing
the electrode system, conveyor belt, and material bed.

The oven is the primary simulation domain. Geometry includes:
- Upper electrode assembly (2 perforated plates + frame)
- Lower electrode trays (beneath belt)
- Side walls with viewing windows
- Infeed and outfeed openings

All dimensions sourced from MachineConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from ..config import MachineConfig


@dataclass
class OvenGeometryParams:
    """Parameters controlling oven mesh generation."""
    length: float = 1.5            # m  (oven length along conveyor X)
    width: float = 0.8             # m  (belt width, Z direction)
    height: float = 0.30           # m  (max electrode gap, Y direction)
    wall_thickness: float = 0.005  # m
    resolution: int = 32           # cells per longest dimension

    @classmethod
    def from_machine(cls, config: MachineConfig) -> "OvenGeometryParams":
        return cls(
            length=config.oven_length_m,
            width=config.belt_width_m,
            height=config.electrode_gap_max_m,
        )


class OvenGeometry:
    """Generates the oven chamber mesh and simulation grid.

    Usage::

        oven = OvenGeometry(OvenGeometryParams.from_machine(config))
        vertices, indices = oven.generate_mesh()
        grid_shape = oven.get_grid_shape()
    """

    def __init__(self, params: Optional[OvenGeometryParams] = None):
        self.params = params or OvenGeometryParams()

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate oven wall mesh.

        Returns:
            (vertices, indices, metadata) triangle mesh.
        """
        # TODO: Implement oven wall geometry
        raise NotImplementedError

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
            0 — air gap (above bed)
            1 — material bed
            2 — belt / wear-strip / top-sheet stack

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
            # else: 0 — air gap (default)

        return mask
