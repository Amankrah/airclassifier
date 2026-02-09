"""
Support Legs Geometry
=====================

Machine support legs / stands.
Four legs positioned at corners to support the machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import box_mesh, concat_meshes

if TYPE_CHECKING:
    from airclassifier.geometry.connection_ports import ConnectionPort


@dataclass
class SupportLegsParams:
    """Support legs parameters."""

    # Leg dimensions
    leg_width: float = 0.10     # [m] Leg cross-section (square)
    leg_height: float = 0.80    # [m] Leg height (to bottom of housing)

    # Machine footprint for leg positioning
    machine_length: float = 5.5
    machine_width: float = 2.9

    # Leg inset from machine edges
    inset_x: float = 0.3
    inset_z: float = 0.3

    # Foot pad
    foot_width: float = 0.15
    foot_height: float = 0.02

    # Position offset
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_machine(cls, config) -> "SupportLegsParams":
        """Create params from MachineConfig."""
        return cls(
            machine_length=config.machine_length_m,
            machine_width=config.machine_width_m,
            leg_height=0.80,  # Standard industrial conveyor height
        )

    @property
    def leg_positions(self) -> List[Tuple[float, float]]:
        """Return (x, z) positions for the four legs."""
        return [
            (self.inset_x, self.inset_z),                                    # Front-left
            (self.inset_x, self.machine_width - self.inset_z),               # Front-right
            (self.machine_length - self.inset_x, self.inset_z),              # Back-left
            (self.machine_length - self.inset_x, self.machine_width - self.inset_z),  # Back-right
        ]


class SupportLegsGeometry:
    """Four support legs at machine corners.

    Each leg is a vertical column with a foot pad at the bottom.
    """

    def __init__(self, params: Optional[SupportLegsParams] = None):
        self.params = params or SupportLegsParams()
        self._vertices: Optional[np.ndarray] = None
        self._triangles: Optional[np.ndarray] = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate support legs mesh.

        Returns:
            (vertices, triangles, metadata)
        """
        if self._vertices is not None:
            return self._vertices, self._triangles, {"type": "support_legs"}

        p = self.params
        cx, cy, cz = p.center
        w = p.leg_width
        h = p.leg_height

        parts = []

        for lx, lz in p.leg_positions:
            # Leg column
            leg = box_mesh(
                cx + lx - w/2, cy, cz + lz - w/2,
                w, h, w
            )
            parts.append(leg)

            # Foot pad
            fw = p.foot_width
            fh = p.foot_height
            foot = box_mesh(
                cx + lx - fw/2, cy, cz + lz - fw/2,
                fw, fh, fw
            )
            parts.append(foot)

        self._vertices, self._triangles = concat_meshes(parts)
        return self._vertices, self._triangles, {"type": "support_legs"}

    @property
    def ports(self) -> Dict[str, "ConnectionPort"]:
        """Support legs have no connection ports."""
        return {}
