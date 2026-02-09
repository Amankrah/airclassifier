"""
HMI Panel Geometry
==================

Control console / Human-Machine Interface panel.
Side-mounted touchscreen panel for operator control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import box_mesh, concat_meshes

if TYPE_CHECKING:
    from airclassifier.geometry.connection_ports import ConnectionPort


@dataclass
class HMIPanelParams:
    """HMI control panel parameters."""

    # Panel dimensions
    width: float = 0.6          # [m] Panel width (X direction)
    height: float = 0.8         # [m] Panel height (Y direction)
    depth: float = 0.15         # [m] Panel depth (Z direction, into machine)

    # Screen area
    screen_width: float = 0.4
    screen_height: float = 0.3
    screen_offset_y: float = 0.3  # From bottom of panel

    # Position
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_machine(cls, config) -> "HMIPanelParams":
        """Create params from MachineConfig."""
        return cls()


class HMIPanelGeometry:
    """Control console panel for operator interface.

    A side-mounted panel with a touchscreen display area
    for controlling the GP-15 machine.
    """

    def __init__(self, params: Optional[HMIPanelParams] = None):
        self.params = params or HMIPanelParams()
        self._vertices: Optional[np.ndarray] = None
        self._triangles: Optional[np.ndarray] = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate HMI panel mesh.

        Returns:
            (vertices, triangles, metadata)
        """
        if self._vertices is not None:
            return self._vertices, self._triangles, {"type": "hmi_panel"}

        p = self.params
        cx, cy, cz = p.center

        parts = []

        # Main panel body
        main = box_mesh(cx, cy, cz, p.width, p.height, p.depth)
        parts.append(main)

        # Screen bezel (slightly raised)
        screen_x = cx + (p.width - p.screen_width) / 2
        screen_y = cy + p.screen_offset_y
        screen_z = cz - 0.005  # Slightly in front

        bezel = box_mesh(
            screen_x, screen_y, screen_z,
            p.screen_width, p.screen_height, 0.01
        )
        parts.append(bezel)

        self._vertices, self._triangles = concat_meshes(parts)
        return self._vertices, self._triangles, {"type": "hmi_panel"}

    @property
    def ports(self) -> Dict[str, "ConnectionPort"]:
        """HMI panel has no physical connection ports."""
        return {}
