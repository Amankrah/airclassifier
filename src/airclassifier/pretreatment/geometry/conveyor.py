"""
Conveyor Belt Geometry
======================

Belt geometry, material bed mesh, and advection domain.

The conveyor moves material continuously through the oven along the
positive X-axis. The belt rides on Teflon wear strips above the lower
electrode. Material is deposited at the infeed and exits at the outfeed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from ..config import MachineConfig


@dataclass
class ConveyorParams:
    """Conveyor geometry parameters."""
    belt_width_m: float = 0.8
    belt_length_m: float = 1.5     # within oven
    belt_thickness_m: float = 0.002
    wear_strip_thickness_m: float = 0.001
    top_sheet_thickness_m: float = 0.0005

    @classmethod
    def from_machine(cls, config: MachineConfig) -> "ConveyorParams":
        return cls(
            belt_width_m=config.belt_width_m,
            belt_length_m=config.oven_length_m,
            belt_thickness_m=config.belt_thickness_m,
            wear_strip_thickness_m=config.wear_strip_thickness_m,
            top_sheet_thickness_m=config.top_sheet_thickness_m,
        )


class ConveyorGeometry:
    """Generates the conveyor belt and material bed meshes."""

    def __init__(self, params: Optional[ConveyorParams] = None):
        self.params = params or ConveyorParams()

    def generate_belt_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate belt surface mesh for visualization."""
        # TODO: Implement belt mesh generation
        raise NotImplementedError

    def generate_bed_mesh(self, bed_depth_m: float) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate material bed mesh (rectangular slab on belt)."""
        # TODO: Implement bed mesh generation
        raise NotImplementedError
