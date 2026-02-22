"""
Drive Geometry
==============

Hammer mill drive system: motor, belt/coupling, and belt guard.
This is primarily for visualization; the physics uses rpm directly.

Physical components:
    - Electric motor (simplified box)
    - Motor mount/base plate
    - Belt guard / coupling cover
    - Pulley (optional detail)

Coordinate system:
    X = along rotor axis (motor is offset in Z)
    Y = vertical
    Z = lateral (motor alongside housing)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import box_mesh, cylinder_mesh, concat_meshes

if TYPE_CHECKING:
    from ...config import MillConfig
    from .housing import HousingParams


@dataclass
class DriveParams:
    """Drive system geometry parameters."""

    # --- Motor dimensions ---
    motor_length_m: float = 0.35                  # Motor body length (X)
    motor_width_m: float = 0.20                   # Motor width (Z)
    motor_height_m: float = 0.22                  # Motor height (Y)

    # --- Motor position (relative to housing) ---
    motor_x_offset_m: float = -0.10               # X offset from housing start
    motor_z_offset_m: float = 0.35                # Z offset (beside housing)
    motor_y_offset_m: float = -0.15               # Y offset (below rotor centerline)

    # --- Base plate ---
    base_plate_width_m: float = 0.30
    base_plate_depth_m: float = 0.45
    base_plate_thickness_m: float = 0.015

    # --- Belt guard ---
    belt_guard_length_m: float = 0.15
    belt_guard_width_m: float = 0.25
    belt_guard_height_m: float = 0.20

    # --- Pulley (on motor shaft) ---
    pulley_radius_m: float = 0.06
    pulley_width_m: float = 0.04

    # --- Mill pulley (on rotor shaft) ---
    mill_pulley_radius_m: float = 0.10
    mill_pulley_width_m: float = 0.04

    @classmethod
    def from_mill_config(cls, config: "MillConfig") -> "DriveParams":
        """Create drive params from mill configuration."""
        # Scale motor size based on power
        power_factor = config.motor_power_kw / 22.0
        return cls(
            motor_length_m=0.35 * (0.8 + 0.2 * power_factor),
            motor_width_m=0.20 * (0.8 + 0.2 * power_factor),
            motor_height_m=0.22 * (0.8 + 0.2 * power_factor),
            motor_z_offset_m=config.housing_inner_radius_m + 0.15,
        )

    @classmethod
    def from_housing(cls, housing_params: "HousingParams", config: "MillConfig") -> "DriveParams":
        """Create drive params positioned relative to housing."""
        params = cls.from_mill_config(config)
        params.motor_z_offset_m = housing_params.outer_radius_m + 0.10
        params.motor_x_offset_m = housing_params.center_x_m - 0.05
        return params


class DriveGeometry:
    """Generates drive system meshes.

    Creates motor, base plate, belt guard, and pulleys for
    visualization. The motor is positioned beside the housing
    at the drive end of the rotor.

    Animation:
        The motor pulley can optionally rotate (scaled from rotor rpm
        by the pulley ratio). For simplicity, this is typically not
        animated.
    """

    def __init__(self, params: Optional[DriveParams] = None):
        self.params = params or DriveParams()
        self._cached_verts: Optional[np.ndarray] = None
        self._cached_tris: Optional[np.ndarray] = None

    def generate_mesh(
        self,
        resolution: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the drive system mesh.

        Args:
            resolution: Number of segments for cylindrical parts.

        Returns:
            (vertices, triangles, metadata)
        """
        p = self.params
        parts = []

        # Motor center position
        motor_x = p.motor_x_offset_m
        motor_y = p.motor_y_offset_m
        motor_z = p.motor_z_offset_m

        # --- Base plate ---
        base_x = motor_x - (p.base_plate_width_m - p.motor_length_m) / 2
        base_y = motor_y - p.motor_height_m / 2 - p.base_plate_thickness_m
        base_z = motor_z - p.base_plate_depth_m / 2 + p.motor_width_m / 2

        parts.append(box_mesh(
            base_x, base_y, base_z,
            p.base_plate_width_m, p.base_plate_thickness_m, p.base_plate_depth_m,
        ))

        # --- Motor body ---
        parts.append(box_mesh(
            motor_x,
            motor_y - p.motor_height_m / 2,
            motor_z,
            p.motor_length_m,
            p.motor_height_m,
            p.motor_width_m,
        ))

        # --- Motor pulley ---
        pulley_x = motor_x + p.motor_length_m
        pulley_y = motor_y
        pulley_z = motor_z + p.motor_width_m / 2 - p.pulley_width_m / 2

        motor_pulley = cylinder_mesh(
            center=(pulley_x, pulley_y, pulley_z),
            radius=p.pulley_radius_m,
            height=p.pulley_width_m,
            resolution=resolution,
            axis="z",
        )
        parts.append(motor_pulley)

        # --- Mill pulley (on rotor shaft, near housing) ---
        mill_pulley_x = -0.02  # Just outside housing drive end
        mill_pulley = cylinder_mesh(
            center=(mill_pulley_x, 0.0, -p.mill_pulley_width_m / 2),
            radius=p.mill_pulley_radius_m,
            height=p.mill_pulley_width_m,
            resolution=resolution,
            axis="z",
        )
        parts.append(mill_pulley)

        # --- Belt guard ---
        guard_x = motor_x + p.motor_length_m - 0.02
        guard_y = motor_y - p.belt_guard_height_m / 2
        guard_z = motor_z + p.motor_width_m / 2

        # Guard is a box covering the belt area
        parts.append(box_mesh(
            guard_x,
            guard_y,
            guard_z - p.belt_guard_width_m / 2,
            p.belt_guard_length_m,
            p.belt_guard_height_m,
            p.belt_guard_width_m,
        ))

        self._cached_verts, self._cached_tris = concat_meshes(parts)

        metadata = {
            "type": "drive",
            "animation_type": None,  # Static (or "rotate" for pulley)
            "motor_position": (motor_x, motor_y, motor_z),
            "motor_dimensions": (p.motor_length_m, p.motor_height_m, p.motor_width_m),
            "pulley_ratio": p.mill_pulley_radius_m / p.pulley_radius_m,
        }

        return self._cached_verts, self._cached_tris, metadata

    @property
    def ports(self) -> Dict[str, any]:
        """Drive system has no connection ports."""
        return {}
