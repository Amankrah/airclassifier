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

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING, List

import numpy as np

from ..mesh_utils import box_mesh, cylinder_mesh, concat_meshes, arc_surface_mesh, disc_mesh

if TYPE_CHECKING:
    from ...config import MillConfig
    from .housing import HousingParams


@dataclass
class DriveParams:
    """Drive system geometry parameters.

    Proportional rules (when built from config/housing):
    - Motor size scales with motor_power_kw.
    - Pulley radii: mill pulley ~0.5× rotor radius, motor pulley from speed ratio.
    - Pulley width ~0.4× pulley radius (belt contact).
    - Base plate extends beyond motor by ~5% each side; thickness ~1.5% of motor length.
    - Belt guard clearance: inner_r = max(pulley radii) + 2% of housing radius.
    - Motor Y offset: below rotor centerline by ~40% of housing outer radius.
    """

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
        """Create drive params with dimensions proportional to mill configuration."""
        # Motor size scales with power (reference 22 kW)
        power_factor = config.motor_power_kw / 22.0
        motor_length = 0.35 * (0.8 + 0.2 * power_factor)
        motor_width = 0.20 * (0.8 + 0.2 * power_factor)
        motor_height = 0.22 * (0.8 + 0.2 * power_factor)

        # Pulleys proportional to rotor: mill pulley ~0.5× rotor radius
        rotor_radius = config.rotor_diameter_m / 2.0
        mill_pulley_r = rotor_radius * 0.5
        # Motor pulley from approximate speed ratio (mill slower or same; keep ratio ~1.5–2)
        motor_pulley_r = mill_pulley_r * 0.6
        # Pulley width proportional to radius (belt contact)
        pulley_width = max(0.03, mill_pulley_r * 0.4)

        # Base plate: extends beyond motor; thickness proportional to motor size
        base_margin = 0.02
        base_width = motor_length + 2 * base_margin
        base_depth = motor_width + 0.15
        base_thickness = max(0.012, motor_length * 0.04)

        # Belt guard: span and clearance from pulley sizes and housing scale
        ref_scale = config.housing_inner_radius_m
        guard_length = abs(mill_pulley_r * 2) + ref_scale * 0.2
        guard_width = pulley_width + ref_scale * 0.15
        guard_height = (mill_pulley_r + motor_pulley_r) + ref_scale * 0.1

        return cls(
            motor_length_m=motor_length,
            motor_width_m=motor_width,
            motor_height_m=motor_height,
            motor_z_offset_m=config.housing_inner_radius_m + config.housing_wall_thickness_m + motor_width / 2.0 + 0.02,
            base_plate_width_m=base_width,
            base_plate_depth_m=base_depth,
            base_plate_thickness_m=base_thickness,
            belt_guard_length_m=guard_length,
            belt_guard_width_m=guard_width,
            belt_guard_height_m=guard_height,
            pulley_radius_m=motor_pulley_r,
            mill_pulley_radius_m=mill_pulley_r,
            pulley_width_m=pulley_width,
            mill_pulley_width_m=pulley_width,
        )

    @classmethod
    def from_housing(cls, housing_params: "HousingParams", config: "MillConfig") -> "DriveParams":
        """Create drive params positioned and scaled relative to housing.

        - Motor flush against housing drive end (X=0).
        - Motor beside housing in +Z with minimal clearance.
        - Motor Y offset proportional to housing (below rotor centerline).
        """
        params = cls.from_mill_config(config)
        # Motor flush against drive end: right face of motor at housing start (X=0)
        params.motor_x_offset_m = housing_params.center_x_m - params.motor_length_m
        # Motor beside housing
        params.motor_z_offset_m = (
            housing_params.outer_radius_m + params.motor_width_m / 2.0 + 0.02
        )
        # Motor below centerline by proportion of housing size (belt line)
        params.motor_y_offset_m = -housing_params.outer_radius_m * 0.65
        return params

    def validate_proportions(self, tolerance: float = 0.15) -> list:
        """Check that key proportions are within expected ranges. Returns list of (check_name, ok, message)."""
        checks = []
        # Pulley width vs radius
        r = max(self.pulley_radius_m, self.mill_pulley_radius_m)
        w = self.pulley_width_m
        ratio = w / r if r > 0 else 0
        checks.append(("pulley_width_ratio", 0.2 <= ratio <= 0.8, f"pulley width/radius={ratio:.2f}"))
        # Base plate contains motor
        ok_base = (self.base_plate_width_m >= self.motor_length_m * 0.9 and
                   self.base_plate_depth_m >= self.motor_width_m * 1.2)
        checks.append(("base_contains_motor", ok_base, "base plate vs motor footprint"))
        # Belt guard clears pulleys (inner_r > max pulley in geometry; checked in mesh)
        checks.append(("guard_clearance", self.belt_guard_width_m >= self.pulley_width_m * 1.1,
                       "guard width vs pulley width"))
        return checks


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
        resolution: int = 24,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the drive system mesh.

        Motor is a cylindrical body (realistic NEMA-style) with base plate and feet,
        shaft pulley, and a curved belt guard. Mill pulley sits at housing drive end.

        Args:
            resolution: Number of segments for cylindrical parts.

        Returns:
            (vertices, triangles, metadata)
        """
        p = self.params
        parts = []

        # Motor frame: left (shaft) end at motor_x, body extends in +X
        motor_x = p.motor_x_offset_m
        motor_y = p.motor_y_offset_m
        motor_z = p.motor_z_offset_m

        # Cylindrical body: radius from motor cross-section (Y-Z), length along X
        body_radius = min(p.motor_height_m, p.motor_width_m) * 0.48  # slight margin
        body_length = p.motor_length_m

        # --- Motor body (cylinder along X, shaft end at motor_x + body_length) ---
        motor_body = cylinder_mesh(
            center=(motor_x, motor_y, motor_z),
            radius=body_radius,
            height=body_length,
            resolution=resolution,
            axis="x",
        )
        parts.append(motor_body)

        # --- End bells (slightly larger radius, short) for a more realistic look ---
        bell_len = 0.015
        bell_radius = body_radius * 1.08
        end_bell_drive = cylinder_mesh(
            center=(motor_x, motor_y, motor_z),
            radius=bell_radius,
            height=bell_len,
            resolution=resolution,
            axis="x",
        )
        parts.append(end_bell_drive)
        end_bell_shaft = cylinder_mesh(
            center=(motor_x + body_length - bell_len, motor_y, motor_z),
            radius=bell_radius,
            height=bell_len,
            resolution=resolution,
            axis="x",
        )
        parts.append(end_bell_shaft)

        # --- Base plate (rectangular bed under motor; size from params) ---
        base_x = motor_x - (p.base_plate_width_m - body_length) / 2
        base_y = motor_y - body_radius - p.base_plate_thickness_m
        base_z = motor_z - p.base_plate_depth_m / 2
        parts.append(box_mesh(
            base_x, base_y, base_z,
            p.base_plate_width_m, p.base_plate_thickness_m, p.base_plate_depth_m,
        ))

        # --- Motor feet (two pads under the base plate, NEMA style) ---
        foot_len = body_length * 0.35
        foot_w = 0.04
        foot_h = 0.02
        for x_off in (0.15 * body_length, 0.65 * body_length):
            fx = motor_x + x_off - foot_len / 2
            fz = motor_z - p.base_plate_depth_m / 2
            parts.append(box_mesh(
                fx, base_y - foot_h, fz,
                foot_len, foot_h, foot_w,
            ))
            parts.append(box_mesh(
                fx, base_y - foot_h, fz + p.base_plate_depth_m - foot_w,
                foot_len, foot_h, foot_w,
            ))

        # --- Motor shaft pulley (at drive end of motor, +X) ---
        pulley_x = motor_x + body_length
        pulley_y = motor_y
        pulley_z = motor_z

        motor_pulley = cylinder_mesh(
            center=(pulley_x, pulley_y, pulley_z),
            radius=p.pulley_radius_m,
            height=p.pulley_width_m,
            resolution=resolution,
            axis="z",
        )
        parts.append(motor_pulley)

        # --- Mill pulley (on rotor shaft at housing drive end, same Z as motor pulley for belt) ---
        mill_pulley_x = 0.0
        mill_pulley = cylinder_mesh(
            center=(mill_pulley_x, 0.0, pulley_z),
            radius=p.mill_pulley_radius_m,
            height=p.mill_pulley_width_m,
            resolution=resolution,
            axis="z",
        )
        parts.append(mill_pulley)

        # --- Belt guard (curved half-cylinder over belt path between pulleys) ---
        parts.append(self._belt_guard_mesh(p, pulley_x, motor_y, motor_z, resolution))

        self._cached_verts, self._cached_tris = concat_meshes(parts)

        metadata = {
            "type": "drive",
            "animation_type": None,  # Static (or "rotate" for pulley)
            "motor_position": (motor_x, motor_y, motor_z),
            "motor_dimensions": (body_length, body_radius * 2, body_radius * 2),
            "pulley_ratio": p.mill_pulley_radius_m / p.pulley_radius_m,
        }

        return self._cached_verts, self._cached_tris, metadata

    def generate_mesh_parts(
        self,
        resolution: int = 24,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Generate drive meshes as separate parts for color-coded visualization.

        Returns:
            Dict mapping part name to (vertices, triangles):
            - drive_motor: cylindrical body + end bells
            - drive_base: base plate
            - drive_feet: motor feet (all four)
            - drive_pulley_motor: motor shaft pulley
            - drive_pulley_mill: mill (rotor) pulley
            - drive_guard: belt guard
        """
        p = self.params
        motor_x = p.motor_x_offset_m
        motor_y = p.motor_y_offset_m
        motor_z = p.motor_z_offset_m
        body_radius = min(p.motor_height_m, p.motor_width_m) * 0.48
        body_length = p.motor_length_m
        bell_len = 0.015
        bell_radius = body_radius * 1.08
        base_x = motor_x - 0.02
        base_y = motor_y - body_radius - p.base_plate_thickness_m
        foot_len = body_length * 0.35
        foot_w = 0.04
        foot_h = 0.02
        pulley_x = motor_x + body_length

        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Motor (body + end bells)
        motor_body = cylinder_mesh(
            center=(motor_x, motor_y, motor_z),
            radius=body_radius,
            height=body_length,
            resolution=resolution,
            axis="x",
        )
        end_bell_drive = cylinder_mesh(
            center=(motor_x, motor_y, motor_z),
            radius=bell_radius,
            height=bell_len,
            resolution=resolution,
            axis="x",
        )
        end_bell_shaft = cylinder_mesh(
            center=(motor_x + body_length - bell_len, motor_y, motor_z),
            radius=bell_radius,
            height=bell_len,
            resolution=resolution,
            axis="x",
        )
        # Cooling fins (thin rings along motor body)
        fin_inner = body_radius - 0.003
        fin_outer = body_radius + 0.012
        fin_thick = 0.004
        n_fins = 8
        fin_spacing = (body_length - bell_len * 2) / (n_fins + 1)
        fins = []
        for i in range(n_fins):
            fx = motor_x + bell_len + fin_spacing * (i + 1)
            d = disc_mesh(
                center=(fx, motor_y, motor_z),
                inner_radius=fin_inner,
                outer_radius=fin_outer,
                thickness=fin_thick,
                resolution=resolution,
                axis="x",
            )
            fins.append(d)
        # Fan cover (back of motor, drive end)
        fan = disc_mesh(
            center=(motor_x, motor_y, motor_z),
            inner_radius=0.025,
            outer_radius=bell_radius,
            thickness=0.008,
            resolution=resolution,
            axis="x",
        )
        out["drive_motor"] = concat_meshes(
            [motor_body, end_bell_drive, end_bell_shaft, *fins, fan]
        )

        # Base plate (dimensions from params for proportional consistency)
        out["drive_base"] = box_mesh(
            motor_x - (p.base_plate_width_m - body_length) / 2,
            base_y,
            motor_z - p.base_plate_depth_m / 2,
            p.base_plate_width_m,
            p.base_plate_thickness_m,
            p.base_plate_depth_m,
        )

        # Feet (all four as one mesh for one color)
        feet_parts: List[Tuple[np.ndarray, np.ndarray]] = []
        for x_off in (0.15 * body_length, 0.65 * body_length):
            fx = motor_x + x_off - foot_len / 2
            fz = motor_z - p.base_plate_depth_m / 2
            feet_parts.append(box_mesh(
                fx, base_y - foot_h, fz,
                foot_len, foot_h, foot_w,
            ))
            feet_parts.append(box_mesh(
                fx, base_y - foot_h, fz + p.base_plate_depth_m - foot_w,
                foot_len, foot_h, foot_w,
            ))
        out["drive_feet"] = concat_meshes(feet_parts)

        # Motor pulley
        out["drive_pulley_motor"] = cylinder_mesh(
            center=(pulley_x, motor_y, motor_z),
            radius=p.pulley_radius_m,
            height=p.pulley_width_m,
            resolution=resolution,
            axis="z",
        )

        # Mill pulley
        out["drive_pulley_mill"] = cylinder_mesh(
            center=(0.0, 0.0, motor_z),
            radius=p.mill_pulley_radius_m,
            height=p.mill_pulley_width_m,
            resolution=resolution,
            axis="z",
        )

        # Belt (visible strip connecting mill pulley to motor pulley — shows drive connection)
        out["drive_belt"] = self._belt_mesh(p, motor_y, motor_z)

        # Belt guard (curved half-cylinder)
        out["drive_guard"] = self._belt_guard_mesh(p, pulley_x, motor_y, motor_z, resolution)

        return out

    def _belt_mesh(
        self,
        p: "DriveParams",
        motor_y: float,
        motor_z: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Visible belt run between mill pulley (Y=0) and motor pulley (Y=motor_y) at Z=motor_z."""
        belt_thick_x = max(0.008, p.pulley_width_m * 0.35)
        belt_len_y = abs(motor_y) + 0.03
        belt_depth_z = p.pulley_width_m + 0.002
        cx = 0.0
        cy = motor_y / 2.0
        cz = motor_z
        box = box_mesh(
            cx - belt_thick_x / 2,
            cy - belt_len_y / 2,
            cz - belt_depth_z / 2,
            belt_thick_x,
            belt_len_y,
            belt_depth_z,
        )
        return box

    def _belt_guard_mesh(
        self,
        p: "DriveParams",
        pulley_x: float,
        motor_y: float,
        motor_z: float,
        resolution: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Curved belt guard: half-cylinder tunnel along the belt path (Y) over the pulleys."""
        inner_r = max(p.mill_pulley_radius_m, p.pulley_radius_m) + 0.02
        outer_r = inner_r + 0.03
        # Pulleys are at X=0; belt runs in Y from (0, 0, z) to (0, motor_y, z)
        center_y = motor_y / 2.0
        guard_length_y = abs(motor_y) + p.pulley_width_m + 0.04
        arc = arc_surface_mesh(
            center=(0.0, center_y, motor_z),
            inner_radius=inner_r,
            outer_radius=outer_r,
            start_angle_rad=-math.pi / 2,
            end_angle_rad=math.pi / 2,
            length=guard_length_y,
            radial_resolution=max(8, resolution // 2),
            axial_resolution=8,
            axis="y",
        )
        return arc

    @property
    def ports(self) -> Dict[str, any]:
        """Drive system has no connection ports."""
        return {}
