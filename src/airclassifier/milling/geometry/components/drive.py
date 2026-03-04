"""
Drive Geometry
==============

Hammer mill drive system: motor and pulleys.
This is primarily for visualization; the physics uses rpm directly.

Physical components:
    - Electric motor (simplified cylinder)
    - Motor mount/base plate
    - Pulleys (mill pulley on rotor, motor pulley on motor shaft)

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

from ..mesh_utils import (
    arc_surface_mesh,
    box_mesh,
    cylinder_mesh,
    frustum_mesh,
    concat_meshes,
    disc_mesh,
)

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
    motor_pulley_x_m: float = 0.04   # X position: positive = allowance between motor body and pulley

    # --- Mill pulley (on rotor shaft, outside housing at drive end -X) ---
    mill_pulley_radius_m: float = 0.10
    mill_pulley_width_m: float = 0.04
    mill_pulley_x_m: float = -0.03   # X position: negative = outside housing (drive end)

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

        - Mill pulley outside housing in -X (drive end) with allowance.
        - Motor pulley at same X as mill pulley for positional alignment (belt plane).
        - Motor placed so its shaft extends to that pulley X with allowance.
        """
        params = cls.from_mill_config(config)
        # Scale factor for clearances (reference: 0.20m pilot scale)
        scale = config.housing_inner_radius_m / 0.20
        # Mill pulley outside housing in -X with scaled allowance
        params.mill_pulley_x_m = -(
            housing_params.end_plate_thickness_m
            + params.mill_pulley_width_m / 2.0
            + 0.04 * scale  # Scaled clearance between end plate and pulley
        )
        # Motor pulley at same X as mill pulley for coordinate alignment (same belt plane)
        params.motor_pulley_x_m = params.mill_pulley_x_m
        # Motor position: shaft extends from motor end to pulley X; scaled shaft allowance
        shaft_allowance_m = 0.05 * scale
        params.motor_x_offset_m = (
            params.mill_pulley_x_m - shaft_allowance_m - params.motor_length_m
        )
        # Motor beside housing with scaled gap
        params.motor_z_offset_m = (
            housing_params.outer_radius_m + params.motor_width_m / 2.0 + 0.02 * scale
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

    Creates motor, base plate, and pulleys for visualization.
    The motor is positioned beside the housing at the drive end of the rotor.

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
        and shaft pulley. Mill pulley sits at housing drive end.

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

        # --- Motor output shaft: extends in +X from motor body to pulley (allowance between body and pulley) ---
        pulley_x = p.motor_pulley_x_m
        shaft_radius = 0.02
        shaft_start_x = motor_x + body_length
        motor_shaft = cylinder_mesh(
            center=(shaft_start_x, motor_y, motor_z),
            radius=shaft_radius,
            height=pulley_x - shaft_start_x,
            resolution=resolution,
            axis="x",
        )
        parts.append(motor_shaft)

        # --- Motor shaft pulley (at motor_pulley_x_m for allowance from motor body) ---
        pulley_y = motor_y
        pulley_z = motor_z

        motor_pulley = self._v_groove_pulley_mesh(
            center=(pulley_x, pulley_y, pulley_z),
            radius=p.pulley_radius_m,
            width_m=p.pulley_width_m,
            resolution=resolution,
            axis="x",
        )
        parts.append(motor_pulley)

        # --- Mill pulley (on rotor shaft outside housing at drive end, -X; same side as motor) ---
        mill_pulley_x = p.mill_pulley_x_m
        mill_pulley = self._v_groove_pulley_mesh(
            center=(mill_pulley_x, 0.0, 0.0),
            radius=p.mill_pulley_radius_m,
            width_m=p.mill_pulley_width_m,
            resolution=resolution,
            axis="x",
        )
        parts.append(mill_pulley)

        # --- Belt (loop in Y-Z plane at pulley X; both pulleys aligned at same X) ---
        parts.append(self._belt_mesh(p, pulley_x, motor_y, motor_z, resolution))

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
            - drive_shaft: motor output shaft (to pulley)
            - drive_base: base plate
            - drive_feet: motor feet (all four)
            - drive_pulley_motor: motor shaft pulley
            - drive_pulley_mill: mill (rotor) pulley, outside housing
            - drive_belt: belt loop between pulleys
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
        pulley_x = p.motor_pulley_x_m  # allowance from motor body in +X

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

        # Motor output shaft (extends +X from motor body to pulley for allowance)
        shaft_radius = 0.02
        shaft_start_x = motor_x + body_length
        motor_shaft = cylinder_mesh(
            center=(shaft_start_x, motor_y, motor_z),
            radius=shaft_radius,
            height=pulley_x - shaft_start_x,
            resolution=resolution,
            axis="x",
        )
        out["drive_shaft"] = motor_shaft

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

        # Motor pulley (V-groove sheave; motor shaft along X)
        out["drive_pulley_motor"] = self._v_groove_pulley_mesh(
            center=(pulley_x, motor_y, motor_z),
            radius=p.pulley_radius_m,
            width_m=p.pulley_width_m,
            resolution=resolution,
            axis="x",
        )

        # Mill pulley (V-groove sheave on rotor shaft at drive end; on rotor axis)
        out["drive_pulley_mill"] = self._v_groove_pulley_mesh(
            center=(p.mill_pulley_x_m, 0.0, 0.0),
            radius=p.mill_pulley_radius_m,
            width_m=p.mill_pulley_width_m,
            resolution=resolution,
            axis="x",
        )

        # Belt (loop in Y-Z plane at pulley X)
        out["drive_belt"] = self._belt_mesh(p, pulley_x, motor_y, motor_z, resolution)

        return out

    def _v_groove_pulley_mesh(
        self,
        center: Tuple[float, float, float],
        radius: float,
        width_m: float,
        resolution: int,
        axis: str = "z",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """V-groove pulley (sheave) for V-belt drive: hub + two frustums forming the V.

        Standard ~40° included groove angle; groove depth proportional to radius.
        Belt sits in the groove (pitch at groove bottom).
        """
        groove_depth = radius * 0.12
        half_w = width_m / 2
        cx, cy, cz = center
        parts_list: List[Tuple[np.ndarray, np.ndarray]] = []

        # Hub (cylinder at groove bottom radius)
        hub = cylinder_mesh(
            center=center,
            radius=radius - groove_depth,
            height=width_m,
            resolution=resolution,
            axis=axis,
        )
        parts_list.append(hub)

        # Left and right faces of V (frustums: outer radius at rim, groove bottom at centre)
        # Offset for right frustum along the pulley axis
        if axis == "x":
            right_center = (cx + half_w, cy, cz)
        elif axis == "y":
            right_center = (cx, cy + half_w, cz)
        else:
            right_center = (cx, cy, cz + half_w)
        left_f = frustum_mesh(
            center=(cx, cy, cz),
            radius_base=radius,
            radius_top=radius - groove_depth,
            height=half_w,
            resolution=resolution,
            axis=axis,
        )
        right_f = frustum_mesh(
            center=right_center,
            radius_base=radius - groove_depth,
            radius_top=radius,
            height=half_w,
            resolution=resolution,
            axis=axis,
        )
        parts_list.append(left_f)
        parts_list.append(right_f)

        return concat_meshes(parts_list)

    def _belt_mesh(
        self,
        p: "DriveParams",
        pulley_x: float,
        motor_y: float,
        motor_z: float,
        resolution: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Belt as tangent strips + wrap arcs between pulleys in Y-Z plane.

        Both pulleys at same X; belt loop runs in Y-Z plane.
        Mill pulley centered at (0, 0) in Y-Z, motor pulley at (motor_y, motor_z).
        Two straight tangent strips connect the pulleys; wrap arcs close the loop
        around each pulley on the far side.
        """
        groove_frac = 0.12
        R_mill = p.mill_pulley_radius_m * (1 - groove_frac)
        R_motor = p.pulley_radius_m * (1 - groove_frac)
        belt_w = p.pulley_width_m * 0.8
        belt_hw = belt_w / 2
        belt_thick = 0.005  # radial thickness for wrap arcs

        # Groove midpoints along X (pulley center = start_x + width/2)
        mill_gx = p.mill_pulley_x_m + p.mill_pulley_width_m / 2
        motor_gx = pulley_x + p.pulley_width_m / 2

        # Direction from mill to motor center in Y-Z
        dy = motor_y
        dz = motor_z
        d = math.sqrt(dy**2 + dz**2)
        if d < 1e-6:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

        # Angle of center-line (arc convention: Y = R*cos, Z = R*sin)
        phi = math.atan2(dz, dy)
        # Perpendicular unit vector in Y-Z (at angle phi + pi/2)
        py = -dz / d  # -sin(phi)
        pz = dy / d   #  cos(phi)

        parts: List[Tuple[np.ndarray, np.ndarray]] = []

        # --- Tangent strips (two straight runs between pulleys) ---
        for sign in (+1, -1):
            t_mill_y = sign * R_mill * py
            t_mill_z = sign * R_mill * pz
            t_mot_y = motor_y + sign * R_motor * py
            t_mot_z = motor_z + sign * R_motor * pz

            verts = np.array([
                [mill_gx - belt_hw, t_mill_y, t_mill_z],
                [mill_gx + belt_hw, t_mill_y, t_mill_z],
                [motor_gx + belt_hw, t_mot_y, t_mot_z],
                [motor_gx - belt_hw, t_mot_y, t_mot_z],
            ], dtype=np.float32)
            tris = np.array([
                [0, 1, 2], [0, 2, 3],
                [0, 2, 1], [0, 3, 2],
            ], dtype=np.int32)
            parts.append((verts, tris))

        # --- Wrap arcs around each pulley (close the belt loop) ---
        # Mill pulley: far side from motor (phi+pi/2 → phi+3pi/2)
        parts.append(arc_surface_mesh(
            center=(mill_gx - belt_hw, 0.0, 0.0),
            inner_radius=R_mill,
            outer_radius=R_mill + belt_thick,
            start_angle_rad=phi + math.pi / 2,
            end_angle_rad=phi + 3 * math.pi / 2,
            length=belt_w,
            radial_resolution=resolution,
            axial_resolution=2,
            axis="x",
        ))

        # Motor pulley: far side from mill (phi-pi/2 → phi+pi/2)
        parts.append(arc_surface_mesh(
            center=(motor_gx - belt_hw, motor_y, motor_z),
            inner_radius=R_motor,
            outer_radius=R_motor + belt_thick,
            start_angle_rad=phi - math.pi / 2,
            end_angle_rad=phi + math.pi / 2,
            length=belt_w,
            radial_resolution=resolution,
            axial_resolution=2,
            axis="x",
        ))

        return concat_meshes(parts)

    @property
    def ports(self) -> Dict[str, any]:
        """Drive system has no connection ports."""
        return {}
