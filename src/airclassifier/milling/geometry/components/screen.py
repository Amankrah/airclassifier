"""
Screen Geometry
===============

Hammer mill screen: curved perforated plate at the bottom of the
mill chamber. Particles smaller than the aperture size pass through;
larger particles are retained for further breakage.

Physical components:
    - Curved screen plate (arc section of cylinder)
    - Support ribs (optional)
    - Screen frame/holder (optional)

Coordinate system:
    X = along rotor axis
    Y = vertical (screen is below rotor center)
    Z = lateral
    Screen wraps around bottom portion of chamber

The screen is typically a 180-degree arc (semi-circle) at the bottom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import arc_surface_mesh, box_mesh, concat_meshes

if TYPE_CHECKING:
    from ...config import MillConfig


@dataclass
class ScreenParams:
    """Screen geometry parameters.

    Defines the curved perforated screen at the bottom of the mill.
    """

    # --- Screen dimensions ---
    inner_radius_m: float = 0.21                  # Inside radius of screen
    thickness_m: float = 0.003                    # Screen plate thickness
    length_m: float = 0.30                        # Length along rotor axis

    # --- Arc geometry ---
    arc_angle_deg: float = 180.0                  # Angular extent (180 = semicircle)
    start_angle_deg: float = 90.0                 # Start angle (90 = +Z, wraps bottom)

    # --- Position ---
    center_x_m: float = 0.05                      # X position of screen start
    center_y_m: float = 0.0                       # Y position (rotor centerline)
    center_z_m: float = 0.0                       # Z position (rotor centerline)

    # --- Aperture (for physics, not geometry) ---
    aperture_mm: float = 1.5                      # Hole diameter
    open_area: float = 0.40                       # Open area fraction

    # --- Support ribs ---
    rib_count: int = 0                            # Number of support ribs (0 = none)
    rib_width_m: float = 0.010                    # Rib width
    rib_height_m: float = 0.015                   # Rib height (radial)

    @property
    def outer_radius_m(self) -> float:
        """Outer radius of screen."""
        return self.inner_radius_m + self.thickness_m

    @property
    def arc_angle_rad(self) -> float:
        """Arc angle in radians."""
        return math.radians(self.arc_angle_deg)

    @property
    def start_angle_rad(self) -> float:
        """Start angle in radians."""
        return math.radians(self.start_angle_deg)

    @property
    def end_angle_rad(self) -> float:
        """End angle in radians."""
        return self.start_angle_rad + self.arc_angle_rad

    @classmethod
    def from_mill_config(cls, config: "MillConfig") -> "ScreenParams":
        """Create screen params from mill configuration."""
        return cls(
            inner_radius_m=config.screen_inner_radius_m,
            thickness_m=config.screen_thickness_m,
            length_m=config.rotor_length_m,
            arc_angle_deg=config.screen_arc_angle_deg,
            aperture_mm=config.screen_aperture_mm,
            open_area=config.screen_open_area,
            center_x_m=0.05,  # Slight offset into housing
        )


class ScreenGeometry:
    """Generates hammer mill screen meshes.

    The screen is a curved perforated surface. For visualization,
    we render it as a solid curved shell (perforations are too small
    to model geometrically). The aperture info is in metadata for
    physics calculations.

    The screen is static (no animation).
    """

    def __init__(self, params: Optional[ScreenParams] = None):
        self.params = params or ScreenParams()
        self._cached_verts: Optional[np.ndarray] = None
        self._cached_tris: Optional[np.ndarray] = None

    def generate_mesh(
        self,
        radial_resolution: int = 24,
        axial_resolution: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the screen mesh.

        Args:
            radial_resolution: Number of angular segments.
            axial_resolution: Number of segments along length.

        Returns:
            (vertices, triangles, metadata)
        """
        p = self.params
        parts = []

        # --- Main screen arc ---
        screen_verts, screen_tris = arc_surface_mesh(
            center=(p.center_x_m, p.center_y_m, p.center_z_m),
            inner_radius=p.inner_radius_m,
            outer_radius=p.outer_radius_m,
            start_angle_rad=p.start_angle_rad,
            end_angle_rad=p.end_angle_rad,
            length=p.length_m,
            radial_resolution=radial_resolution,
            axial_resolution=axial_resolution,
            axis="x",
        )
        parts.append((screen_verts, screen_tris))

        # --- Screen frame bars at arc edges (screen cradle) ---
        frame_w = 0.008
        frame_h = 0.012
        for edge_angle in (p.start_angle_rad, p.end_angle_rad):
            ey = p.center_y_m + p.outer_radius_m * math.cos(edge_angle)
            ez = p.center_z_m + p.outer_radius_m * math.sin(edge_angle)
            parts.append(box_mesh(
                p.center_x_m,
                ey - frame_h / 2,
                ez - frame_w / 2,
                p.length_m, frame_h, frame_w,
            ))

        # --- Support ribs (radial bars under screen) ---
        if p.rib_count > 0:
            rib_spacing = p.length_m / (p.rib_count + 1)
            for i in range(p.rib_count):
                x_pos = p.center_x_m + (i + 1) * rib_spacing
                # Rib at bottom of screen (Y = -outer_radius)
                rib_verts, rib_tris = box_mesh(
                    x_pos - p.rib_width_m / 2.0,
                    -p.outer_radius_m - p.rib_height_m,
                    p.center_z_m - p.inner_radius_m * 0.8,
                    p.rib_width_m,
                    p.rib_height_m,
                    p.inner_radius_m * 1.6,
                )
                parts.append((rib_verts, rib_tris))

        self._cached_verts, self._cached_tris = concat_meshes(parts)

        metadata = {
            "type": "screen",
            "animation_type": None,  # Static
            "inner_radius_m": p.inner_radius_m,
            "outer_radius_m": p.outer_radius_m,
            "length_m": p.length_m,
            "arc_angle_deg": p.arc_angle_deg,
            "aperture_mm": p.aperture_mm,
            "open_area": p.open_area,
            "center": (p.center_x_m, p.center_y_m, p.center_z_m),
        }

        return self._cached_verts, self._cached_tris, metadata

    def get_screen_surface_sdf(self, point: np.ndarray) -> float:
        """Compute signed distance from a point to screen surface.

        Negative inside screen shell, positive outside.

        Args:
            point: [x, y, z] position

        Returns:
            Signed distance to screen surface [m]
        """
        p = self.params
        # Transform to screen-local coords (relative to arc center)
        local_x = point[0] - p.center_x_m
        local_y = point[1] - p.center_y_m
        local_z = point[2] - p.center_z_m

        # Check if within axial bounds
        if local_x < 0 or local_x > p.length_m:
            return float('inf')  # Outside screen length

        # Radial distance in YZ plane
        r = math.sqrt(local_y ** 2 + local_z ** 2)

        # Angle in YZ plane
        angle = math.atan2(local_z, local_y)

        # Check if within arc angle
        if angle < p.start_angle_rad or angle > p.end_angle_rad:
            return float('inf')  # Outside screen arc

        # Signed distance: negative if between inner and outer radius
        if r < p.inner_radius_m:
            return p.inner_radius_m - r  # Inside screen
        elif r > p.outer_radius_m:
            return r - p.outer_radius_m  # Outside screen
        else:
            # Within screen shell
            return min(r - p.inner_radius_m, p.outer_radius_m - r) * -1.0

    @property
    def ports(self) -> Dict[str, any]:
        """Screen has no external connection ports."""
        return {}
