"""
Housing Geometry
================

Hammer mill housing: the outer casing that encloses the rotor,
hammers, and screen. Includes the feed inlet opening and discharge
outlet.

Physical components:
    - Cylindrical or box-shaped outer casing
    - End plates (with bearing mounts)
    - Feed inlet opening (top)
    - Discharge opening (bottom, below screen)
    - Access doors/panels (optional)

Coordinate system:
    X = along rotor axis
    Y = vertical (up)
    Z = lateral
    Housing encloses the rotor/hammer assembly
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import cylinder_mesh, box_mesh, arc_surface_mesh, concat_meshes

if TYPE_CHECKING:
    from ...config import MillConfig


@dataclass
class HousingParams:
    """Housing geometry parameters.

    Defines the outer casing of the hammer mill.
    """

    # --- Main casing ---
    inner_radius_m: float = 0.22                  # Inside radius of cylindrical section
    wall_thickness_m: float = 0.008               # Wall thickness
    length_m: float = 0.40                        # Total housing length

    # --- End plates ---
    end_plate_thickness_m: float = 0.015          # Thickness of end plates
    bearing_bore_radius_m: float = 0.04           # Bore for shaft bearing

    # --- Feed inlet (top) ---
    feed_opening_width_m: float = 0.15            # Width of feed opening (X direction)
    feed_opening_depth_m: float = 0.12            # Depth of feed opening (Z direction)
    feed_opening_x_offset_m: float = 0.10         # X offset from housing start

    # --- Discharge outlet (bottom) ---
    discharge_opening_width_m: float = 0.25       # Width of discharge (X direction)
    discharge_opening_depth_m: float = 0.18       # Depth of discharge (Z direction)
    discharge_opening_x_offset_m: float = 0.08    # X offset from housing start

    # --- Position ---
    center_x_m: float = 0.0                       # X position of housing start
    center_y_m: float = 0.0                       # Y position (rotor centerline)
    center_z_m: float = 0.0                       # Z position (rotor centerline)

    @property
    def outer_radius_m(self) -> float:
        """Outer radius of housing."""
        return self.inner_radius_m + self.wall_thickness_m

    @classmethod
    def from_mill_config(cls, config: "MillConfig") -> "HousingParams":
        """Create housing params from mill configuration."""
        return cls(
            inner_radius_m=config.housing_inner_radius_m,
            wall_thickness_m=config.housing_wall_thickness_m,
            length_m=config.housing_length_m,
            feed_opening_width_m=config.feed_chute_width_m,
            feed_opening_depth_m=config.feed_chute_height_m,
            discharge_opening_width_m=config.discharge_chute_width_m,
            discharge_opening_depth_m=config.discharge_chute_height_m,
        )


class HousingGeometry:
    """Generates hammer mill housing meshes.

    The housing is a cylindrical casing with openings for feed and
    discharge. The screen fits inside the bottom portion.

    For visualization, we create:
    - Top half of cylindrical shell (above screen level)
    - End plates with bearing bores
    - Flange around feed opening
    - Discharge chute flange

    The housing is static (no animation).
    """

    def __init__(self, params: Optional[HousingParams] = None):
        self.params = params or HousingParams()
        self._cached_verts: Optional[np.ndarray] = None
        self._cached_tris: Optional[np.ndarray] = None

    def generate_mesh(
        self,
        resolution: int = 24,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the housing mesh.

        Args:
            resolution: Number of angular segments.

        Returns:
            (vertices, triangles, metadata)
        """
        p = self.params
        parts = []

        # --- Top arc of housing (above rotor) ---
        # Arc from 0 to 180 degrees (top half)
        top_arc = arc_surface_mesh(
            center=(p.center_x_m, p.center_y_m, p.center_z_m),
            inner_radius=p.inner_radius_m,
            outer_radius=p.outer_radius_m,
            start_angle_rad=0.0,
            end_angle_rad=math.pi,
            length=p.length_m,
            radial_resolution=resolution // 2,
            axial_resolution=8,
            axis="x",
        )
        parts.append(top_arc)

        # --- Side walls (vertical sections on each side) ---
        # Left side wall (-Z side)
        left_wall = box_mesh(
            p.center_x_m,
            -p.inner_radius_m,
            p.center_z_m - p.outer_radius_m,
            p.length_m,
            p.inner_radius_m * 2,
            p.wall_thickness_m,
        )
        parts.append(left_wall)

        # Right side wall (+Z side)
        right_wall = box_mesh(
            p.center_x_m,
            -p.inner_radius_m,
            p.center_z_m + p.inner_radius_m,
            p.length_m,
            p.inner_radius_m * 2,
            p.wall_thickness_m,
        )
        parts.append(right_wall)

        # --- End plates ---
        # Drive end plate (X = 0)
        drive_end = self._create_end_plate(
            p.center_x_m - p.end_plate_thickness_m,
            p,
            resolution,
        )
        parts.append(drive_end)

        # Free end plate
        free_end = self._create_end_plate(
            p.center_x_m + p.length_m,
            p,
            resolution,
        )
        parts.append(free_end)

        # --- Feed inlet flange ---
        feed_flange = self._create_feed_flange(p)
        parts.append(feed_flange)

        # --- Discharge flange ---
        discharge_flange = self._create_discharge_flange(p)
        parts.append(discharge_flange)

        self._cached_verts, self._cached_tris = concat_meshes(parts)

        metadata = {
            "type": "housing",
            "animation_type": None,  # Static
            "inner_radius_m": p.inner_radius_m,
            "outer_radius_m": p.outer_radius_m,
            "length_m": p.length_m,
            "feed_opening": {
                "x": p.center_x_m + p.feed_opening_x_offset_m,
                "y": p.inner_radius_m,
                "width": p.feed_opening_width_m,
                "depth": p.feed_opening_depth_m,
            },
            "discharge_opening": {
                "x": p.center_x_m + p.discharge_opening_x_offset_m,
                "y": -p.inner_radius_m,
                "width": p.discharge_opening_width_m,
                "depth": p.discharge_opening_depth_m,
            },
        }

        return self._cached_verts, self._cached_tris, metadata

    def _create_end_plate(
        self,
        x_pos: float,
        p: HousingParams,
        resolution: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create an end plate with bearing bore."""
        # Simplified: rectangular end plate
        plate = box_mesh(
            x_pos,
            -p.outer_radius_m,
            p.center_z_m - p.outer_radius_m,
            p.end_plate_thickness_m,
            p.outer_radius_m * 2,
            p.outer_radius_m * 2,
        )
        return plate

    def _create_feed_flange(
        self,
        p: HousingParams,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create the feed inlet flange/collar."""
        flange_height = 0.03
        flange_wall = 0.008

        x0 = p.center_x_m + p.feed_opening_x_offset_m
        y0 = p.inner_radius_m
        z0 = p.center_z_m - p.feed_opening_depth_m / 2

        parts = []

        # Front wall
        parts.append(box_mesh(
            x0 - flange_wall, y0, z0,
            flange_wall, flange_height, p.feed_opening_depth_m,
        ))
        # Back wall
        parts.append(box_mesh(
            x0 + p.feed_opening_width_m, y0, z0,
            flange_wall, flange_height, p.feed_opening_depth_m,
        ))
        # Left wall
        parts.append(box_mesh(
            x0, y0, z0 - flange_wall,
            p.feed_opening_width_m, flange_height, flange_wall,
        ))
        # Right wall
        parts.append(box_mesh(
            x0, y0, z0 + p.feed_opening_depth_m,
            p.feed_opening_width_m, flange_height, flange_wall,
        ))

        return concat_meshes(parts)

    def _create_discharge_flange(
        self,
        p: HousingParams,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create the discharge outlet flange/collar."""
        flange_height = 0.05
        flange_wall = 0.008

        x0 = p.center_x_m + p.discharge_opening_x_offset_m
        y0 = -p.outer_radius_m - flange_height
        z0 = p.center_z_m - p.discharge_opening_depth_m / 2

        parts = []

        # Front wall
        parts.append(box_mesh(
            x0 - flange_wall, y0, z0,
            flange_wall, flange_height, p.discharge_opening_depth_m,
        ))
        # Back wall
        parts.append(box_mesh(
            x0 + p.discharge_opening_width_m, y0, z0,
            flange_wall, flange_height, p.discharge_opening_depth_m,
        ))
        # Left wall
        parts.append(box_mesh(
            x0, y0, z0 - flange_wall,
            p.discharge_opening_width_m, flange_height, flange_wall,
        ))
        # Right wall
        parts.append(box_mesh(
            x0, y0, z0 + p.discharge_opening_depth_m,
            p.discharge_opening_width_m, flange_height, flange_wall,
        ))

        return concat_meshes(parts)

    @property
    def ports(self) -> Dict[str, Tuple[float, float, float]]:
        """Connection ports for feed and discharge."""
        p = self.params
        return {
            "feed_inlet": (
                p.center_x_m + p.feed_opening_x_offset_m + p.feed_opening_width_m / 2,
                p.inner_radius_m + 0.03,  # Above housing top
                p.center_z_m,
            ),
            "discharge_outlet": (
                p.center_x_m + p.discharge_opening_x_offset_m + p.discharge_opening_width_m / 2,
                -p.outer_radius_m - 0.05,  # Below housing bottom
                p.center_z_m,
            ),
        }
