"""
Feed Chute Geometry
===================

Hammer mill feed chute: the inlet duct that delivers material
from the pretreatment stage (or hopper) to the mill housing.

Physical components:
    - Rectangular or tapered chute
    - Inlet flange (connection to pretreatment/hopper)
    - Outlet flange (connection to housing feed opening)
    - Optional feed gate/valve

Coordinate system:
    X = along rotor axis
    Y = vertical (chute descends from above)
    Z = lateral
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import box_mesh, concat_meshes

if TYPE_CHECKING:
    from ...config import MillConfig
    from .housing import HousingParams


@dataclass
class FeedChuteParams:
    """Feed chute geometry parameters."""

    # --- Chute dimensions ---
    inlet_width_m: float = 0.18                   # Width at inlet (X direction)
    inlet_depth_m: float = 0.15                   # Depth at inlet (Z direction)
    outlet_width_m: float = 0.15                  # Width at outlet
    outlet_depth_m: float = 0.12                  # Depth at outlet
    length_m: float = 0.25                        # Chute length (vertical)
    wall_thickness_m: float = 0.003               # Wall thickness

    # --- Position (outlet connects to housing feed opening) ---
    outlet_x_m: float = 0.15                      # X position of outlet center
    outlet_y_m: float = 0.22                      # Y position of outlet (top of housing)
    outlet_z_m: float = 0.0                       # Z position of outlet center

    # --- Flange dimensions ---
    inlet_flange_width_m: float = 0.025           # Flange width around inlet
    outlet_flange_width_m: float = 0.020          # Flange width around outlet

    @property
    def inlet_y_m(self) -> float:
        """Y position of inlet (top of chute)."""
        return self.outlet_y_m + self.length_m

    @classmethod
    def from_mill_config(cls, config: "MillConfig") -> "FeedChuteParams":
        """Create feed chute params from mill configuration."""
        return cls(
            inlet_width_m=config.feed_chute_width_m + 0.03,
            inlet_depth_m=config.feed_chute_height_m + 0.03,
            outlet_width_m=config.feed_chute_width_m,
            outlet_depth_m=config.feed_chute_height_m,
            length_m=config.feed_chute_length_m,
            outlet_x_m=0.05 + config.feed_chute_width_m / 2 + 0.05,
            outlet_y_m=config.housing_inner_radius_m,
        )

    @classmethod
    def from_housing(cls, housing_params: "HousingParams", config: "MillConfig") -> "FeedChuteParams":
        """Create feed chute params aligned to housing geometry."""
        hp = housing_params
        return cls(
            outlet_width_m=hp.feed_opening_width_m,
            outlet_depth_m=hp.feed_opening_depth_m,
            inlet_width_m=hp.feed_opening_width_m + 0.03,
            inlet_depth_m=hp.feed_opening_depth_m + 0.03,
            length_m=config.feed_chute_length_m,
            outlet_x_m=hp.center_x_m + hp.feed_opening_x_offset_m + hp.feed_opening_width_m / 2,
            outlet_y_m=hp.inner_radius_m,
            outlet_z_m=hp.center_z_m,
        )


class FeedChuteGeometry:
    """Generates feed chute meshes.

    The feed chute is a tapered rectangular duct that delivers
    material to the hammer mill. It connects the pretreatment
    outlet (or hopper) to the housing feed inlet.

    The chute is static (no animation).
    """

    def __init__(self, params: Optional[FeedChuteParams] = None):
        self.params = params or FeedChuteParams()
        self._cached_verts: Optional[np.ndarray] = None
        self._cached_tris: Optional[np.ndarray] = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the feed chute mesh.

        Returns:
            (vertices, triangles, metadata)
        """
        p = self.params
        parts = []

        # Calculate dimensions
        out_w = p.outlet_width_m
        out_d = p.outlet_depth_m
        in_w = p.inlet_width_m
        in_d = p.inlet_depth_m
        t = p.wall_thickness_m

        # Outlet position (bottom of chute)
        out_x = p.outlet_x_m - out_w / 2
        out_y = p.outlet_y_m
        out_z = p.outlet_z_m - out_d / 2

        # Inlet position (top of chute)
        in_x = p.outlet_x_m - in_w / 2
        in_y = p.inlet_y_m
        in_z = p.outlet_z_m - in_d / 2

        # For a tapered chute, we create 4 trapezoidal walls
        # Simplified: use boxes for the walls at outlet dimensions
        # (true taper would require custom quad mesh generation)

        # --- Chute walls (using outlet dimensions, approximation) ---
        avg_w = (out_w + in_w) / 2
        avg_d = (out_d + in_d) / 2
        cx = p.outlet_x_m - avg_w / 2
        cz = p.outlet_z_m - avg_d / 2

        # Front wall (-X side)
        parts.append(box_mesh(
            cx - t, out_y, cz,
            t, p.length_m, avg_d,
        ))

        # Back wall (+X side)
        parts.append(box_mesh(
            cx + avg_w, out_y, cz,
            t, p.length_m, avg_d,
        ))

        # Left wall (-Z side)
        parts.append(box_mesh(
            cx, out_y, cz - t,
            avg_w, p.length_m, t,
        ))

        # Right wall (+Z side)
        parts.append(box_mesh(
            cx, out_y, cz + avg_d,
            avg_w, p.length_m, t,
        ))

        # --- Inlet flange ---
        flange_w = p.inlet_flange_width_m
        flange_t = 0.005

        # Front flange
        parts.append(box_mesh(
            in_x - flange_w, in_y, in_z - flange_w,
            flange_w, flange_t, in_d + 2 * flange_w,
        ))
        # Back flange
        parts.append(box_mesh(
            in_x + in_w, in_y, in_z - flange_w,
            flange_w, flange_t, in_d + 2 * flange_w,
        ))
        # Left flange
        parts.append(box_mesh(
            in_x, in_y, in_z - flange_w,
            in_w, flange_t, flange_w,
        ))
        # Right flange
        parts.append(box_mesh(
            in_x, in_y, in_z + in_d,
            in_w, flange_t, flange_w,
        ))

        self._cached_verts, self._cached_tris = concat_meshes(parts)

        metadata = {
            "type": "feed_chute",
            "animation_type": None,  # Static
            "inlet_width_m": p.inlet_width_m,
            "inlet_depth_m": p.inlet_depth_m,
            "outlet_width_m": p.outlet_width_m,
            "outlet_depth_m": p.outlet_depth_m,
            "length_m": p.length_m,
            "inlet_position": (p.outlet_x_m, p.inlet_y_m, p.outlet_z_m),
            "outlet_position": (p.outlet_x_m, p.outlet_y_m, p.outlet_z_m),
        }

        return self._cached_verts, self._cached_tris, metadata

    @property
    def ports(self) -> Dict[str, Tuple[float, float, float]]:
        """Connection ports."""
        p = self.params
        return {
            "inlet": (p.outlet_x_m, p.inlet_y_m, p.outlet_z_m),
            "outlet": (p.outlet_x_m, p.outlet_y_m, p.outlet_z_m),
        }
