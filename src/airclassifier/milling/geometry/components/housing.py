"""
Housing Geometry
================

Hammer mill housing: the outer casing that encloses the rotor,
hammers, and screen.  Includes the feed inlet opening (top),
circular end plates with bearing bores, and a discharge funnel
(bottom) that connects to a bag collector.

Physical components:
    - Cylindrical shell (top half; screen closes the bottom)
    - Circular end plates with bearing bores
    - Feed inlet flange / collar (top)
    - Discharge funnel with bag-collector flange (bottom)

Coordinate system:
    X = along rotor axis
    Y = vertical (up)
    Z = lateral
    Housing encloses the rotor/hammer assembly

Material flow:
    Feed chute (top) → chamber → screen (bottom) → discharge funnel → bag collector
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import box_mesh, disc_mesh, arc_surface_mesh, concat_meshes

if TYPE_CHECKING:
    from ...config import MillConfig


# Housing shell spans 300° (leaving a 60° slot at the very bottom
# for the discharge opening).  The half-angle is used for both the
# arc mesh and for positioning the discharge funnel top.
_SHELL_HALF_ANGLE = 5 * math.pi / 6   # 150° each side → 300° total


# ---------------------------------------------------------------------------
# Local mesh helper
# ---------------------------------------------------------------------------

def _rect_frustum(
    bot_center: Tuple[float, float, float],
    bot_w: float,
    bot_d: float,
    top_center: Tuple[float, float, float],
    top_w: float,
    top_d: float,
    wall_t: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hollow rectangular frustum (open-ended tapered duct).

    Cross-section changes linearly from *bottom* to *top*.

    Args:
        bot_center: (x, y, z) centre of the bottom rectangle.
        bot_w, bot_d: Bottom width (X) and depth (Z).
        top_center: (x, y, z) centre of the top rectangle.
        top_w, top_d: Top width (X) and depth (Z).
        wall_t: Wall thickness (inward offset).
    """
    bx, by, bz = bot_center
    tx, ty, tz = top_center

    verts = np.array([
        # Outer bottom (0-3)
        [bx - bot_w / 2, by, bz - bot_d / 2],
        [bx + bot_w / 2, by, bz - bot_d / 2],
        [bx + bot_w / 2, by, bz + bot_d / 2],
        [bx - bot_w / 2, by, bz + bot_d / 2],
        # Outer top (4-7)
        [tx - top_w / 2, ty, tz - top_d / 2],
        [tx + top_w / 2, ty, tz - top_d / 2],
        [tx + top_w / 2, ty, tz + top_d / 2],
        [tx - top_w / 2, ty, tz + top_d / 2],
        # Inner bottom (8-11)
        [bx - bot_w / 2 + wall_t, by, bz - bot_d / 2 + wall_t],
        [bx + bot_w / 2 - wall_t, by, bz - bot_d / 2 + wall_t],
        [bx + bot_w / 2 - wall_t, by, bz + bot_d / 2 - wall_t],
        [bx - bot_w / 2 + wall_t, by, bz + bot_d / 2 - wall_t],
        # Inner top (12-15)
        [tx - top_w / 2 + wall_t, ty, tz - top_d / 2 + wall_t],
        [tx + top_w / 2 - wall_t, ty, tz - top_d / 2 + wall_t],
        [tx + top_w / 2 - wall_t, ty, tz + top_d / 2 - wall_t],
        [tx - top_w / 2 + wall_t, ty, tz + top_d / 2 - wall_t],
    ], dtype=np.float32)

    tris = []
    for i in range(4):
        j = (i + 1) % 4
        tris.extend([[i, i + 4, j + 4], [i, j + 4, j]])
        tris.extend([[8 + j, 8 + j + 4, 8 + i + 4],
                     [8 + j, 8 + i + 4, 8 + i]])
        tris.extend([[4 + i, 12 + i, 12 + j],
                     [4 + i, 12 + j, 4 + j]])
        tris.extend([[i, j, 8 + j], [i, 8 + j, 8 + i]])

    return verts, np.array(tris, dtype=np.int32)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class HousingParams:
    """Housing geometry parameters.

    Defines the outer casing of the hammer mill, including the
    discharge funnel for bag-collector connection.
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
    discharge_opening_width_m: float = 0.25       # Width of discharge top (X direction)
    discharge_opening_depth_m: float = 0.18       # Depth of discharge top (Z direction)
    discharge_opening_x_offset_m: float = 0.08    # X offset from housing start

    # --- Discharge funnel (taper to bag collector) ---
    discharge_funnel_height_m: float = 0.15       # Funnel taper height
    discharge_outlet_width_m: float = 0.10        # Bag-collector neck width (X)
    discharge_outlet_depth_m: float = 0.10        # Bag-collector neck depth (Z)

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
        ir = config.housing_inner_radius_m
        return cls(
            inner_radius_m=ir,
            wall_thickness_m=config.housing_wall_thickness_m,
            length_m=config.housing_length_m,
            feed_opening_width_m=config.feed_chute_width_m,
            feed_opening_depth_m=config.feed_chute_height_m,
            discharge_opening_width_m=config.discharge_chute_width_m,
            discharge_opening_depth_m=config.discharge_chute_height_m,
            discharge_funnel_height_m=max(0.12, ir * 0.70),
            discharge_outlet_width_m=config.discharge_chute_width_m * 0.50,
            discharge_outlet_depth_m=config.discharge_chute_height_m * 0.55,
        )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class HousingGeometry:
    """Generates hammer mill housing meshes.

    The housing is a cylindrical casing (top semicircle) with circular
    end plates.  The screen closes the bottom half.  A discharge funnel
    below the screen collects fines and channels them to a bag-collector
    connection flange.

    Static geometry (no animation).
    """

    def __init__(self, params: Optional[HousingParams] = None):
        self.params = params or HousingParams()
        self._cached_verts: Optional[np.ndarray] = None
        self._cached_tris: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_mesh(
        self,
        resolution: int = 24,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate combined housing mesh (backward compatible).

        Returns:
            (vertices, triangles, metadata)
        """
        parts_dict = self.generate_mesh_parts(resolution)
        all_parts = list(parts_dict.values())
        self._cached_verts, self._cached_tris = concat_meshes(all_parts)

        p = self.params
        metadata = {
            "type": "housing",
            "animation_type": None,
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
                "y": p.outer_radius_m * math.cos(_SHELL_HALF_ANGLE),
                "width": p.discharge_opening_width_m,
                "depth": 2 * p.outer_radius_m * math.sin(_SHELL_HALF_ANGLE),
            },
        }

        return self._cached_verts, self._cached_tris, metadata

    def generate_mesh_parts(
        self,
        resolution: int = 24,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Generate housing meshes as separate parts for color-coded rendering.

        Returns:
            Dict mapping part name to (vertices, triangles):
            - housing: cylindrical shell + end plates + feed flange
            - housing_discharge: discharge funnel + bag-collector flange
        """
        p = self.params
        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # ---- Housing shell + end plates + feed flange ----
        shell_parts: List[Tuple[np.ndarray, np.ndarray]] = []

        # Main shell arc: 300° (-150° → +150°)
        # Wraps almost all the way around, leaving a 60° slot at the
        # bottom for the discharge opening.  The screen sits inside
        # the lower portion; the funnel connects at the arc edges.
        top_arc = arc_surface_mesh(
            center=(p.center_x_m, p.center_y_m, p.center_z_m),
            inner_radius=p.inner_radius_m,
            outer_radius=p.outer_radius_m,
            start_angle_rad=-_SHELL_HALF_ANGLE,
            end_angle_rad=_SHELL_HALF_ANGLE,
            length=p.length_m,
            radial_resolution=max(resolution, 32),
            axial_resolution=8,
            axis="x",
        )
        shell_parts.append(top_arc)

        # Circular end plates (disc_mesh with bearing bore)
        shell_parts.append(self._create_end_plate(
            p.center_x_m - p.end_plate_thickness_m, p, resolution,
        ))
        shell_parts.append(self._create_end_plate(
            p.center_x_m + p.length_m, p, resolution,
        ))

        # Feed inlet flange (collar at top of housing)
        shell_parts.append(self._create_feed_flange(p))

        out["housing"] = concat_meshes(shell_parts)

        # ---- Discharge funnel + bag-collector flange ----
        out["housing_discharge"] = self._create_discharge_chute(p)

        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_end_plate(
        self,
        x_pos: float,
        p: HousingParams,
        resolution: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a circular end plate with bearing bore."""
        return disc_mesh(
            center=(x_pos + p.end_plate_thickness_m / 2, p.center_y_m, p.center_z_m),
            inner_radius=p.bearing_bore_radius_m,
            outer_radius=p.outer_radius_m,
            thickness=p.end_plate_thickness_m,
            resolution=resolution,
            axis="x",
        )

    def _create_feed_flange(
        self,
        p: HousingParams,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create the feed inlet flange/collar at the top of the housing."""
        flange_height = 0.03
        flange_wall = 0.008

        x0 = p.center_x_m + p.feed_opening_x_offset_m
        y0 = p.inner_radius_m
        z0 = p.center_z_m - p.feed_opening_depth_m / 2

        parts: List[Tuple[np.ndarray, np.ndarray]] = []

        # Front wall (-X)
        parts.append(box_mesh(
            x0 - flange_wall, y0, z0,
            flange_wall, flange_height, p.feed_opening_depth_m,
        ))
        # Back wall (+X)
        parts.append(box_mesh(
            x0 + p.feed_opening_width_m, y0, z0,
            flange_wall, flange_height, p.feed_opening_depth_m,
        ))
        # Left wall (-Z)
        parts.append(box_mesh(
            x0, y0, z0 - flange_wall,
            p.feed_opening_width_m, flange_height, flange_wall,
        ))
        # Right wall (+Z)
        parts.append(box_mesh(
            x0, y0, z0 + p.feed_opening_depth_m,
            p.feed_opening_width_m, flange_height, flange_wall,
        ))

        return concat_meshes(parts)

    def _create_discharge_chute(
        self,
        p: HousingParams,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create discharge funnel + bag-collector flange below the screen.

        The funnel top connects at the housing arc edge level (where
        the 300° shell ends at ±150°).  At this level the arc edges
        are below the screen bottom, so the funnel sits entirely
        outside the grinding chamber.

        Sections (top → bottom):
            1. Rectangular funnel tapering from discharge opening to outlet
            2. Bag-collector connection flange (bolting ring)
        """
        wall_t = 0.004
        parts: List[Tuple[np.ndarray, np.ndarray]] = []

        # Funnel top centre — at the housing arc edge Y level.
        # At ±150°: Y = R·cos(150°) = -R·√3/2, Z = R·sin(150°) = R/2
        # The hopper spans the full housing length (X) so that the
        # front/back walls close the bottom of the shell arc.
        top_cx = p.center_x_m + p.length_m / 2          # centred on housing
        top_y = p.outer_radius_m * math.cos(_SHELL_HALF_ANGLE)   # negative
        top_cz = p.center_z_m

        # Funnel top spans the full housing extent
        top_width = p.length_m                           # full housing length (X)
        arc_edge_z = p.outer_radius_m * math.sin(_SHELL_HALF_ANGLE)
        top_depth = 2 * arc_edge_z                       # full arc-edge span (Z)

        # Funnel bottom centre (outlet centred below housing)
        bot_cx = p.center_x_m + p.length_m / 2
        bot_y = top_y - p.discharge_funnel_height_m

        # Tapered rectangular duct — wide hopper to narrow outlet
        parts.append(_rect_frustum(
            (top_cx, top_y, top_cz),
            top_width,
            top_depth,
            (bot_cx, bot_y, top_cz),
            p.discharge_outlet_width_m,
            p.discharge_outlet_depth_m,
            wall_t,
        ))

        # Bag-collector flange (wider ring at funnel bottom)
        fw = 0.02   # flange extension beyond outlet
        ft = 0.008  # flange plate thickness
        fy = bot_y - ft
        ow = p.discharge_outlet_width_m
        od = p.discharge_outlet_depth_m

        # Four bars forming the flange frame
        # Front (-X)
        parts.append(box_mesh(
            bot_cx - ow / 2 - fw, fy, top_cz - od / 2 - fw,
            fw, ft, od + 2 * fw,
        ))
        # Back (+X)
        parts.append(box_mesh(
            bot_cx + ow / 2, fy, top_cz - od / 2 - fw,
            fw, ft, od + 2 * fw,
        ))
        # Left (-Z)
        parts.append(box_mesh(
            bot_cx - ow / 2, fy, top_cz - od / 2 - fw,
            ow, ft, fw,
        ))
        # Right (+Z)
        parts.append(box_mesh(
            bot_cx - ow / 2, fy, top_cz + od / 2,
            ow, ft, fw,
        ))

        return concat_meshes(parts)

    # ------------------------------------------------------------------
    # Ports
    # ------------------------------------------------------------------

    @property
    def ports(self) -> Dict[str, Tuple[float, float, float]]:
        """Connection ports for feed and discharge."""
        p = self.params
        return {
            "feed_inlet": (
                p.center_x_m + p.feed_opening_x_offset_m + p.feed_opening_width_m / 2,
                p.inner_radius_m + 0.03,
                p.center_z_m,
            ),
            "discharge_outlet": (
                p.center_x_m + p.length_m / 2,
                p.outer_radius_m * math.cos(_SHELL_HALF_ANGLE) - p.discharge_funnel_height_m - 0.008,
                p.center_z_m,
            ),
        }
