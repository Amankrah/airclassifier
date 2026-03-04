"""
Feed Chute / Hopper Geometry
============================

Realistic gravity-fed hammer-mill hopper for pouring seeds
(e.g. yellow peas) into the mill housing.

Physical sections (bottom → top):
    - Outlet flange    – bolted ring connecting to housing feed opening
    - Throat           – short straight duct matching the housing opening
    - Transition       – tapered section widening from throat to hopper body
    - Hopper body      – open-top rectangular bin with slight outward draft
    - Top rim          – folded/rolled lip for rigidity and safe handling

Design notes:
    - Transition wall angle ≥ 60° from horizontal to ensure mass-flow
      of granular seeds (~8 mm yellow peas, angle of repose ~28°).
    - Hopper body has slight outward draft so poured material doesn't
      bridge against vertical walls.

Coordinate system:
    X = along rotor axis
    Y = vertical  (hopper rises above housing)
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


# ---------------------------------------------------------------------------
# Mesh helper
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

    Generates a rectangular duct section whose cross-section changes
    linearly from *bottom* to *top*, with a constant wall thickness.

    Args:
        bot_center: (x, y, z) centre of the bottom rectangle.
        bot_w, bot_d: Bottom width (X) and depth (Z).
        top_center: (x, y, z) centre of the top rectangle.
        top_w, top_d: Top width (X) and depth (Z).
        wall_t: Wall thickness (inward offset).

    Returns:
        (vertices [16, 3], triangles [N, 3])
    """
    bx, by, bz = bot_center
    tx, ty, tz = top_center

    verts = np.array([
        # Outer bottom (0-3): -X-Z, +X-Z, +X+Z, -X+Z
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
        # Outer faces (normals point outward)
        tris.extend([[i, i + 4, j + 4], [i, j + 4, j]])
        # Inner faces (normals point inward — visible from inside)
        tris.extend([[8 + j, 8 + j + 4, 8 + i + 4],
                     [8 + j, 8 + i + 4, 8 + i]])
        # Top rim strip (normals up)
        tris.extend([[4 + i, 12 + i, 12 + j],
                     [4 + i, 12 + j, 4 + j]])
        # Bottom rim strip (normals down)
        tris.extend([[i, j, 8 + j], [i, 8 + j, 8 + i]])

    return verts, np.array(tris, dtype=np.int32)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class FeedChuteParams:
    """Realistic feed-hopper geometry parameters.

    Sections (bottom → top):
        outlet flange → throat → transition → hopper body → rim
    """

    # --- Throat (matches housing feed opening) ---
    throat_width_m: float = 0.15            # Width at feed opening (X)
    throat_depth_m: float = 0.12            # Depth at feed opening (Z)
    throat_height_m: float = 0.035          # Short straight section

    # --- Transition (tapered from throat to hopper) ---
    transition_height_m: float = 0.15       # Height of tapered section

    # --- Hopper body (open-top bin) ---
    hopper_width_m: float = 0.30            # Width at hopper bottom (X)
    hopper_depth_m: float = 0.25            # Depth at hopper bottom (Z)
    hopper_height_m: float = 0.20           # Vertical height of the bin
    hopper_draft_m: float = 0.005           # Outward draft per side at top

    # --- Wall & rim ---
    wall_thickness_m: float = 0.003         # 3 mm steel sheet
    rim_width_m: float = 0.012              # Folded lip width
    rim_height_m: float = 0.015             # Folded lip height

    # --- Outlet flange (bolts to housing) ---
    flange_width_m: float = 0.020           # Flange ring width
    flange_thickness_m: float = 0.006       # Flange plate thickness

    # --- Position (outlet centre, bottom of throat) ---
    outlet_x_m: float = 0.15               # X centre of the opening
    outlet_y_m: float = 0.22               # Y position (top of housing)
    outlet_z_m: float = 0.0                # Z centre of the opening

    # ---- Derived heights ----

    @property
    def throat_top_y(self) -> float:
        return self.outlet_y_m + self.throat_height_m

    @property
    def transition_top_y(self) -> float:
        return self.throat_top_y + self.transition_height_m

    @property
    def hopper_top_y(self) -> float:
        return self.transition_top_y + self.hopper_height_m

    @property
    def rim_top_y(self) -> float:
        return self.hopper_top_y + self.rim_height_m

    @property
    def inlet_y_m(self) -> float:
        """Y position of the top of the hopper (where seeds are poured)."""
        return self.rim_top_y

    @property
    def total_height_m(self) -> float:
        return self.inlet_y_m - self.outlet_y_m

    @classmethod
    def from_mill_config(cls, config: "MillConfig") -> "FeedChuteParams":
        """Create hopper params from mill configuration."""
        tw = config.feed_chute_width_m
        td = config.feed_chute_height_m
        # Scale factor based on housing radius (reference: 0.20m pilot scale)
        scale = config.housing_inner_radius_m / 0.20
        return cls(
            throat_width_m=tw,
            throat_depth_m=td,
            throat_height_m=0.035 * scale,
            hopper_width_m=tw * 2.0,
            hopper_depth_m=td * 2.0,
            hopper_height_m=0.20 * scale,
            hopper_draft_m=0.005 * scale,
            transition_height_m=max(tw, td) * 1.0,
            wall_thickness_m=0.003 * scale,
            rim_width_m=0.012 * scale,
            rim_height_m=0.015 * scale,
            flange_width_m=0.020 * scale,
            flange_thickness_m=0.006 * scale,
            outlet_x_m=0.05 * scale + tw / 2 + 0.05 * scale,
            outlet_y_m=config.housing_inner_radius_m,
        )

    @classmethod
    def from_housing(
        cls,
        housing_params: "HousingParams",
        config: "MillConfig",
    ) -> "FeedChuteParams":
        """Create hopper params aligned to housing feed opening."""
        hp = housing_params
        tw = hp.feed_opening_width_m
        td = hp.feed_opening_depth_m
        # Scale factor based on housing radius (reference: 0.20m pilot scale)
        scale = hp.inner_radius_m / 0.20
        return cls(
            throat_width_m=tw,
            throat_depth_m=td,
            throat_height_m=0.035 * scale,
            hopper_width_m=tw * 2.0,
            hopper_depth_m=td * 2.0,
            hopper_height_m=0.20 * scale,
            hopper_draft_m=0.005 * scale,
            transition_height_m=max(tw, td) * 1.0,
            wall_thickness_m=0.003 * scale,
            rim_width_m=0.012 * scale,
            rim_height_m=0.015 * scale,
            flange_width_m=0.020 * scale,
            flange_thickness_m=0.006 * scale,
            # Note: hp.feed_opening_x_offset_m is already scaled in HousingParams
            outlet_x_m=(
                hp.center_x_m
                + hp.feed_opening_x_offset_m
                + tw / 2
            ),
            outlet_y_m=hp.inner_radius_m,
            outlet_z_m=hp.center_z_m,
        )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class FeedChuteGeometry:
    """Generates a realistic feed-hopper mesh.

    The hopper is an open-top rectangular bin with tapered walls
    that funnels poured material (seeds) into the hammer-mill
    housing through a straight throat section.

    Static geometry (no animation).
    """

    def __init__(self, params: Optional[FeedChuteParams] = None):
        self.params = params or FeedChuteParams()
        self._cached_verts: Optional[np.ndarray] = None
        self._cached_tris: Optional[np.ndarray] = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the feed-hopper mesh.

        Returns:
            (vertices, triangles, metadata)
        """
        p = self.params
        parts = []

        cx = p.outlet_x_m
        cz = p.outlet_z_m
        t = p.wall_thickness_m
        tw, td = p.throat_width_m, p.throat_depth_m

        # ── 1. Outlet flange ──────────────────────────────────────
        fw = p.flange_width_m
        ft = p.flange_thickness_m
        fy = p.outlet_y_m - ft          # flange sits just below throat

        # Four bars forming a rectangular frame around the opening
        # Front bar (-X)
        parts.append(box_mesh(
            cx - tw / 2 - fw, fy, cz - td / 2 - fw,
            fw, ft, td + 2 * fw,
        ))
        # Back bar (+X)
        parts.append(box_mesh(
            cx + tw / 2, fy, cz - td / 2 - fw,
            fw, ft, td + 2 * fw,
        ))
        # Left bar (-Z)
        parts.append(box_mesh(
            cx - tw / 2, fy, cz - td / 2 - fw,
            tw, ft, fw,
        ))
        # Right bar (+Z)
        parts.append(box_mesh(
            cx - tw / 2, fy, cz + td / 2,
            tw, ft, fw,
        ))

        # ── 2. Throat (straight duct) ────────────────────────────
        parts.append(_rect_frustum(
            (cx, p.outlet_y_m, cz), tw, td,
            (cx, p.throat_top_y, cz), tw, td,
            t,
        ))

        # ── 3. Transition (tapered from throat → hopper) ─────────
        hw, hd = p.hopper_width_m, p.hopper_depth_m
        parts.append(_rect_frustum(
            (cx, p.throat_top_y, cz), tw, td,
            (cx, p.transition_top_y, cz), hw, hd,
            t,
        ))

        # ── 4. Hopper body (slight outward draft) ────────────────
        draft = p.hopper_draft_m
        parts.append(_rect_frustum(
            (cx, p.transition_top_y, cz), hw, hd,
            (cx, p.hopper_top_y, cz), hw + 2 * draft, hd + 2 * draft,
            t,
        ))

        # ── 5. Top rim / rolled lip ──────────────────────────────
        rw = p.rim_width_m
        rh = p.rim_height_m
        top_hw = hw + 2 * draft
        top_hd = hd + 2 * draft
        ry = p.hopper_top_y

        # Four thicker bars forming the rim around the hopper top
        # Front rim (-X)
        parts.append(box_mesh(
            cx - top_hw / 2 - rw, ry, cz - top_hd / 2 - rw,
            rw, rh, top_hd + 2 * rw,
        ))
        # Back rim (+X)
        parts.append(box_mesh(
            cx + top_hw / 2, ry, cz - top_hd / 2 - rw,
            rw, rh, top_hd + 2 * rw,
        ))
        # Left rim (-Z)
        parts.append(box_mesh(
            cx - top_hw / 2, ry, cz - top_hd / 2 - rw,
            top_hw, rh, rw,
        ))
        # Right rim (+Z)
        parts.append(box_mesh(
            cx - top_hw / 2, ry, cz + top_hd / 2,
            top_hw, rh, rw,
        ))

        # ── Combine ──────────────────────────────────────────────
        self._cached_verts, self._cached_tris = concat_meshes(parts)

        metadata = {
            "type": "feed_chute",
            "animation_type": None,     # Static
            "throat_width_m": tw,
            "throat_depth_m": td,
            "hopper_width_m": hw,
            "hopper_depth_m": hd,
            "total_height_m": p.total_height_m,
            "inlet_position": (cx, p.inlet_y_m, cz),
            "outlet_position": (cx, p.outlet_y_m, cz),
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
