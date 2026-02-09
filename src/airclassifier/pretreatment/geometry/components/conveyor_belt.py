"""
Conveyor Belt Geometry
======================

Belt geometry, material bed mesh, and advection domain.

This is a refactored version of the original conveyor.py with added
ConnectionPort support for assembly alignment.

The conveyor moves material continuously through the oven along the
positive X-axis. The belt rides on Teflon wear strips above the lower
electrode. Material is deposited at the infeed and exits at the outfeed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import box_mesh, concat_meshes, cylinder_mesh

if TYPE_CHECKING:
    from airclassifier.geometry.connection_ports import ConnectionPort
    from ...config import MachineConfig


@dataclass
class ConveyorBeltParams:
    """GP-15 conveyor belt geometry parameters (from manual + engineering guide).

    The conveyor is the *entire base assembly* upon which the applicator
    (oven) sits.  It provides structural support and guidance for the belt.

    Reference dimensions (Manual Appendix B, Engineering Guide §2.2–2.3):
      Machine envelope:   5.5 × 2.9 × 2.2 m  (L × W × H overall)
      Belt width (usable): 800 mm
      Belt total loop:     ~16 m
      Belt type:           plain reinforced PTFE (Teflon), modular link
      Belt speed:          0.1–2.0 m/min
      Conveyor drive:      0.75 kW, inverter-controlled, direct drive
      Belt tension:        pneumatic gravity roller, 4 bar / 60 psi
    """

    # ── Structural frame ─────────────────────────────────────────────
    frame_length_m: float = 5.50        # [m]  Overall frame length (X)
    frame_width_m: float = 1.10         # [m]  Overall frame width  (Z) — wider than belt
    frame_height_m: float = 0.60        # [m]  Frame depth below belt top (Y down)
    leg_height_m: float = 0.70          # [m]  Support legs below frame bottom

    # Steel section profiles (box section / C-channel)
    rail_section_m: float = 0.06        # [m]  Side-rail cross-section height & width
    cross_member_w_m: float = 0.04      # [m]  Cross member section width
    cross_member_h_m: float = 0.06      # [m]  Cross member section height
    leg_section_m: float = 0.06         # [m]  Leg square section side

    # ── Belt ─────────────────────────────────────────────────────────
    belt_width_m: float = 0.80          # [m]  Usable belt width (Manual: 800 mm)
    belt_thickness_m: float = 0.002     # [m]  ~2 mm (estimated, TBD — MEASURE)
    wear_strip_thickness_m: float = 0.001  # [m] Teflon wear strips ~1 mm
    top_sheet_thickness_m: float = 0.0005  # [m] Top sheet ~0.5 mm

    # ── Roller system (from manual belt-path diagram) ────────────────
    # Head (drive) roller — outfeed end, motor-driven
    head_roller_radius_m: float = 0.075   # [m]  ~150 mm diameter drive roller
    # Tail (nose) roller — infeed end, free-spinning
    tail_roller_radius_m: float = 0.060   # [m]  ~120 mm diameter nose roller
    # Gravity tension roller — pneumatic, weighted, below return path
    tension_roller_radius_m: float = 0.075  # [m]  ~150 mm dia, pneumatically loaded
    tension_drop_m: float = 0.25          # [m]  Drop below return run (gravity sag)
    # Belt tracking rollers — 2-roller assembly on return path
    tracker_roller_radius_m: float = 0.035  # [m]  ~70 mm dia tracking rollers
    # Return path support rollers
    return_roller_radius_m: float = 0.030   # [m]  ~60 mm dia support rollers
    return_roller_count: int = 4            # Number of return support rollers
    # Top carrying idlers (between infeed/outfeed within oven)
    carrying_idler_radius_m: float = 0.025  # [m]  ~50 mm dia — belt mostly rides on electrode
    carrying_idler_count: int = 3           # Sparse — belt supported by lower electrode trays

    roller_resolution: int = 20             # Cylinder mesh segments

    # ── End geometry ─────────────────────────────────────────────────
    # Nose plates at infeed/outfeed where belt wraps around head/tail
    nose_length_m: float = 0.20           # [m]  Horizontal extent of nose section
    nose_drop_m: float = 0.15             # [m]  Vertical drop at nose

    # ── Attenuation ducts ────────────────────────────────────────────
    duct_length_m: float = 0.40           # [m]  RF attenuation duct length each end
    duct_height_m: float = 0.10           # [m]  Duct opening height
    duct_wall_thickness_m: float = 0.003  # [m]  Sheet metal walls

    # ── Position ─────────────────────────────────────────────────────
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_machine(cls, config: "MachineConfig") -> "ConveyorBeltParams":
        """Create params from MachineConfig."""
        return cls(
            belt_thickness_m=config.belt_thickness_m,
            wear_strip_thickness_m=config.wear_strip_thickness_m,
            top_sheet_thickness_m=config.top_sheet_thickness_m,
        )

    @property
    def belt_stack_thickness_m(self) -> float:
        """Total belt stack: belt + wear strips + top sheet ≈ 3.5 mm."""
        return (self.belt_thickness_m +
                self.wear_strip_thickness_m +
                self.top_sheet_thickness_m)

    @property
    def belt_center_z(self) -> float:
        """Z centre of the belt within the frame."""
        return self.frame_width_m / 2.0

    @property
    def belt_z0(self) -> float:
        """Belt left edge Z."""
        return (self.frame_width_m - self.belt_width_m) / 2.0

    @property
    def belt_z1(self) -> float:
        """Belt right edge Z."""
        return (self.frame_width_m + self.belt_width_m) / 2.0

    # Kept for backward compatibility with the belt mesh generator
    @property
    def bed_length_m(self) -> float:
        return self.frame_length_m

    @property
    def bed_width_m(self) -> float:
        return self.belt_width_m

    @property
    def bed_thickness_m(self) -> float:
        return self.frame_height_m

    @property
    def pulley_radius_m(self) -> float:
        return self.head_roller_radius_m

    @property
    def end_inward_length_m(self) -> float:
        return self.nose_length_m

    @property
    def idler_count(self) -> int:
        return self.carrying_idler_count

    @property
    def pulley_resolution(self) -> int:
        return self.roller_resolution


class ConveyorBeltGeometry:
    """Generates the conveyor belt and material bed meshes.

    This class extends the original ConveyorGeometry with
    ConnectionPort support for hopper and material flow.
    """

    def __init__(self, params: Optional[ConveyorBeltParams] = None):
        self.params = params or ConveyorBeltParams()
        self._belt_vertices: Optional[np.ndarray] = None
        self._belt_triangles: Optional[np.ndarray] = None
        self._bed_vertices: Optional[np.ndarray] = None
        self._bed_triangles: Optional[np.ndarray] = None
        self._bed_structure_vertices: Optional[np.ndarray] = None
        self._bed_structure_triangles: Optional[np.ndarray] = None
        self._wheels_vertices: Optional[np.ndarray] = None
        self._wheels_triangles: Optional[np.ndarray] = None

    def _end_cap_prism_mesh(
        self,
        x0: float,
        x1: float,
        y_top: float,
        y_vertical: float,
        y_bottom: float,
        z0: float,
        z1: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build prism for one end: top, then 15 cm vertical, then bend inwards to bottom.

        Profile in X-Y (left end): (x0, y_top) -> (x0, y_vertical) -> (x1, y_bottom) -> (x1, y_top).
        Extruded along Z from z0 to z1. Vertices: 4 at z=0, 4 at z=z1.
        """
        verts = np.array([
            [x0, y_top, z0],       # 0
            [x0, y_vertical, z0], # 1
            [x1, y_bottom, z0],   # 2
            [x1, y_top, z0],      # 3
            [x0, y_top, z1],
            [x0, y_vertical, z1],
            [x1, y_bottom, z1],
            [x1, y_top, z1],
        ], dtype=np.float32)
        tris = np.array([
            [0, 3, 7], [0, 7, 4],   # top (y_top)
            [0, 1, 2], [0, 2, 3],   # front (z=z0)
            [4, 7, 6], [4, 6, 5],   # back (z=z1)
            [0, 4, 5], [0, 5, 1],   # left (x=x0)
            [3, 2, 6], [3, 6, 7],   # right (x=x1)
            [1, 5, 6], [1, 6, 2],   # bottom (slanted + vertical)
        ], dtype=np.int32)
        return verts, tris

    def generate_bed_structure_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the GP-15 conveyor structural frame.

        Real structure (from manual Illustration 3 + §2.2/2.3):
          - Two longitudinal side rails (box-section steel beams)
          - Transverse cross members connecting the rails
          - Support legs (4 pairs)
          - Nose plates at infeed/outfeed (tapered ends for belt wrap)
          - Top deck plates where the lower electrode trays sit
          - Attenuation duct walls at infeed and outfeed

        All geometry Y-up: belt carrying surface at y ≈ 0, frame extends
        downward to y = −frame_height, legs further to y = −(frame_height
        + leg_height).
        """
        if self._bed_structure_vertices is not None:
            return (
                self._bed_structure_vertices,
                self._bed_structure_triangles,
                {"type": "conveyor_bed_structure"},
            )
        p = self.params
        L = p.frame_length_m
        W = p.frame_width_m
        H = p.frame_height_m        # frame depth below belt surface
        rs = p.rail_section_m        # rail profile size
        cmw = p.cross_member_w_m
        cmh = p.cross_member_h_m
        ls = p.leg_section_m
        lh = p.leg_height_m
        nose = p.nose_length_m
        nd = p.nose_drop_m

        parts = []

        # ── 1. Longitudinal side rails (run full length at top of frame) ──
        # Left rail (z ≈ 0)
        parts.append(box_mesh(0, -rs, 0, L, rs, rs))
        # Right rail (z ≈ W - rs)
        parts.append(box_mesh(0, -rs, W - rs, L, rs, rs))

        # ── 2. Lower longitudinal rails (bottom of frame) ──
        parts.append(box_mesh(nose, -H, 0, L - 2 * nose, rs, rs))
        parts.append(box_mesh(nose, -H, W - rs, L - 2 * nose, rs, rs))

        # ── 3. Cross members (transverse, connecting side rails) ──
        n_cross = 8
        cross_xs = np.linspace(nose + 0.05, L - nose - 0.05, n_cross)
        for xc in cross_xs:
            # Top cross member
            parts.append(box_mesh(
                float(xc) - cmw / 2, -rs, rs,
                cmw, cmh, W - 2 * rs,
            ))
            # Bottom cross member
            parts.append(box_mesh(
                float(xc) - cmw / 2, -H, rs,
                cmw, cmh, W - 2 * rs,
            ))

        # ── 4. Vertical stiffeners (connect top and bottom rails) ──
        n_stiff = 6
        stiff_xs = np.linspace(nose + 0.15, L - nose - 0.15, n_stiff)
        for xc in stiff_xs:
            # Left side
            parts.append(box_mesh(
                float(xc) - cmw / 2, -H, 0,
                cmw, H - rs, rs,
            ))
            # Right side
            parts.append(box_mesh(
                float(xc) - cmw / 2, -H, W - rs,
                cmw, H - rs, rs,
            ))

        # ── 5. Support legs (4 pairs — 8 total) ──
        leg_xs = [nose + 0.15, L * 0.35, L * 0.65, L - nose - 0.15]
        for xl in leg_xs:
            # Left leg
            parts.append(box_mesh(
                xl - ls / 2, -H - lh, -ls * 0.3,
                ls, lh, ls,
            ))
            # Right leg
            parts.append(box_mesh(
                xl - ls / 2, -H - lh, W - rs + ls * 0.3 - ls,
                ls, lh, ls,
            ))
        # Foot cross-braces (connect leg pairs at bottom for stability)
        for xl in leg_xs:
            parts.append(box_mesh(
                xl - cmw / 2, -H - lh, 0,
                cmw, cmw, W,
            ))

        # ── 6. Nose plates (tapered end sections) ──
        # Infeed nose (x = 0 to nose) — trapezoidal side profile
        # Left side plate
        nose_verts_L = np.array([
            [0, 0, 0],                   # 0 tip top-left
            [0, -nd, 0],                 # 1 tip bottom-left
            [nose, -H, 0],               # 2 inner bottom-left
            [nose, 0, 0],                # 3 inner top-left
            [0, 0, rs],                  # 4 tip top-right
            [0, -nd, rs],               # 5 tip bottom-right
            [nose, -H, rs],              # 6 inner bottom-right
            [nose, 0, rs],               # 7 inner top-right
        ], dtype=np.float32)
        nose_tris = np.array([
            [0, 3, 7], [0, 7, 4],       # top
            [0, 1, 2], [0, 2, 3],       # front (z=0)
            [4, 7, 6], [4, 6, 5],       # back (z=rs)
            [0, 4, 5], [0, 5, 1],       # end face (x=0)
            [3, 2, 6], [3, 6, 7],       # inner face (x=nose)
            [1, 5, 6], [1, 6, 2],       # bottom slant
        ], dtype=np.int32)
        parts.append((nose_verts_L, nose_tris))

        # Infeed nose right side
        nose_verts_R = nose_verts_L.copy()
        nose_verts_R[:, 2] = np.where(
            nose_verts_L[:, 2] < rs / 2, W - rs, W
        )
        parts.append((nose_verts_R, nose_tris.copy()))

        # Outfeed nose (mirror at x = L)
        out_verts_L = np.array([
            [L, 0, 0],
            [L, -nd, 0],
            [L - nose, -H, 0],
            [L - nose, 0, 0],
            [L, 0, rs],
            [L, -nd, rs],
            [L - nose, -H, rs],
            [L - nose, 0, rs],
        ], dtype=np.float32)
        parts.append((out_verts_L, nose_tris.copy()))

        out_verts_R = out_verts_L.copy()
        out_verts_R[:, 2] = np.where(
            out_verts_L[:, 2] < rs / 2, W - rs, W
        )
        parts.append((out_verts_R, nose_tris.copy()))

        # ── 7. Top deck plates (where lower electrode trays sit) ──
        # Thin sheet metal supporting the electrode trays across belt width
        deck_thick = 0.003   # 3 mm sheet
        deck_z0 = p.belt_z0
        deck_z1 = p.belt_z1
        # Deck runs between nose sections
        parts.append(box_mesh(
            nose, -deck_thick, deck_z0,
            L - 2 * nose, deck_thick, deck_z1 - deck_z0,
        ))

        # ── 8. Attenuation duct side walls ──
        dL = p.duct_length_m
        dH = p.duct_height_m
        dW = p.duct_wall_thickness_m
        # Infeed duct (x = nose to nose + dL)
        parts.append(box_mesh(nose, 0, 0, dL, dH, dW))          # left wall
        parts.append(box_mesh(nose, 0, W - dW, dL, dH, dW))     # right wall
        parts.append(box_mesh(nose, dH, 0, dL, dW, W))          # top
        # Outfeed duct (x = L - nose - dL to L - nose)
        x_out = L - nose - dL
        parts.append(box_mesh(x_out, 0, 0, dL, dH, dW))
        parts.append(box_mesh(x_out, 0, W - dW, dL, dH, dW))
        parts.append(box_mesh(x_out, dH, 0, dL, dW, W))

        self._bed_structure_vertices, self._bed_structure_triangles = \
            concat_meshes(parts)
        return (
            self._bed_structure_vertices,
            self._bed_structure_triangles,
            {
                "type": "conveyor_bed_structure",
                "frame_length_m": L,
                "frame_width_m": W,
                "frame_height_m": H,
                "leg_height_m": lh,
                "nose_length_m": nose,
                "duct_length_m": dL,
            },
        )

    def generate_wheels_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the GP-15 roller system (from manual belt-path diagram).

        Roller layout (Illustration 11: Belt Path, Manual p.36):
          1.  Head (drive) roller — outfeed end, motor-driven, largest
          2.  Tail (nose) roller  — infeed end, free-spinning
          3.  Gravity tension roller — pneumatically loaded, below return run,
              creates a sag in the belt to maintain constant tension
          4.  Belt tracking rollers — 2-roller assembly on the return path,
              pneumatic actuators correct lateral drift
          5.  Return support rollers — small guide rollers under the return run
          6.  Carrying idlers — sparse, between oven duct openings (belt
              mostly rides on the lower electrode trays inside the oven)

        All rollers span the belt width Z ∈ [belt_z0, belt_z1] (800 mm)
        and are centred on the belt, except the head/tail which may be
        slightly wider.
        """
        if self._wheels_vertices is not None:
            return self._wheels_vertices, self._wheels_triangles, {"type": "conveyor_wheels"}
        p = self.params
        L = p.frame_length_m
        W = p.frame_width_m
        H = p.frame_height_m
        nose = p.nose_length_m
        res = p.roller_resolution

        z0 = p.belt_z0
        bw = p.belt_width_m   # roller length = belt width

        # Head and tail roller widths slightly wider for tracking
        head_bw = bw + 0.04   # 4 cm wider total for edge guidance
        head_z0 = z0 - 0.02

        # Y positions — belt carrying surface at y = belt_stack_thickness
        # Rollers sit so their top touches the belt underside (y ≈ 0)
        head_cy = p.belt_stack_thickness_m - p.head_roller_radius_m
        tail_cy = p.belt_stack_thickness_m - p.tail_roller_radius_m

        # X positions — head at outfeed end, tail at infeed end
        x_head = L - nose
        x_tail = nose

        # Return run level — midway down the frame
        y_return = -H * 0.55

        # Tension roller drops below the return run
        tension_cy = y_return - p.tension_drop_m
        tension_x = L * 0.40   # slightly toward infeed (per manual illustration)

        # Tracking roller pair — just after the tension roller toward outfeed
        tracker_x1 = L * 0.55
        tracker_x2 = L * 0.62
        tracker_cy = y_return + p.tracker_roller_radius_m

        parts = []

        # ── 1. Head (drive) roller ──
        head_v, head_t = cylinder_mesh(
            (x_head, head_cy, head_z0), p.head_roller_radius_m, head_bw,
            resolution=res, axis="z",
        )
        parts.append((head_v, head_t))

        # ── 2. Tail (nose) roller ──
        tail_v, tail_t = cylinder_mesh(
            (x_tail, tail_cy, head_z0), p.tail_roller_radius_m, head_bw,
            resolution=res, axis="z",
        )
        parts.append((tail_v, tail_t))

        # ── 3. Gravity tension roller ──
        tension_v, tension_t = cylinder_mesh(
            (tension_x, tension_cy, z0), p.tension_roller_radius_m, bw,
            resolution=res, axis="z",
        )
        parts.append((tension_v, tension_t))

        # Tension roller pneumatic arm shafts (visual only)
        arm_r = 0.012
        # Left arm
        parts.append(cylinder_mesh(
            (tension_x, tension_cy, z0 - 0.03), arm_r, 0.03,
            resolution=8, axis="z",
        ))
        # Right arm
        parts.append(cylinder_mesh(
            (tension_x, tension_cy, z0 + bw), arm_r, 0.03,
            resolution=8, axis="z",
        ))
        # Vertical arm links to frame
        parts.append(box_mesh(
            tension_x - arm_r, tension_cy, z0 - 0.03,
            arm_r * 2, -tension_cy - H + 0.05, arm_r * 2,
        ))
        parts.append(box_mesh(
            tension_x - arm_r, tension_cy, z0 + bw + 0.01,
            arm_r * 2, -tension_cy - H + 0.05, arm_r * 2,
        ))

        # ── 4. Belt tracking rollers (2-roller assembly) ──
        for tx in [tracker_x1, tracker_x2]:
            tv, tt = cylinder_mesh(
                (tx, tracker_cy, z0), p.tracker_roller_radius_m, bw,
                resolution=res, axis="z",
            )
            parts.append((tv, tt))

        # ── 5. Return support rollers ──
        n_ret = max(1, p.return_roller_count)
        # Distribute between tail and head on the return path, avoiding
        # the tension sag zone and tracking zone.
        ret_x_start = x_tail + 0.15
        ret_x_end = tension_x - 0.30
        ret_x2_start = tracker_x2 + 0.15
        ret_x2_end = x_head - 0.15

        # Split return rollers: some before tension, some after tracking
        n_before = max(1, n_ret // 2)
        n_after = max(1, n_ret - n_before)

        ret_cy = y_return + p.return_roller_radius_m

        if n_before > 0:
            for xr in np.linspace(ret_x_start, ret_x_end, n_before):
                rv, rt = cylinder_mesh(
                    (float(xr), ret_cy, z0), p.return_roller_radius_m, bw,
                    resolution=res, axis="z",
                )
                parts.append((rv, rt))

        if n_after > 0:
            for xr in np.linspace(ret_x2_start, ret_x2_end, n_after):
                rv, rt = cylinder_mesh(
                    (float(xr), ret_cy, z0), p.return_roller_radius_m, bw,
                    resolution=res, axis="z",
                )
                parts.append((rv, rt))

        # ── 6. Carrying idlers (sparse, within oven section) ──
        n_carry = max(0, min(p.carrying_idler_count, 6))
        if n_carry > 0:
            carry_cy = p.belt_stack_thickness_m - p.carrying_idler_radius_m
            carry_start = x_tail + p.duct_length_m + 0.10
            carry_end = x_head - p.duct_length_m - 0.10
            for xc in np.linspace(carry_start, carry_end, n_carry):
                cv, ct = cylinder_mesh(
                    (float(xc), carry_cy, z0),
                    p.carrying_idler_radius_m, bw,
                    resolution=res, axis="z",
                )
                parts.append((cv, ct))

        self._wheels_vertices, self._wheels_triangles = concat_meshes(parts)

        # Store roller positions for the belt path generator
        self._roller_layout = {
            "head": (x_head, head_cy, p.head_roller_radius_m, "top"),
            "tail": (x_tail, tail_cy, p.tail_roller_radius_m, "top"),
            "tension": (tension_x, tension_cy, p.tension_roller_radius_m, "ret"),
            "tracker_1": (tracker_x1, tracker_cy, p.tracker_roller_radius_m, "ret"),
            "tracker_2": (tracker_x2, tracker_cy, p.tracker_roller_radius_m, "ret"),
            "return_rollers_before": [
                (float(xr), ret_cy, p.return_roller_radius_m, "ret")
                for xr in (np.linspace(ret_x_start, ret_x_end, n_before)
                           if n_before > 0 else [])
            ],
            "return_rollers_after": [
                (float(xr), ret_cy, p.return_roller_radius_m, "ret")
                for xr in (np.linspace(ret_x2_start, ret_x2_end, n_after)
                           if n_after > 0 else [])
            ],
            "carrying_idlers": [
                (float(xc), carry_cy, p.carrying_idler_radius_m, "top")
                for xc in (np.linspace(carry_start, carry_end, n_carry)
                           if n_carry > 0 else [])
            ] if n_carry > 0 else [],
        }

        return (
            self._wheels_vertices,
            self._wheels_triangles,
            {
                "type": "conveyor_wheels",
                "head_x": x_head,
                "tail_x": x_tail,
                "tension_x": tension_x,
                "roller_count": len(parts),
            },
        )

    # ------------------------------------------------------------------
    # Belt tangent / arc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _external_tangent_angles(
        cx1: float, cy1: float, r1: float,
        cx2: float, cy2: float, r2: float,
    ) -> Tuple[float, float]:
        """Departure / arrival angles for the upper external tangent.

        Both circles wrapped CW (belt on the same side of both).
        Returns ``(angle_on_c1, angle_on_c2)`` where the tangent point on
        circle *i* is ``(cx + r*cos(a), cy + r*sin(a))``.
        """
        import math
        dx, dy = cx2 - cx1, cy2 - cy1
        d = math.sqrt(dx * dx + dy * dy)
        base = math.atan2(dy, dx)
        if d < 1e-9:
            return base + math.pi / 2, base + math.pi / 2
        ratio = max(-1.0, min(1.0, (r1 - r2) / d))
        alpha = math.asin(ratio)
        tang = base + math.pi / 2 - alpha
        return tang, tang

    @staticmethod
    def _lower_external_tangent_angles(
        cx1: float, cy1: float, r1: float,
        cx2: float, cy2: float, r2: float,
    ) -> Tuple[float, float]:
        """Lower external tangent (belt on the underside of both rollers)."""
        import math
        dx, dy = cx2 - cx1, cy2 - cy1
        d = math.sqrt(dx * dx + dy * dy)
        base = math.atan2(dy, dx)
        if d < 1e-9:
            return base - math.pi / 2, base - math.pi / 2
        ratio = max(-1.0, min(1.0, (r1 - r2) / d))
        alpha = math.asin(ratio)
        tang = base - math.pi / 2 + alpha
        return tang, tang

    @staticmethod
    def _arc_points_cw(
        cx: float, cy: float, r: float,
        angle_start: float, angle_end: float,
        n_seg: int = 12,
    ) -> list:
        """Sample *n_seg + 1* points on a **clockwise** arc."""
        import math
        sweep = angle_end - angle_start
        while sweep > 0:
            sweep -= 2 * math.pi
        while sweep < -2 * math.pi:
            sweep += 2 * math.pi
        if abs(sweep) < 1e-6:
            sweep = -2 * math.pi
        pts = []
        for i in range(n_seg + 1):
            a = angle_start + sweep * (i / n_seg)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        return pts

    # ------------------------------------------------------------------

    def generate_belt_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate belt as one realistic continuous loop.

        The belt path is computed from roller centres and radii so the
        belt follows tangent lines between consecutive rollers and wraps
        around each roller with a proper arc:

            1.  Tail pulley (first top) → top idlers → Head pulley
                — top carrying run, belt rides *on top* of each roller.
            2.  CW arc around head pulley (top → bottom).
            3.  Return idlers (head-end → tail-end, right to left)
                — return run, belt hangs *below* each roller.
            4.  CW arc around tail pulley (bottom → top).

        Between every consecutive pair the belt follows the external-
        tangent line whose angle is determined by the two roller centres
        and radii.  On each roller the belt traces a CW arc from the
        arrival tangent point to the departure tangent point.

        The 2-D centre-line (X–Y) is extruded across Z ∈ [0, W] and
        given thickness ``belt_thickness_m``.
        """
        if self._belt_vertices is not None:
            return (self._belt_vertices, self._belt_triangles,
                    {"type": "belt", "material": "PTFE"})

        import math

        p = self.params
        half_t = p.belt_thickness_m / 2.0
        W = p.belt_width_m
        z0_belt = p.belt_z0

        # Ensure roller layout is available (calls generate_wheels_mesh
        # if not already done, which stores _roller_layout).
        if not hasattr(self, '_roller_layout') or self._roller_layout is None:
            self.generate_wheels_mesh()
        rl = self._roller_layout

        # ---- Collect rollers in CW loop order ----
        # Top run:   tail → carrying idlers → head  (left → right)
        # Return:    head → return_after → tracker → tension → return_before → tail
        top_rollers = [rl["tail"]]
        top_rollers.extend(rl.get("carrying_idlers", []))
        top_rollers.append(rl["head"])

        # Return path runs right → left (head end back to tail end)
        return_rollers: list = []
        return_rollers.extend(rl.get("return_rollers_after", []))
        return_rollers.append(rl["tracker_2"])
        return_rollers.append(rl["tracker_1"])
        return_rollers.append(rl["tension"])
        return_rollers.extend(rl.get("return_rollers_before", []))

        loop = top_rollers + return_rollers
        N = len(loop)
        n_top = len(top_rollers)

        # ---- Compute tangent departure/arrival angles for every edge ----
        dep_angle = [0.0] * N
        arr_angle = [0.0] * N

        for i in range(N):
            j = (i + 1) % N
            cx1, cy1, r1, k1 = loop[i]
            cx2, cy2, r2, k2 = loop[j]

            if k1 == 'top' and k2 == 'top':
                # Upper external tangent (belt on top of both)
                a1, a2 = self._external_tangent_angles(
                    cx1, cy1, r1, cx2, cy2, r2)
            elif k1 == 'ret' and k2 == 'ret':
                # Lower external tangent (belt under both)
                a1, a2 = self._lower_external_tangent_angles(
                    cx1, cy1, r1, cx2, cy2, r2)
            elif k1 == 'top' and k2 == 'ret':
                # Head transition: belt goes from top-side of head
                # pulley to underside of first return idler.
                # Compute departure on c1 and arrival on c2 directly.
                dx, dy = cx2 - cx1, cy2 - cy1
                d = math.sqrt(dx * dx + dy * dy)
                base = math.atan2(dy, dx)
                if d > 1e-9:
                    ratio = max(-1.0, min(1.0, (r1 + r2) / d))
                    gamma = math.asin(ratio)
                    a1 = base + math.pi / 2 - gamma
                    a2 = a1 + math.pi
                else:
                    a1 = math.pi / 2
                    a2 = -math.pi / 2
            else:
                # k1 == 'ret' and k2 == 'top' — tail transition:
                # belt goes from underside of last return idler to
                # top-side of tail pulley.
                dx, dy = cx2 - cx1, cy2 - cy1
                d = math.sqrt(dx * dx + dy * dy)
                base = math.atan2(dy, dx)
                if d > 1e-9:
                    ratio = max(-1.0, min(1.0, (r1 + r2) / d))
                    gamma = math.asin(ratio)
                    a1 = base - math.pi / 2 + gamma
                    a2 = a1 + math.pi
                else:
                    a1 = -math.pi / 2
                    a2 = math.pi / 2

            dep_angle[i] = a1
            arr_angle[j] = a2

        # Special-case: no return idlers → direct bottom path
        if len(return_rollers) == 0:
            i_head = n_top - 1
            cx1, cy1, r1, _ = loop[i_head]
            cx2, cy2, r2, _ = loop[0]
            a1, a2 = self._lower_external_tangent_angles(
                cx1, cy1, r1, cx2, cy2, r2)
            dep_angle[i_head] = a1
            arr_angle[0] = a2

        # ---- Build 2-D centre-line ----
        arc_res = 16
        path_2d: list = []

        for i in range(N):
            cx, cy, ri, _ = loop[i]
            arc_pts = self._arc_points_cw(cx, cy, ri,
                                          arr_angle[i], dep_angle[i],
                                          n_seg=arc_res)
            path_2d.extend(arc_pts)

            # Straight tangent to next roller (arrival point)
            j = (i + 1) % N
            cx2, cy2, r2, _ = loop[j]
            arr_pt = (cx2 + r2 * math.cos(arr_angle[j]),
                      cy2 + r2 * math.sin(arr_angle[j]))
            path_2d.append(arr_pt)

        path_2d.append(path_2d[0])

        # Remove near-duplicate consecutive points
        cleaned: list = [path_2d[0]]
        for pt in path_2d[1:]:
            dx = pt[0] - cleaned[-1][0]
            dy = pt[1] - cleaned[-1][1]
            if math.sqrt(dx * dx + dy * dy) > 1e-6:
                cleaned.append(pt)
        path = np.array(cleaned, dtype=np.float64)

        # ---- Per-vertex outward normal ----
        M = len(path)
        normals = np.zeros((M, 2), dtype=np.float64)
        for k in range(M):
            k_prev = (k - 1) % M
            k_next = (k + 1) % M
            tx = path[k_next, 0] - path[k_prev, 0]
            ty = path[k_next, 1] - path[k_prev, 1]
            length = math.sqrt(tx * tx + ty * ty)
            if length < 1e-12:
                normals[k] = (0.0, 1.0)
            else:
                normals[k, 0] = ty / length
                normals[k, 1] = -tx / length

        # Ensure normals point outward (CW loop has negative signed area)
        signed_area = 0.0
        for k in range(M):
            k_next = (k + 1) % M
            signed_area += (path[k, 0] * path[k_next, 1]
                            - path[k_next, 0] * path[k, 1])
        signed_area *= 0.5
        if signed_area > 0:
            normals = -normals

        # ---- Extrude to 3-D mesh ----
        z_lo = z0_belt                    # belt left edge
        z_hi = z0_belt + W               # belt right edge
        verts = np.zeros((M * 4, 3), dtype=np.float32)
        for k in range(M):
            ox = path[k, 0] + normals[k, 0] * half_t
            oy = path[k, 1] + normals[k, 1] * half_t
            ix = path[k, 0] - normals[k, 0] * half_t
            iy = path[k, 1] - normals[k, 1] * half_t
            b = k * 4
            verts[b + 0] = [ox, oy, z_lo]
            verts[b + 1] = [ox, oy, z_hi]
            verts[b + 2] = [ix, iy, z_lo]
            verts[b + 3] = [ix, iy, z_hi]

        tris_list = []
        for k in range(M - 1):
            a = k * 4
            b = (k + 1) * 4
            # Outer face
            tris_list.append([a + 0, b + 0, b + 1])
            tris_list.append([a + 0, b + 1, a + 1])
            # Inner face (reversed winding)
            tris_list.append([a + 2, a + 3, b + 3])
            tris_list.append([a + 2, b + 3, b + 2])
            # z = 0 edge
            tris_list.append([a + 0, a + 2, b + 2])
            tris_list.append([a + 0, b + 2, b + 0])
            # z = W edge
            tris_list.append([a + 1, b + 1, b + 3])
            tris_list.append([a + 1, b + 3, a + 3])

        tris = np.array(tris_list, dtype=np.int32)

        self._belt_vertices = verts
        self._belt_triangles = tris
        return (self._belt_vertices, self._belt_triangles,
                {"type": "belt", "material": "PTFE"})

    def generate_bed_mesh(
        self, bed_depth_m: float,
        slant_angle_from_vertical_deg: float = 45.0,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate material bed mesh with vertical face and 45° slant.

        The bed has:
        - A vertical face at the infeed (60cm)
        - A 45° slant from vertical that drops to belt level
        - The slant horizontal extent = bed_depth (at 45°, drop equals horizontal distance)

        Side view profile:
            INFEED
            ┌──┐
            │  │ 60cm vertical face
            │  │
            │  └╲
            │    ╲  45° from vertical
            │     ╲
            └──────╲─────────────────────────────
            ←─60cm─→ slant section
            ←────────── 577 cm total ───────────→

        Args:
            bed_depth_m: Bed depth at infeed (vertical face height) [m].
            slant_angle_from_vertical_deg: Slant angle from vertical (default 45°).

        Returns:
            (vertices, triangles, metadata)
        """
        import math
        p = self.params
        y_base = p.belt_stack_thickness_m

        # Bed sits on the belt, centred within the frame
        slant_angle_rad = math.radians(slant_angle_from_vertical_deg)
        slant_horizontal = bed_depth_m * math.tan(slant_angle_rad)

        z0 = p.belt_z0
        z1 = p.belt_z1

        # Create triangular prism mesh for the bed
        # The bed is a triangular cross-section extruded along Z (width)
        verts = np.array([
            # Left side (z0) - triangular profile
            [0.0, y_base, z0],                           # 0: infeed-bottom-left
            [0.0, y_base + bed_depth_m, z0],             # 1: infeed-top-left
            [slant_horizontal, y_base, z0],              # 2: slant-end-left

            # Right side (z1) - triangular profile
            [0.0, y_base, z1],                           # 3: infeed-bottom-right
            [0.0, y_base + bed_depth_m, z1],             # 4: infeed-top-right
            [slant_horizontal, y_base, z1],              # 5: slant-end-right
        ], dtype=np.float32)

        # Triangles for the triangular prism (5 faces)
        tris = np.array([
            # Left triangular face (z0)
            [0, 2, 1],
            # Right triangular face (z1)
            [3, 4, 5],
            # Bottom face (belt level)
            [0, 3, 5], [0, 5, 2],
            # Front vertical face (infeed)
            [0, 1, 4], [0, 4, 3],
            # Sloped top face (45° from vertical)
            [1, 2, 5], [1, 5, 4],
        ], dtype=np.int32)

        return verts, tris, {
            "type": "material_bed",
            "profile": "triangular_wedge",
            "infeed_depth_m": bed_depth_m,
            "slant_horizontal_m": slant_horizontal,
            "slant_angle_deg": slant_angle_from_vertical_deg,
        }

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate combined belt mesh (for compatibility).

        Returns:
            (vertices, triangles, metadata)
        """
        return self.generate_belt_mesh()

    @property
    def ports(self) -> Dict[str, "ConnectionPort"]:
        """Connection ports for material flow.

        Ports:
        - material_inlet: Start of flat top (feeder, x = nose_length)
        - material_outlet: End of flat top (collector, x = L - nose_length)
        """
        from airclassifier.geometry.connection_ports import ConnectionPort, PortType

        p = self.params
        y_top = p.belt_stack_thickness_m + 0.05
        S = p.nose_length_m
        L = p.frame_length_m
        z_mid = p.belt_center_z

        return {
            'material_inlet': ConnectionPort(
                position=(S, y_top, z_mid),
                direction=(-1.0, 0.0, 0.0),
                width=p.belt_width_m,
                height=0.1,
                port_type=PortType.GRAVITY,
                name="belt_material_inlet",
            ),
            'material_outlet': ConnectionPort(
                position=(L - S, y_top, z_mid),
                direction=(1.0, 0.0, 0.0),
                width=p.belt_width_m,
                height=0.1,
                port_type=PortType.GRAVITY,
                name="belt_material_outlet",
            ),
        }


# Backward compatibility aliases
ConveyorParams = ConveyorBeltParams
ConveyorGeometry = ConveyorBeltGeometry