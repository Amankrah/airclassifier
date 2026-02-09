"""
Infeed Hopper & Feed Tunnel Geometry
=====================================

GP-15 infeed system: gravity hopper with compound back wall and
feed tunnel that connects to the oven chamber infeed wall.

Based on actual GP-15 machine photographs and measurements:

    Top-down view of hopper:

        ┌────────────────────┐
        │                    │  69.5 cm (Z, across belt)
        │    HOPPER BIN      │
        │    (open top)      │
        └────────────────────┘
              60 cm (X, along belt)

    Side profile (X-Y) — complete infeed assembly:

        back wall               front wall
        ┌──┐                      ┌──┐
        │  │ 6.9 cm vertical      │  │ 25.5 cm
        │  ╲                      │  │
        │   ╲ 35.3 cm @ 45°      │  │
        │    ╲                    │  │
        │     ╲___________________│__│ ← belt level
        │        sizing gate ↑   │
        │                        │
        ←── 60 cm hopper ──→←gap→
                                 ←── 24.8 cm tunnel ──→←──── OVEN WALL
                                                             x = 1.65

    Photo measurements:
        Hopper width (Z):     69.5 cm  (photo 1 bottom)
        Hopper depth (X):     60.0 cm  (photo 1 bottom)
        Front opening:        25.5 cm  (photo 1 top)
        Back vertical:         6.9 cm  (photo 1 top, ~69 mm)
        Back slant length:    35.3 cm  (photo 1 top)
        Feed tunnel length:   24.8 cm  (photo 2)
        Tunnel gap offset:    10.7 cm  (photo 2, hopper→tunnel)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import box_mesh, concat_meshes, hollow_box_mesh

if TYPE_CHECKING:
    from airclassifier.geometry.connection_ports import ConnectionPort
    from ..components.oven_chamber import OvenChamberParams


# ─────────────────────────────────────────────────────────────────────
#  Parameters
# ─────────────────────────────────────────────────────────────────────

@dataclass
class InfeedHopperParams:
    """Infeed hopper parameters from actual GP-15 measurements.

    The hopper has a compound back wall:
    - Vertical section at top (6.9 cm)
    - Slanted section at 45 deg from vertical (35.3 cm long)

    The feed tunnel is a rectangular duct that connects the hopper
    discharge area to the oven chamber infeed wall, providing RF
    attenuation.

    Coordinate system:
    - X: Conveyor direction (material flows in +X toward oven)
    - Y: Vertical (0 = belt surface)
    - Z: Across belt width (centred on belt)
    """

    # ── Hopper bin dimensions (from photographs) ──────────────────
    hopper_width_m: float = 0.695        # [m] across belt (Z) — 69.5 cm
    hopper_depth_m: float = 0.600        # [m] along belt (X) — 60 cm
    front_wall_height_m: float = 0.255   # [m] front/discharge wall — 25.5 cm

    # ── Back wall compound geometry ───────────────────────────────
    back_vertical_height_m: float = 0.069  # [m] vertical at top — 6.9 cm
    back_slant_length_m: float = 0.353     # [m] slant surface — 35.3 cm
    back_slant_angle_deg: float = 45.0     # [deg] from vertical

    # ── Feed tunnel (attenuation duct to oven) ────────────────────
    tunnel_length_m: float = 0.248       # [m] — 24.8 cm (photo 2)
    tunnel_height_m: float = 0.258       # [m] internal height — ~25.8 cm
    tunnel_wall_thickness_m: float = 0.003  # [m] sheet metal

    # ── Gap between hopper front and tunnel entrance ──────────────
    hopper_tunnel_gap_m: float = 0.107   # [m] — 10.7 cm (photo 2)

    # ── Construction ──────────────────────────────────────────────
    plate_thickness_m: float = 0.003     # [m] 3 mm stainless steel
    flange_width_m: float = 0.025        # [m] top-rim stiffener flange
    flange_height_m: float = 0.020       # [m] flange fold-over height

    # ── Sizing gate (adjustable plate at discharge) ───────────────
    sizing_gate_height_m: float = 0.050  # [m] gate opening (bed depth)
    sizing_gate_thickness_m: float = 0.005  # [m] gate plate

    # ── Positioning on conveyor (set by from_oven) ────────────────
    # Oven infeed wall X coordinate
    oven_infeed_x_m: float = 1.65        # [m] default from OvenChamberParams

    # Belt Z geometry (from conveyor)
    belt_z0_m: float = 0.15              # [m] belt left edge
    belt_width_m: float = 0.80           # [m] belt width
    frame_width_m: float = 1.10          # [m] conveyor frame width

    @classmethod
    def from_oven(
        cls,
        oven_params: "OvenChamberParams",
        sizing_gate_height_m: float = 0.050,
    ) -> "InfeedHopperParams":
        """Create hopper params aligned to oven chamber geometry.

        Positions the feed tunnel flush against the oven infeed wall
        and the hopper at the tunnel's outer end.

        Args:
            oven_params: Oven chamber parameters.
            sizing_gate_height_m: Bed depth control gate opening.
        """
        return cls(
            oven_infeed_x_m=oven_params.oven_x_start_m,
            belt_z0_m=oven_params.conveyor_belt_z0_m,
            belt_width_m=oven_params.rf_zone_width_m,
            frame_width_m=oven_params.oven_width_m,
            sizing_gate_height_m=sizing_gate_height_m,
        )

    # ── Derived geometry ──────────────────────────────────────────

    @property
    def slant_angle_rad(self) -> float:
        return math.radians(self.back_slant_angle_deg)

    @property
    def slant_vertical_m(self) -> float:
        """Vertical drop of the slant section."""
        return self.back_slant_length_m * math.cos(self.slant_angle_rad)

    @property
    def slant_horizontal_m(self) -> float:
        """Horizontal (X) extent of the slant section."""
        return self.back_slant_length_m * math.sin(self.slant_angle_rad)

    @property
    def total_back_height_m(self) -> float:
        """Full back wall height = vertical + slant vertical."""
        return self.back_vertical_height_m + self.slant_vertical_m

    @property
    def belt_z_center_m(self) -> float:
        """Belt centre in Z."""
        return self.belt_z0_m + self.belt_width_m / 2.0

    # ── Key X positions (world coordinates) ───────────────────────

    @property
    def tunnel_inner_x(self) -> float:
        """Tunnel end touching oven wall (highest X)."""
        return self.oven_infeed_x_m

    @property
    def tunnel_outer_x(self) -> float:
        """Tunnel end facing hopper (lower X)."""
        return self.oven_infeed_x_m - self.tunnel_length_m

    @property
    def hopper_front_x(self) -> float:
        """Hopper front face / discharge X."""
        return self.tunnel_outer_x - self.hopper_tunnel_gap_m

    @property
    def hopper_back_x(self) -> float:
        """Hopper back edge X (top of back wall)."""
        return self.hopper_front_x - self.hopper_depth_m


# ─────────────────────────────────────────────────────────────────────
#  Geometry
# ─────────────────────────────────────────────────────────────────────

class InfeedHopperGeometry:
    """Realistic GP-15 infeed hopper and feed tunnel.

    The assembly consists of:

    1. **Hopper bin** — stainless steel funnel with compound back wall
       (vertical + 45-deg slant), short front wall, trapezoidal sides,
       flanged top rim, and bottom support lips.

    2. **Sizing gate** — adjustable plate at the hopper discharge that
       controls bed depth on the belt.

    3. **Feed tunnel** — rectangular attenuation duct connecting the
       hopper zone to the oven chamber infeed wall.  The belt passes
       through it carrying product into the oven.

    4. **Mounting brackets** — small supports on the conveyor frame.

    All positioned in conveyor world coordinates so the tunnel inner
    face is flush with the oven infeed wall.
    """

    def __init__(self, params: Optional[InfeedHopperParams] = None):
        self.params = params or InfeedHopperParams()
        self._hopper_verts: Optional[np.ndarray] = None
        self._hopper_tris: Optional[np.ndarray] = None
        self._tunnel_verts: Optional[np.ndarray] = None
        self._tunnel_tris: Optional[np.ndarray] = None
        self._combined_verts: Optional[np.ndarray] = None
        self._combined_tris: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────────
    #  Hopper bin
    # ─────────────────────────────────────────────────────────────

    def generate_hopper_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the hopper bin mesh.

        Returns:
            (vertices, triangles, metadata)
        """
        if self._hopper_verts is not None:
            return self._hopper_verts, self._hopper_tris, self._hopper_meta()

        p = self.params
        t = p.plate_thickness_m

        # ── Z extents (across belt) ────────────────────────────────
        z_center = p.belt_z_center_m
        half_w = p.hopper_width_m / 2.0
        z0 = z_center - half_w          # left side
        z1 = z_center + half_w          # right side

        # ── X extents (along belt) ─────────────────────────────────
        x_front = p.hopper_front_x      # discharge face (toward oven)
        x_back_top = p.hopper_back_x    # back edge at top

        # ── Back wall profile (X-Y) ───────────────────────────────
        slant_v = p.slant_vertical_m    # vertical drop of slant
        slant_h = p.slant_horizontal_m  # horizontal extent of slant

        y_belt = 0.0                    # belt surface
        y_top = p.total_back_height_m   # top of back wall
        y_slant_start = y_top - p.back_vertical_height_m
        y_slant_end = y_slant_start - slant_v
        x_slant_end = x_back_top + slant_h  # slant bottom X

        y_front_top = p.front_wall_height_m  # front wall top

        parts: List[Tuple[np.ndarray, np.ndarray]] = []

        # ── 1. Front wall ──────────────────────────────────────────
        # Vertical plate at x_front, from belt to front_wall_height
        parts.append(box_mesh(
            x_front, y_belt, z0,
            t, y_front_top, p.hopper_width_m,
        ))

        # ── 2. Back wall — vertical section ────────────────────────
        # From y_slant_start to y_top at x_back_top
        parts.append(box_mesh(
            x_back_top - t, y_slant_start, z0,
            t, p.back_vertical_height_m, p.hopper_width_m,
        ))

        # ── 3. Back wall — slanted section ─────────────────────────
        # From (x_back_top, y_slant_start) down to (x_slant_end, y_slant_end)
        parts.append(self._slanted_plate(
            top_x=x_back_top, top_y=y_slant_start,
            bot_x=x_slant_end, bot_y=max(y_slant_end, y_belt),
            z0=z0, z1=z1, thickness=t,
        ))

        # ── 4. Left side wall ──────────────────────────────────────
        parts.append(self._side_wall(
            z_pos=z0 - t,
            x_front=x_front, y_front_top=y_front_top,
            x_back_top=x_back_top, y_top=y_top,
            y_slant_start=y_slant_start,
            x_slant_end=x_slant_end,
            y_slant_end=max(y_slant_end, y_belt),
            y_belt=y_belt, thickness=t,
        ))

        # ── 5. Right side wall ─────────────────────────────────────
        parts.append(self._side_wall(
            z_pos=z1,
            x_front=x_front, y_front_top=y_front_top,
            x_back_top=x_back_top, y_top=y_top,
            y_slant_start=y_slant_start,
            x_slant_end=x_slant_end,
            y_slant_end=max(y_slant_end, y_belt),
            y_belt=y_belt, thickness=t,
        ))

        # ── 6. Top rim flanges (folded stiffener around top edge) ──
        fl_w = p.flange_width_m
        fl_h = p.flange_height_m
        # Front rim (horizontal bar across front wall top, in Z)
        parts.append(box_mesh(
            x_front - fl_w, y_front_top, z0 - t,
            fl_w + t, fl_h, p.hopper_width_m + 2 * t,
        ))
        # Back rim (horizontal bar across back wall top, in Z)
        parts.append(box_mesh(
            x_back_top - t - fl_w, y_top, z0 - t,
            fl_w + t, fl_h, p.hopper_width_m + 2 * t,
        ))
        # Left side rim (runs along X from front to back at z0)
        parts.append(box_mesh(
            x_back_top - fl_w, y_front_top, z0 - t - fl_w,
            x_front - x_back_top + 2 * fl_w, fl_h, fl_w,
        ))
        # Right side rim (runs along X from front to back at z1)
        parts.append(box_mesh(
            x_back_top - fl_w, y_front_top, z1 + t,
            x_front - x_back_top + 2 * fl_w, fl_h, fl_w,
        ))

        # ── 7. Bottom support lips (guide rails at belt level) ─────
        lip_h = 0.015   # 15 mm
        lip_w = 0.010   # 10 mm
        # Left guide rail along belt edge
        parts.append(box_mesh(
            x_slant_end, y_belt - lip_h, z0 - t,
            x_front - x_slant_end, lip_h, lip_w,
        ))
        # Right guide rail
        parts.append(box_mesh(
            x_slant_end, y_belt - lip_h, z1 + t - lip_w,
            x_front - x_slant_end, lip_h, lip_w,
        ))

        # ── 8. Sizing gate (hangs from front wall inside) ──────────
        gate_y_bot = y_belt + p.sizing_gate_height_m
        gate_h = y_front_top - gate_y_bot
        if gate_h > 0.005:
            parts.append(box_mesh(
                x_front + t, gate_y_bot, z0,
                p.sizing_gate_thickness_m, gate_h, p.hopper_width_m,
            ))
            # Gate adjustment bracket (visual, small L-bracket on top)
            bracket_h = 0.020
            parts.append(box_mesh(
                x_front + t, y_front_top - bracket_h, z_center - 0.03,
                p.sizing_gate_thickness_m + 0.008, bracket_h, 0.06,
            ))

        # ── 9. Mounting feet (4 small pads on the frame) ───────────
        foot_s = 0.04    # 40 mm square
        foot_h = 0.015
        foot_y = y_belt - lip_h - foot_h
        for fx in [x_slant_end + 0.02, x_front - 0.06]:
            for fz in [z0 - t, z1 + t - foot_s]:
                parts.append(box_mesh(fx, foot_y, fz, foot_s, foot_h, foot_s))

        self._hopper_verts, self._hopper_tris = concat_meshes(parts)
        return self._hopper_verts, self._hopper_tris, self._hopper_meta()

    def _hopper_meta(self) -> dict:
        p = self.params
        return {
            "type": "infeed_hopper",
            "hopper_front_x": p.hopper_front_x,
            "hopper_back_x": p.hopper_back_x,
            "hopper_width_m": p.hopper_width_m,
            "hopper_depth_m": p.hopper_depth_m,
            "total_height_m": p.total_back_height_m,
            "sizing_gate_m": p.sizing_gate_height_m,
        }

    # ─────────────────────────────────────────────────────────────
    #  Feed tunnel
    # ─────────────────────────────────────────────────────────────

    def generate_tunnel_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the feed tunnel fitted to the oven infeed wall.

        The tunnel is a rectangular duct:
        - Inner end: flush with the oven infeed wall (x = oven_infeed_x)
        - Outer end: faces the hopper discharge zone
        - Belt passes through carrying material into the oven
        - Acts as an RF attenuation duct

        Returns:
            (vertices, triangles, metadata)
        """
        if self._tunnel_verts is not None:
            return self._tunnel_verts, self._tunnel_tris, self._tunnel_meta()

        p = self.params
        tw = p.tunnel_wall_thickness_m

        # Tunnel X extent
        x0 = p.tunnel_outer_x           # outer end (toward hopper)
        tL = p.tunnel_length_m           # length along X

        # Tunnel Y extent: from belt level (y=0) upward
        y0 = 0.0
        tH = p.tunnel_height_m

        # Tunnel Z extent: centred, slightly wider than belt
        z_center = p.belt_z_center_m
        tunnel_z_width = p.hopper_width_m + 0.04  # 2 cm clearance each side
        tz0 = z_center - tunnel_z_width / 2.0
        tW = tunnel_z_width

        parts: List[Tuple[np.ndarray, np.ndarray]] = []

        # ── Tunnel walls (hollow box, open at both X ends) ─────────
        # Bottom wall
        parts.append(box_mesh(x0, y0 - tw, tz0, tL, tw, tW))
        # Top wall
        parts.append(box_mesh(x0, y0 + tH, tz0, tL, tw, tW))
        # Left wall (z = tz0)
        parts.append(box_mesh(x0, y0, tz0, tL, tH, tw))
        # Right wall (z = tz0 + tW - tw)
        parts.append(box_mesh(x0, y0, tz0 + tW - tw, tL, tH, tw))

        # ── Tunnel end flange (outer end, around opening) ──────────
        fl = 0.015  # 15 mm flange
        # Top flange
        parts.append(box_mesh(x0 - fl, y0 + tH, tz0 - fl, fl, tw, tW + 2 * fl))
        # Bottom flange
        parts.append(box_mesh(x0 - fl, y0 - tw, tz0 - fl, fl, tw, tW + 2 * fl))
        # Left flange
        parts.append(box_mesh(x0 - fl, y0, tz0 - fl, fl, tH, tw + fl))
        # Right flange
        parts.append(box_mesh(x0 - fl, y0, tz0 + tW, fl, tH, tw + fl))

        # ── Internal stiffener ribs (2 along length) ──────────────
        rib_t = 0.003
        for rib_x_frac in [0.33, 0.66]:
            rib_x = x0 + tL * rib_x_frac
            # Top rib
            parts.append(box_mesh(
                rib_x - rib_t / 2, y0 + tH - 0.015, tz0,
                rib_t, 0.015, tW,
            ))

        self._tunnel_verts, self._tunnel_tris = concat_meshes(parts)
        return self._tunnel_verts, self._tunnel_tris, self._tunnel_meta()

    def _tunnel_meta(self) -> dict:
        p = self.params
        return {
            "type": "infeed_tunnel",
            "tunnel_outer_x": p.tunnel_outer_x,
            "tunnel_inner_x": p.tunnel_inner_x,
            "tunnel_length_m": p.tunnel_length_m,
            "tunnel_height_m": p.tunnel_height_m,
            "fitted_to_oven_x": p.oven_infeed_x_m,
        }

    # ─────────────────────────────────────────────────────────────
    #  Combined mesh
    # ─────────────────────────────────────────────────────────────

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate complete infeed assembly (hopper + tunnel).

        Returns:
            (vertices, triangles, metadata)
        """
        if self._combined_verts is not None:
            return (self._combined_verts, self._combined_tris,
                    {"type": "infeed_assembly"})

        hv, ht, h_meta = self.generate_hopper_mesh()
        tv, tt, t_meta = self.generate_tunnel_mesh()

        self._combined_verts, self._combined_tris = concat_meshes([
            (hv, ht), (tv, tt),
        ])

        return self._combined_verts, self._combined_tris, {
            "type": "infeed_assembly",
            "hopper": h_meta,
            "tunnel": t_meta,
        }

    # ─────────────────────────────────────────────────────────────
    #  Mesh primitives
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _slanted_plate(
        top_x: float, top_y: float,
        bot_x: float, bot_y: float,
        z0: float, z1: float,
        thickness: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Angled plate from (top_x, top_y) to (bot_x, bot_y), extruded Z.

        The thickness is offset along the outward normal.
        """
        dx = bot_x - top_x
        dy = bot_y - top_y
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            return (np.empty((0, 3), dtype=np.float32),
                    np.empty((0, 3), dtype=np.int32))

        # Outward normal (pointing away from hopper interior)
        nx = -dy / length
        ny = dx / length
        ox = nx * thickness
        oy = ny * thickness

        verts = np.array([
            # Inner face
            [top_x, top_y, z0],          # 0
            [top_x, top_y, z1],          # 1
            [bot_x, bot_y, z0],          # 2
            [bot_x, bot_y, z1],          # 3
            # Outer face
            [top_x + ox, top_y + oy, z0],  # 4
            [top_x + ox, top_y + oy, z1],  # 5
            [bot_x + ox, bot_y + oy, z0],  # 6
            [bot_x + ox, bot_y + oy, z1],  # 7
        ], dtype=np.float32)

        tris = np.array([
            [0, 2, 3], [0, 3, 1],   # inner face
            [4, 5, 7], [4, 7, 6],   # outer face
            [0, 1, 5], [0, 5, 4],   # top edge
            [2, 6, 7], [2, 7, 3],   # bottom edge
            [0, 4, 6], [0, 6, 2],   # left edge (z0)
            [1, 3, 7], [1, 7, 5],   # right edge (z1)
        ], dtype=np.int32)

        return verts, tris

    @staticmethod
    def _side_wall(
        z_pos: float,
        x_front: float, y_front_top: float,
        x_back_top: float, y_top: float,
        y_slant_start: float,
        x_slant_end: float, y_slant_end: float,
        y_belt: float, thickness: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a trapezoidal side wall following the hopper profile.

        Profile points (CCW when viewed from outside):
            0: front-bottom  (x_front, y_belt)
            1: front-top     (x_front, y_front_top)
            2: back-top      (x_back_top, y_top)
            3: slant-start   (x_back_top, y_slant_start)
            4: slant-end     (x_slant_end, y_slant_end)

        Extruded by ``thickness`` in Z.
        """
        profile = [
            (x_front, y_belt),
            (x_front, y_front_top),
            (x_back_top, y_top),
            (x_back_top, y_slant_start),
            (x_slant_end, y_slant_end),
        ]
        n = len(profile)

        verts = []
        for x, y in profile:
            verts.append([x, y, z_pos])
        for x, y in profile:
            verts.append([x, y, z_pos + thickness])
        verts = np.array(verts, dtype=np.float32)

        tris = []
        # Inner face (fan from vertex 0)
        for i in range(1, n - 1):
            tris.append([0, i + 1, i])
        # Outer face (fan from vertex n)
        for i in range(1, n - 1):
            tris.append([n, n + i, n + i + 1])
        # Edge quads connecting inner → outer
        for i in range(n):
            j = (i + 1) % n
            tris.append([i, j, n + j])
            tris.append([i, n + j, n + i])

        return verts, np.array(tris, dtype=np.int32)

    # ─────────────────────────────────────────────────────────────
    #  Connection ports
    # ─────────────────────────────────────────────────────────────

    @property
    def ports(self) -> Dict[str, "ConnectionPort"]:
        """Connection ports for material flow.

        - discharge: Where material exits hopper onto belt
        - inlet: Open top for loading
        - tunnel_outlet: Inner tunnel end (at oven wall)
        """
        from airclassifier.geometry.connection_ports import ConnectionPort, PortType

        p = self.params
        z_c = p.belt_z_center_m

        return {
            'discharge': ConnectionPort(
                position=(p.hopper_front_x, p.sizing_gate_height_m / 2, z_c),
                direction=(1.0, 0.0, 0.0),
                width=p.hopper_width_m,
                height=p.sizing_gate_height_m,
                port_type=PortType.GRAVITY,
                name="hopper_discharge",
            ),
            'inlet': ConnectionPort(
                position=(
                    (p.hopper_front_x + p.hopper_back_x) / 2,
                    p.total_back_height_m,
                    z_c,
                ),
                direction=(0.0, 1.0, 0.0),
                width=p.hopper_depth_m,
                height=p.hopper_width_m,
                port_type=PortType.GRAVITY,
                name="hopper_inlet",
            ),
            'tunnel_outlet': ConnectionPort(
                position=(
                    p.tunnel_inner_x,
                    p.tunnel_height_m / 2,
                    z_c,
                ),
                direction=(1.0, 0.0, 0.0),
                width=p.hopper_width_m + 0.04,
                height=p.tunnel_height_m,
                port_type=PortType.RECTANGULAR,
                name="tunnel_to_oven",
            ),
        }


# Backward compatibility
HopperParams = InfeedHopperParams
HopperGeometry = InfeedHopperGeometry
