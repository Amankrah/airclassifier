"""
RF Generator Cabinet Geometry
==============================

The RF generator is a separate floor-standing cabinet at the **back**
of the GP-15 oven chamber (behind it in +Z), on the **outfeed half**.
The EMU sits adjacent to it on the infeed half — see ``emu.py``.

It contains the triode valve oscillator, HV transformer, tank circuit,
tuning components, and forced-air cooling system.  The distinctive
cooling vent grid is visible on the back (+Z) face.

Physical layout (Manual — Illustration 3):

    Top-down (X–Z):

        ═══════════════════════════════════
        belt direction →→→                  z = 0 (operator side)
        ═══════════════════════════════════
        ┌──────────────────────────────────┐
        │         OVEN CHAMBER             │  z = 0 … oven_width
        └──────────────────────────────────┘
        ┌──────────┐         ┌─────────────┐
        │   EMU    │         │  GENERATOR  │  z = oven_width … +depth
        │          │         │   ▦▦ vents  │
        └──────────┘         └─────────────┘
        ↑ infeed half         ↑ outfeed half

    The generator stands on the floor (same Y as conveyor legs).
    RF power is fed into the oven via copper feed strips through
    the oven back wall.

Coordinate system:
    Y = 0 is deck/belt surface.
    Floor = Y = -(frame_height + leg_height) ≈ -1.30 m.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import box_mesh, cylinder_mesh, concat_meshes

if TYPE_CHECKING:
    from airclassifier.geometry.connection_ports import ConnectionPort
    from .oven_chamber import OvenChamberParams


@dataclass
class GeneratorParams:
    """RF generator cabinet parameters.

    The cabinet sits behind the oven (+Z), on the outfeed half (higher X).
    It stands on the floor.
    """

    # ── Cabinet dimensions ────────────────────────────────────────
    cabinet_length_m: float = 1.00       # [m] along X (covers outfeed half of oven back)
    cabinet_depth_m: float = 0.60        # [m] along Z (how far behind oven)
    cabinet_height_m: float = 2.40       # [m] from floor to top (reaches oven ceiling)
    wall_thickness_m: float = 0.020      # [m] heavy steel for HV safety

    # ── Cooling vent grid (on back face, +Z, facing away from oven)
    vent_rows: int = 8
    vent_cols: int = 4
    vent_slot_width_m: float = 0.12
    vent_slot_height_m: float = 0.015
    vent_spacing_x_m: float = 0.03
    vent_spacing_y_m: float = 0.06
    vent_start_y_offset_m: float = 0.30  # above cabinet base
    vent_start_x_inset_m: float = 0.10

    # ── Base plinth ───────────────────────────────────────────────
    plinth_height_m: float = 0.08
    plinth_inset_m: float = 0.03

    # ── RF output ─────────────────────────────────────────────────
    rf_output_diameter_m: float = 0.05
    rf_output_height_m: float = 1.30     # above floor

    # ── RF feed connection (generator → electrode via oven wall) ──
    # 6 copper feed strips + 4 tuning feed plates + 8 tuning bridges
    feed_strip_count: int = 6
    feed_strip_width_m: float = 0.050    # [m] ~2 inches
    feed_strip_thickness_m: float = 0.002  # [m] copper strip
    tuning_plate_count: int = 4
    tuning_bridge_count: int = 8
    busbar_section_m: float = 0.030      # [m] main RF busbar section

    # ── Gap between generator and oven ──────────────────────────
    gap_from_oven_m: float = 1.30        # [m] 130 cm clearance

    # ── Positioning (set by from_oven) ────────────────────────────
    oven_x_start_m: float = 1.65
    oven_x_end_m: float = 4.15
    oven_width_m: float = 1.10           # [m] oven back wall Z position
    oven_height_m: float = 1.10          # [m] oven internal height
    rf_zone_x_start_m: float = 2.00
    rf_zone_x_end_m: float = 3.80
    rf_zone_width_m: float = 0.80
    belt_z0_m: float = 0.15
    floor_y_m: float = -1.30             # [m] floor level

    @classmethod
    def from_oven(
        cls,
        oven_params: "OvenChamberParams",
        floor_y_m: float = -1.30,
    ) -> "GeneratorParams":
        """Position the generator behind the oven (+Z), outfeed half.

        Height is calculated so the cabinet top reaches the oven ceiling.
        """
        # Generator top = oven ceiling (flush)
        height = oven_params.oven_height_m - floor_y_m

        return cls(
            cabinet_height_m=height,
            oven_x_start_m=oven_params.oven_x_start_m,
            oven_x_end_m=oven_params.oven_x_end_m,
            oven_width_m=oven_params.oven_width_m,
            oven_height_m=oven_params.oven_height_m,
            rf_zone_x_start_m=oven_params.rf_zone_x_start,
            rf_zone_x_end_m=oven_params.rf_zone_x_end,
            rf_zone_width_m=oven_params.rf_zone_width_m,
            belt_z0_m=oven_params.conveyor_belt_z0_m,
            floor_y_m=floor_y_m,
        )

    @property
    def cabinet_x_start(self) -> float:
        """X position — aligned to oven outfeed half."""
        return self.oven_x_end_m - self.cabinet_length_m

    @property
    def cabinet_z_start(self) -> float:
        """Z position — behind the oven back wall, with gap."""
        return self.oven_width_m + self.gap_from_oven_m

    @property
    def base_y(self) -> float:
        """Y position of the cabinet base (floor level)."""
        return self.floor_y_m


class GeneratorGeometry:
    """RF generator cabinet behind the oven (outfeed half).

    Stands on the floor behind the oven's back wall.
    Contains oscillator, HV transformer, tank circuit.
    Distinctive cooling vent grid on the back face.
    """

    def __init__(self, params: Optional[GeneratorParams] = None):
        self.params = params or GeneratorParams()
        self._vertices: Optional[np.ndarray] = None
        self._triangles: Optional[np.ndarray] = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        if self._vertices is not None:
            return self._vertices, self._triangles, self._get_meta()

        p = self.params
        wt = p.wall_thickness_m

        x0 = p.cabinet_x_start
        z0 = p.cabinet_z_start
        y0 = p.base_y             # floor level
        cL = p.cabinet_length_m   # X
        cD = p.cabinet_depth_m    # Z
        cH = p.cabinet_height_m   # Y from floor

        parts = []

        # ── 1. Cabinet box ────────────────────────────────────────
        # Left wall (x = x0)
        parts.append(box_mesh(x0, y0, z0, wt, cH, cD))
        # Right wall (x = x0 + cL)
        parts.append(box_mesh(x0 + cL - wt, y0, z0, wt, cH, cD))
        # Top
        parts.append(box_mesh(x0, y0 + cH, z0, cL, wt, cD))
        # Bottom
        parts.append(box_mesh(x0, y0, z0, cL, wt, cD))
        # Oven-facing wall (z = z0) — RF output goes through here
        parts.append(box_mesh(x0, y0, z0, cL, cH, wt))
        # Back wall (z = z0 + cD) — has cooling vent grid
        parts.append(box_mesh(x0, y0, z0 + cD - wt, cL, cH, wt))

        # ── 2. Internal shelf ─────────────────────────────────────
        shelf_y = y0 + cH * 0.55
        parts.append(box_mesh(x0 + wt, shelf_y, z0 + wt,
                              cL - 2 * wt, wt, cD - 2 * wt))

        # ── 3. Base plinth + levelling feet ───────────────────────
        ph = p.plinth_height_m
        pi = p.plinth_inset_m
        parts.append(box_mesh(x0 - pi, y0 - ph, z0 - pi,
                              cL + 2 * pi, ph, cD + 2 * pi))
        foot_s = 0.04
        foot_h = 0.03
        for fx in [x0 - pi, x0 + cL + pi - foot_s]:
            for fz in [z0 - pi, z0 + cD + pi - foot_s]:
                parts.append(box_mesh(fx, y0 - ph - foot_h, fz,
                                      foot_s, foot_h, foot_s))

        # ── 4. Cooling vent grid (back face, +Z) ─────────────────
        vent_z = z0 + cD - wt - 0.001
        vsx = p.vent_start_x_inset_m
        vsy = p.vent_start_y_offset_m
        for row in range(min(p.vent_rows, 8)):
            for col in range(min(p.vent_cols, 4)):
                vx = x0 + vsx + col * (p.vent_slot_width_m + p.vent_spacing_x_m)
                vy = y0 + vsy + row * (p.vent_slot_height_m + p.vent_spacing_y_m)
                parts.append(box_mesh(
                    vx, vy, vent_z,
                    p.vent_slot_width_m * 0.85, p.vent_slot_height_m,
                    wt + 0.005,
                ))

        # ── 5. RF output stub (oven-facing wall, -Z side) ────────
        rf_x = x0 + cL / 2 - 0.02
        rf_y = y0 + p.rf_output_height_m
        parts.append(box_mesh(rf_x, rf_y, z0 - 0.04, 0.04, 0.04, 0.04 + wt))

        # ── 6. Warning / HV label ────────────────────────────────
        parts.append(box_mesh(
            x0 + cL * 0.35, y0 + cH * 0.70,
            z0 + cD, 0.15, 0.08, 0.001,
        ))

        self._vertices, self._triangles = concat_meshes(parts)
        return self._vertices, self._triangles, self._get_meta()

    def generate_rf_feed_mesh(
        self,
        electrode_gap_m: float = 0.200,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the RF feed connection from generator to electrodes.

        This is the physical copper busbar and tuning structure that
        carries RF power from the generator oscillator output, through
        the oven back wall, along the oven ceiling, and down to the
        upper electrode assembly via feed strips and tuning plates.

        Path (Engineering Guide §2.2.1):

            GENERATOR
                │  RF output at (x_center, rf_out_y, z = oven_width)
                │
                ├─ Conduit through oven back wall
                │
                ├─ Main busbar along oven ceiling (Y ≈ oven_height)
                │   runs from back wall (z = oven_width) to belt center
                │
                ├─ 4 Tuning feed plates (distribution, above RF zone)
                │
                └─ 6 Feed strip drops to electrode frame
                    (these already exist in electrode.py — this connects
                     the top of those strips to the busbar)

        Args:
            electrode_gap_m: Current electrode gap for strip positioning.

        Returns:
            (vertices, triangles, metadata)
        """
        p = self.params
        parts = []

        # Key coordinates
        oven_z_back = p.oven_width_m         # oven back wall
        gen_z_start = p.cabinet_z_start      # generator front face (includes gap)
        gen_x_center = p.cabinet_x_start + p.cabinet_length_m / 2
        z_belt_center = p.belt_z0_m + p.rf_zone_width_m / 2
        rf_x0 = p.rf_zone_x_start_m
        rf_xL = p.rf_zone_x_end_m - rf_x0   # RF zone length
        rf_x_center = (p.rf_zone_x_start_m + p.rf_zone_x_end_m) / 2

        y_ceiling = p.oven_height_m - 0.03   # just below oven ceiling
        bus_s = p.busbar_section_m            # busbar cross-section
        fs_t = p.feed_strip_thickness_m
        fs_w = p.feed_strip_width_m

        # ── 1. Enclosed conduit housing: generator → oven wall ─────
        # Runs at the generator's X centre so it physically connects
        # to the generator cabinet face.  Inside the oven, a
        # distribution rail connects the conduit to the RF zone.
        #
        #   Top view (X-Z):
        #
        #       RF zone              conduit           generator
        #       X=1.46..3.26    ←── distribution ──→  X=3.11
        #                            rail               │
        #       oven                                    │ conduit
        #       z=1.10 ────────────────────────────── z=2.40

        conduit_x = gen_x_center - bus_s / 2  # aligned to generator centre
        conduit_y = y_ceiling - bus_s

        # Conduit runs from oven back wall, across the gap, and
        # physically INTO the generator cabinet (past its front wall)
        conduit_z0 = oven_z_back                           # starts at oven wall
        conduit_z1 = gen_z_start + p.cabinet_depth_m * 0.5  # halfway into cabinet
        conduit_span = conduit_z1 - conduit_z0

        # Conduit enclosure dimensions (larger than busbar)
        enc_w = bus_s + 0.06              # enclosure width (X)
        enc_h = bus_s + 0.06              # enclosure height (Y)
        enc_cx = conduit_x - 0.03         # enclosure left edge X
        enc_cy = conduit_y - 0.03         # enclosure bottom edge Y
        wt_enc = 0.003                    # enclosure wall thickness

        # Bottom wall
        parts.append(box_mesh(enc_cx, enc_cy, conduit_z0,
                              enc_w, wt_enc, conduit_span))
        # Top wall (lid)
        parts.append(box_mesh(enc_cx, enc_cy + enc_h - wt_enc, conduit_z0,
                              enc_w, wt_enc, conduit_span))
        # Left side wall
        parts.append(box_mesh(enc_cx, enc_cy, conduit_z0,
                              wt_enc, enc_h, conduit_span))
        # Right side wall
        parts.append(box_mesh(enc_cx + enc_w - wt_enc, enc_cy, conduit_z0,
                              wt_enc, enc_h, conduit_span))

        # Copper busbar inside the conduit
        parts.append(box_mesh(conduit_x, conduit_y, conduit_z0,
                              bus_s, bus_s, conduit_span))

        # Flanged end plates where conduit meets oven and generator
        fl = 0.05
        fl_t = 0.006
        # Oven-side flange
        parts.append(box_mesh(
            enc_cx - fl / 2, enc_cy - fl / 2, conduit_z0 - fl_t,
            enc_w + fl, enc_h + fl, fl_t,
        ))
        # Generator-side flange
        parts.append(box_mesh(
            enc_cx - fl / 2, enc_cy - fl / 2, conduit_z1,
            enc_w + fl, enc_h + fl, fl_t,
        ))

        # Support brackets along the conduit span (every ~0.6 m)
        bracket_w = enc_w + 0.04
        bracket_h = 0.015
        bracket_d = 0.015
        n_brackets = max(2, int(conduit_span / 0.6))
        for i in range(n_brackets):
            bz = conduit_z0 + conduit_span * (i + 0.5) / n_brackets
            # U-bracket under conduit
            parts.append(box_mesh(
                enc_cx - 0.02, enc_cy - bracket_h, bz - bracket_d / 2,
                bracket_w, bracket_h, bracket_d,
            ))

        # ── 2. Vertical riser inside oven: conduit → ceiling ──────
        # From the conduit entry at the oven wall, a vertical riser
        # carries the busbar up to the oven ceiling level
        riser_y_bot = conduit_y
        riser_y_top = y_ceiling
        if riser_y_top > riser_y_bot + bus_s:
            # Riser enclosure
            parts.append(box_mesh(
                enc_cx, riser_y_bot, oven_z_back - enc_w,
                enc_w, riser_y_top - riser_y_bot, enc_w,
            ))
            # Copper inside riser
            parts.append(box_mesh(
                conduit_x, riser_y_bot, oven_z_back - bus_s - 0.03,
                bus_s, riser_y_top - riser_y_bot, bus_s,
            ))

        # ── 3. Main busbar along oven ceiling (Z direction) ────────
        # Runs from the oven back wall toward belt centre at the
        # conduit X position (generator's X centre).
        busbar_z0 = z_belt_center - bus_s / 2
        busbar_z1 = oven_z_back - 0.08
        busbar_len_z = busbar_z1 - busbar_z0
        parts.append(box_mesh(
            conduit_x, y_ceiling,
            busbar_z0,
            bus_s, bus_s, busbar_len_z,
        ))

        # ── 4. Distribution rail along X (above RF zone) ─────────
        # Spans the full RF zone length at belt centre Z,
        # connecting the feed strips to the main busbar.
        parts.append(box_mesh(
            rf_x0, y_ceiling,
            z_belt_center - bus_s / 2,
            rf_xL, bus_s, bus_s,
        ))

        # ── 4b. Connecting piece: distribution rail → main busbar ─
        # The distribution rail (at rf_x_center) and main busbar
        # (at gen_x_center) are at different X positions.  A short
        # bus connects them along X at the ceiling.
        conn_x_min = min(rf_x_center, conduit_x + bus_s / 2)
        conn_x_max = max(rf_x_center, conduit_x + bus_s / 2)
        conn_x_len = conn_x_max - conn_x_min
        if conn_x_len > bus_s:
            parts.append(box_mesh(
                conn_x_min, y_ceiling,
                z_belt_center - bus_s / 2,
                conn_x_len, bus_s, bus_s,
            ))

        # ── 5. Tuning feed plates (4, from ceiling rail downward) ─
        # These hang from the distribution rail and bridge to the
        # electrode feed strip tops
        n_tp = p.tuning_plate_count
        tp_w = 0.08   # plate width (X)
        tp_d = 0.06   # plate depth (Z)
        tp_h = 0.15   # how far they hang down from rail

        tp_xs = np.linspace(rf_x0 + 0.10, rf_x0 + rf_xL - 0.10, n_tp)
        for tx in tp_xs:
            parts.append(box_mesh(
                float(tx) - tp_w / 2, y_ceiling - tp_h,
                z_belt_center - tp_d / 2,
                tp_w, tp_h, tp_d,
            ))

        # ── 6. Tuning bridges (8, small cross-pieces) ────────────
        # Short horizontal bars between adjacent tuning plates
        n_tb = p.tuning_bridge_count
        tb_xs = np.linspace(rf_x0 + 0.05, rf_x0 + rf_xL - 0.05, n_tb)
        for tx in tb_xs:
            parts.append(box_mesh(
                float(tx) - 0.015, y_ceiling - tp_h - 0.01,
                z_belt_center - 0.025,
                0.030, 0.010, 0.050,
            ))

        # ── 7. Feed strip extensions (from tuning plates down to ──
        #        electrode frame top — bridges the gap between the
        #        tuning structure and the existing electrode feed strips)
        # The electrode feed strips top out at:
        #   y = gap + plate_t(0.003) + silicon(0.015) + beam(0.040) + strip_h(0.060)
        #     ≈ gap + 0.118
        # The tuning plates bottom is at y_ceiling - tp_h
        feed_strip_top_y = electrode_gap_m + 0.118
        tuning_bottom_y = y_ceiling - tp_h

        if tuning_bottom_y > feed_strip_top_y + 0.02:
            drop_h = tuning_bottom_y - feed_strip_top_y
            n_fs = p.feed_strip_count
            fs_xs = np.linspace(rf_x0 + 0.08, rf_x0 + rf_xL - 0.08, n_fs)
            for fx in fs_xs:
                parts.append(box_mesh(
                    float(fx), feed_strip_top_y,
                    z_belt_center - fs_w / 2,
                    fs_t, drop_h, fs_w,
                ))

        verts, tris = concat_meshes(parts)
        return verts, tris, {
            "type": "rf_feed_connection",
            "feed_strip_count": p.feed_strip_count,
            "tuning_plate_count": p.tuning_plate_count,
            "tuning_bridge_count": p.tuning_bridge_count,
        }

    def _get_meta(self) -> dict:
        p = self.params
        return {
            "type": "generator_cabinet",
            "x_start": p.cabinet_x_start,
            "z_start": p.cabinet_z_start,
            "z_end": p.cabinet_z_start + p.cabinet_depth_m,
            "height_m": p.cabinet_height_m,
            "floor_y": p.base_y,
        }

    @property
    def ports(self) -> Dict[str, "ConnectionPort"]:
        from airclassifier.geometry.connection_ports import ConnectionPort, PortType
        p = self.params
        return {
            'rf_output': ConnectionPort(
                position=(
                    p.cabinet_x_start + p.cabinet_length_m / 2,
                    p.base_y + p.rf_output_height_m,
                    p.cabinet_z_start,
                ),
                direction=(0.0, 0.0, -1.0),  # Points into oven (-Z)
                diameter=p.rf_output_diameter_m,
                port_type=PortType.SLIP,
                name="rf_output",
            ),
        }
