"""
EMU (Environment Management Unit) Geometry
============================================

The EMU sits at the **back** of the GP-15 oven chamber (behind it in
+Z) on the **infeed half** of the oven.  It contains the heater banks
and blower fans that circulate warm air through the oven, plus the
extraction duct and fan on top.

The RF generator sits adjacent to the EMU, also at the back, on the
outfeed half — see ``generator.py``.

Physical layout (Manual — Illustration 3):

    Top-down (X–Z), looking down:

        ═══════════════════════════════════
        belt direction →→→→→                z = 0 (front/operator side)
        ═══════════════════════════════════
        ┌──────────────────────────────────┐
        │         OVEN CHAMBER             │  z = 0 … oven_width
        └──────────────────────────────────┘
        ┌──────────┐         ┌─────────────┐
        │   EMU    │         │  GENERATOR  │  z = oven_width … +depth
        │ heaters  │         │  (RF osc.)  │
        │ blower   │         │   vents ▦▦  │
        └──────────┘         └─────────────┘
        ↑ infeed half         ↑ outfeed half

    Both sit on the floor behind the oven, at the same Y level
    as the conveyor frame base.

Coordinate system:
    Y = 0 is deck/belt surface.  Floor is at Y = -(frame_height + leg_height).
    The EMU and generator stand on the floor, so their bases are at floor Y.
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
class EMUParams:
    """Environment Management Unit housing parameters.

    The EMU sits behind the oven (+Z), on the infeed half (lower X).
    Its base is on the floor, and it extends up to above oven height
    to house the extraction duct and fan.
    """

    # ── Housing envelope ──────────────────────────────────────────
    housing_length_m: float = 1.00       # [m] along X (covers infeed half of oven back)
    housing_depth_m: float = 0.60        # [m] along Z (how far behind the oven)
    housing_height_m: float = 2.70       # [m] total height from floor (above oven ceiling for fan)

    # ── Construction ──────────────────────────────────────────────
    wall_thickness_m: float = 0.004      # [m] sheet metal

    # ── Extraction duct (top, fan-driven exhaust) ─────────────────
    duct_diameter_m: float = 0.25        # [m] 250 mm (Manual)
    duct_height_m: float = 0.35          # [m] duct stub above housing

    # ── Heater bank boxes (2 banks × 6 × 1 kW = 12 kW total) ─────
    heater_box_width_m: float = 0.30     # [m]
    heater_box_height_m: float = 0.25    # [m]
    heater_box_depth_m: float = 0.12     # [m]
    heater_bank_count: int = 2

    # ── Blower fan ────────────────────────────────────────────────
    blower_diameter_m: float = 0.20      # [m]

    # ── Access panels (on back face, +Z) ──────────────────────────
    access_panel_count: int = 2
    access_panel_width_m: float = 0.30
    access_panel_height_m: float = 0.45

    # ── Gap between EMU and oven ────────────────────────────────
    gap_from_oven_m: float = 1.30        # [m] 130 cm clearance

    # ── Positioning (set by from_oven) ────────────────────────────
    oven_x_start_m: float = 1.65
    oven_x_end_m: float = 4.15
    oven_width_m: float = 1.10           # [m] oven Z extent (back wall at this Z)
    oven_height_m: float = 1.10          # [m] oven internal height
    floor_y_m: float = -1.30             # [m] floor level

    @classmethod
    def from_oven(
        cls,
        oven_params: "OvenChamberParams",
        floor_y_m: float = -1.30,
    ) -> "EMUParams":
        """Create EMU params: behind the oven, infeed half.

        Height is calculated relative to the oven ceiling so the
        extraction duct extends above the oven top.
        """
        # EMU top = oven ceiling + 0.50 m clearance for extraction duct/fan
        emu_top_y = oven_params.oven_height_m + 0.50
        height = emu_top_y - floor_y_m

        return cls(
            housing_height_m=height,
            oven_x_start_m=oven_params.oven_x_start_m,
            oven_x_end_m=oven_params.oven_x_end_m,
            oven_width_m=oven_params.oven_width_m,
            oven_height_m=oven_params.oven_height_m,
            duct_diameter_m=oven_params.extraction_diameter_m,
            floor_y_m=floor_y_m,
        )

    @property
    def emu_x_start(self) -> float:
        """X position — aligned to oven infeed end."""
        return self.oven_x_start_m

    @property
    def emu_z_start(self) -> float:
        """Z position — behind the oven back wall, with gap."""
        return self.oven_width_m + self.gap_from_oven_m

    @property
    def base_y(self) -> float:
        """Y position of the EMU base (floor level)."""
        return self.floor_y_m


class EMUGeometry:
    """EMU housing behind the oven (infeed half).

    Stands on the floor behind the oven's back wall.  Contains heater
    banks, blower fan, and extraction duct with fan motor on top.

    Air flow: intake filters → heaters → blower → into oven →
              over product → extraction fan on top → exhaust.
    """

    def __init__(self, params: Optional[EMUParams] = None):
        self.params = params or EMUParams()
        self._vertices: Optional[np.ndarray] = None
        self._triangles: Optional[np.ndarray] = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        if self._vertices is not None:
            return self._vertices, self._triangles, self._get_meta()

        p = self.params
        wt = p.wall_thickness_m

        x0 = p.emu_x_start
        z0 = p.emu_z_start
        y0 = p.base_y            # floor level
        hL = p.housing_length_m   # X extent
        hD = p.housing_depth_m    # Z extent (behind oven)
        hH = p.housing_height_m   # Y extent from floor

        parts = []

        # ── 1. Main housing walls ─────────────────────────────────
        # Left wall (x = x0)
        parts.append(box_mesh(x0, y0, z0, wt, hH, hD))
        # Right wall (x = x0 + hL)
        parts.append(box_mesh(x0 + hL - wt, y0, z0, wt, hH, hD))
        # Top
        parts.append(box_mesh(x0, y0 + hH, z0, hL, wt, hD))
        # Back face (+Z, away from oven)
        parts.append(box_mesh(x0, y0, z0 + hD - wt, hL, hH, wt))
        # Oven-facing wall (z = z0) — has air passage opening to oven
        passage_h = 0.70
        passage_margin = 0.08
        # Above passage
        parts.append(box_mesh(x0, y0 + passage_h, z0, hL, hH - passage_h - abs(y0), wt))
        # Below passage (sides)
        parts.append(box_mesh(x0, y0, z0, passage_margin, passage_h, wt))
        parts.append(box_mesh(x0 + hL - passage_margin, y0, z0,
                              passage_margin, passage_h, wt))
        # Bottom
        parts.append(box_mesh(x0, y0, z0, hL, wt, hD))

        # ── 2. Internal shelf ─────────────────────────────────────
        shelf_y = y0 + hH * 0.50
        parts.append(box_mesh(x0 + wt, shelf_y, z0 + wt,
                              hL - 2 * wt, wt, hD - 2 * wt))

        # ── 3. Extraction duct + fan motor on top ─────────────────
        duct_cx = x0 + hL / 2
        duct_cz = z0 + hD / 2
        duct_r = p.duct_diameter_m / 2
        # Duct cylinder
        parts.append(cylinder_mesh(
            center=(duct_cx, y0 + hH + wt, duct_cz),
            radius=duct_r, height=p.duct_height_m,
            resolution=16, axis="y",
        ))
        # Fan motor (smaller cylinder on top)
        parts.append(cylinder_mesh(
            center=(duct_cx, y0 + hH + wt + p.duct_height_m, duct_cz),
            radius=duct_r * 0.55, height=0.12,
            resolution=12, axis="y",
        ))
        # Duct flange
        fl = 0.03
        parts.append(box_mesh(
            duct_cx - duct_r - fl, y0 + hH,
            duct_cz - duct_r - fl,
            (duct_r + fl) * 2, wt, (duct_r + fl) * 2,
        ))

        # ── 4. Supply air duct (EMU → oven, across the gap) ───────
        # Rectangular duct carrying heated air from the EMU blower
        # through the gap to the oven back wall.
        oven_z_back = p.oven_width_m             # oven back wall Z
        duct_gap = z0 - oven_z_back              # gap span
        supply_w = hL - 2 * passage_margin        # duct width (X)
        supply_h = 0.40                            # duct height (Y)
        supply_y = 0.05                            # just above deck
        supply_wt = 0.003                          # duct wall thickness

        # Bottom
        parts.append(box_mesh(x0 + passage_margin, supply_y,
                              oven_z_back, supply_w, supply_wt, duct_gap))
        # Top
        parts.append(box_mesh(x0 + passage_margin, supply_y + supply_h,
                              oven_z_back, supply_w, supply_wt, duct_gap))
        # Left side
        parts.append(box_mesh(x0 + passage_margin, supply_y,
                              oven_z_back, supply_wt, supply_h, duct_gap))
        # Right side
        parts.append(box_mesh(x0 + hL - passage_margin - supply_wt, supply_y,
                              oven_z_back, supply_wt, supply_h, duct_gap))
        # Flanges at both ends
        fl_d = 0.006
        fl_e = 0.03
        parts.append(box_mesh(x0 + passage_margin - fl_e,
                              supply_y - fl_e, oven_z_back - fl_d,
                              supply_w + 2 * fl_e, supply_h + 2 * fl_e, fl_d))
        parts.append(box_mesh(x0 + passage_margin - fl_e,
                              supply_y - fl_e, z0,
                              supply_w + 2 * fl_e, supply_h + 2 * fl_e, fl_d))

        # ── 5. Extraction duct (oven top → EMU top, across gap) ───
        # The extraction path:
        #   a) Vertical stub through oven ceiling (visible on oven top)
        #   b) Horizontal run from oven back wall across gap to EMU
        #   c) Vertical riser up to EMU top where fan sits
        ext_r = p.duct_diameter_m / 2
        ext_section = p.duct_diameter_m + 0.02    # square duct section
        ext_cx = x0 + hL / 2                      # centred on EMU in X
        ext_fl = 0.03

        # (a) Vertical stub ON TOP of oven — visible connection point
        #     Goes from inside the oven (y = oven_height - 0.08)
        #     up through the oven ceiling to above it
        stub_y_bot = p.oven_height_m - 0.08
        stub_y_top = p.oven_height_m + ext_section + 0.02
        stub_z_center = p.oven_width_m / 2        # oven centre in Z
        parts.append(box_mesh(
            ext_cx - ext_section / 2, stub_y_bot,
            stub_z_center - ext_section / 2,
            ext_section, stub_y_top - stub_y_bot, ext_section,
        ))
        # Flange ring on oven ceiling
        parts.append(box_mesh(
            ext_cx - ext_section / 2 - ext_fl,
            p.oven_height_m - 0.003,
            stub_z_center - ext_section / 2 - ext_fl,
            ext_section + 2 * ext_fl, 0.006,
            ext_section + 2 * ext_fl,
        ))

        # (b) Horizontal run from stub top across to oven back wall
        #     and then across the gap to the EMU
        horiz_y = stub_y_top - ext_section        # align with stub top
        horiz_z_start = stub_z_center + ext_section / 2  # from stub
        horiz_z_end = z0 + ext_section              # into EMU
        horiz_z_len = horiz_z_end - horiz_z_start
        if horiz_z_len > 0.01:
            parts.append(box_mesh(
                ext_cx - ext_section / 2, horiz_y,
                horiz_z_start,
                ext_section, ext_section, horiz_z_len,
            ))
            # Flanges at transitions
            parts.append(box_mesh(
                ext_cx - ext_section / 2 - ext_fl,
                horiz_y - ext_fl, oven_z_back - fl_d,
                ext_section + 2 * ext_fl, ext_section + 2 * ext_fl, fl_d,
            ))

        # (c) Vertical riser from horizontal run up to EMU top duct
        riser_y_bot = horiz_y + ext_section
        riser_y_top = y0 + hH
        if riser_y_top > riser_y_bot:
            parts.append(box_mesh(
                ext_cx - ext_section / 2, riser_y_bot,
                z0, ext_section, riser_y_top - riser_y_bot, ext_section,
            ))

        # ── 7. Heater bank intake louvres (back face) ────────────
        hb_w = p.heater_box_width_m
        hb_h = p.heater_box_height_m
        hb_d = p.heater_box_depth_m
        for i in range(p.heater_bank_count):
            hb_x = x0 + hL * (0.15 + 0.45 * i)
            hb_y = y0 + hH * 0.15
            # Protruding louvre on back face
            parts.append(box_mesh(hb_x, hb_y, z0 + hD, hb_w, hb_h, hb_d))

        # ── 8. Access panel outlines (back face) ─────────────────
        pw = p.access_panel_width_m
        ph = p.access_panel_height_m
        trim = 0.004
        for i in range(p.access_panel_count):
            px = x0 + hL * (0.15 + 0.45 * i)
            py = y0 + hH * 0.55
            parts.append(box_mesh(px, py, z0 + hD, pw, ph, trim))

        # ── 6. Base feet (4 adjustable feet) ─────────────────────
        foot_s = 0.05
        foot_h = 0.04
        for fx in [x0 + 0.05, x0 + hL - 0.05 - foot_s]:
            for fz in [z0 + 0.05, z0 + hD - 0.05 - foot_s]:
                parts.append(box_mesh(fx, y0 - foot_h, fz, foot_s, foot_h, foot_s))

        self._vertices, self._triangles = concat_meshes(parts)
        return self._vertices, self._triangles, self._get_meta()

    def _get_meta(self) -> dict:
        p = self.params
        return {
            "type": "emu_housing",
            "x_start": p.emu_x_start,
            "z_start": p.emu_z_start,
            "z_end": p.emu_z_start + p.housing_depth_m,
            "height_m": p.housing_height_m,
            "floor_y": p.base_y,
        }

    @property
    def ports(self) -> Dict[str, "ConnectionPort"]:
        """EMU connection ports.

        - air_to_oven:       supply air outlet → oven back wall air_supply port
        - extraction_inlet:  extraction duct inlet ← oven extraction port
        - exhaust:           fan exhaust (top, to atmosphere)
        """
        from airclassifier.geometry.connection_ports import ConnectionPort, PortType
        p = self.params
        emu_x_center = p.emu_x_start + p.housing_length_m / 2

        return {
            'air_to_oven': ConnectionPort(
                position=(
                    emu_x_center,
                    0.25,           # matches oven air_supply Y
                    p.oven_width_m,  # oven back wall Z (duct spans the gap)
                ),
                direction=(0.0, 0.0, -1.0),  # points toward oven (-Z)
                width=p.housing_length_m - 0.16,
                height=0.40,       # matches oven air_supply height
                port_type=PortType.RECTANGULAR,
                name="emu_air_to_oven",
            ),
            'extraction_inlet': ConnectionPort(
                position=(
                    emu_x_center,
                    p.oven_height_m,  # oven ceiling height
                    p.oven_width_m,   # oven back wall Z
                ),
                direction=(0.0, 0.0, -1.0),  # from oven ceiling via duct
                diameter=p.duct_diameter_m + 0.02,  # duct section
                port_type=PortType.CIRCULAR,
                name="emu_extraction_inlet",
            ),
            'exhaust': ConnectionPort(
                position=(
                    emu_x_center,
                    p.base_y + p.housing_height_m + p.duct_height_m,
                    p.emu_z_start + p.housing_depth_m / 2,
                ),
                direction=(0.0, 1.0, 0.0),  # up to atmosphere
                diameter=p.duct_diameter_m,
                port_type=PortType.CIRCULAR,
                name="emu_exhaust",
            ),
        }
