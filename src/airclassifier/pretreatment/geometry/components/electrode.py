"""
Electrode Geometry
==================

GP-15 electrode system: upper and lower electrode assemblies positioned
inside the oven chamber, relative to the conveyor belt.

Physical reality (Manual + Engineering Guide §2.2.1, §2.2.2):

    Upper electrode assembly:
      - Two perforated aluminium plates (11.05.433) side-by-side along X
      - Supported by 6 longitudinal beams (electrode support frame)
      - 8 silicon insulation supports between plates and frame
      - 6 copper feed strips connecting oscillator tank circuit
      - Frame moves vertically on 4 lead screws (0.37 kW motor)
      - Gap range: 20–300 mm (adjustable during operation)

    Lower electrode assembly:
      - Two removable aluminium trays (21-0042-0180) with U-handles
      - Sit on PET insulating supports on the deck plate
      - Covered by 2 top sheets + 24 Teflon wear strips
      - 4 copper chokes (18-turn) for RF impedance matching
      - This is the GROUND electrode (V = 0)

    Cross-section (Y-axis) inside the oven:

        ─ ─ ─ ─ ─ ─ upper electrode frame beams ─ ─ ─ ─ ─
        ┌─────────────┐  ┌─────────────┐
        │  Plate 1    │  │  Plate 2    │   perforated
        │ (holes)     │  │ (holes)     │   aluminium
        └─────────────┘  └─────────────┘
              ↕ seam gap (2 mm)
              ↕
              ↕ electrode gap (20–300 mm)
              ↕
        ┌─────────────────────────────────┐
        │     Tray 1      │    Tray 2     │  solid aluminium
        └─────────────────────────────────┘
           ↑ top sheets + wear strips ↑

Coordinate system:
    Same as oven and conveyor: Y-up, y = 0 at lower electrode surface.
    Electrodes span the RF zone within the oven, aligned to belt Z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import box_mesh, concat_meshes

if TYPE_CHECKING:
    from airclassifier.geometry.connection_ports import ConnectionPort
    from ...config import MachineConfig


@dataclass
class ElectrodeParams:
    """GP-15 electrode geometry parameters.

    All positions are relative to the oven/conveyor coordinate system
    where y = 0 is the deck-plate surface.
    """

    # ── Upper electrode ──────────────────────────────────────────
    plate_count: int = 2                      # two plates side-by-side in X
    plate_thickness_m: float = 0.003          # [m] aluminium plate
    seam_gap_m: float = 0.002                 # [m] gap between two plates

    # Frame beams above the plates
    frame_beam_count: int = 6                 # longitudinal beams
    frame_beam_height_m: float = 0.040        # [m] beam section height
    frame_beam_width_m: float = 0.025         # [m] beam section width

    # Silicon supports (insulation between plate and frame)
    silicon_support_count: int = 8
    silicon_support_height_m: float = 0.015   # [m]

    # Feed strips (copper, from oscillator)
    feed_strip_count: int = 6
    feed_strip_width_m: float = 0.050         # [m] ~2 inches
    feed_strip_thickness_m: float = 0.002     # [m]
    feed_strip_height_m: float = 0.060        # [m] extends above frame

    # Perforation pattern
    perforation_diameter_m: float = 0.006     # [m] circular holes
    perforation_pitch_m: float = 0.012        # [m] centre-to-centre

    # ── Lower electrode ──────────────────────────────────────────
    tray_count: int = 2                       # two trays side-by-side in X
    tray_thickness_m: float = 0.005           # [m] solid aluminium tray
    tray_lip_height_m: float = 0.008          # [m] U-handle lip
    tray_lip_width_m: float = 0.003           # [m]

    # PET supports under trays
    pet_support_height_m: float = 0.012       # [m]
    pet_support_count: int = 6

    # Copper chokes
    choke_count: int = 4
    choke_diameter_m: float = 0.030           # [m] coil OD
    choke_height_m: float = 0.025             # [m]

    # ── Sizing (from oven RF zone) ───────────────────────────────
    # These are set from OvenChamberParams at construction time
    rf_zone_length_m: float = 1.50            # [m]
    rf_zone_width_m: float = 0.80             # [m] = belt width
    rf_zone_x_start_m: float = 2.00           # [m] in conveyor coords
    belt_z0_m: float = 0.15                   # [m] belt left edge in Z

    @classmethod
    def from_oven(cls, oven_params) -> "ElectrodeParams":
        """Create electrode params aligned to oven geometry."""
        return cls(
            rf_zone_length_m=oven_params.rf_zone_length_m,
            rf_zone_width_m=oven_params.rf_zone_width_m,
            rf_zone_x_start_m=oven_params.rf_zone_x_start,
            belt_z0_m=oven_params.conveyor_belt_z0_m,
        )

    @property
    def total_plate_length_m(self) -> float:
        """Combined length of all plates + seam gaps."""
        return (self.plate_count * self.plate_length_m
                + (self.plate_count - 1) * self.seam_gap_m)

    @property
    def plate_length_m(self) -> float:
        """Length of a single electrode plate (X)."""
        gap_total = (self.plate_count - 1) * self.seam_gap_m
        return (self.rf_zone_length_m - gap_total) / self.plate_count


class ElectrodeGeometry:
    """Generates GP-15 upper and lower electrode meshes.

    The electrodes live inside the oven chamber.  The upper electrode
    moves vertically (gap adjustment); the lower electrode is fixed.

    Also provides Phase-3 field correction maps:
      - Perforation correction (area-average + hole-edge enhancement)
      - Centre-seam correction (field dip at plate joint)
      - Palmer fringe-field correction (edge enhancement in Z)
    """

    def __init__(self, params: Optional[ElectrodeParams] = None):
        self.params = params or ElectrodeParams()
        self._upper_verts: Optional[np.ndarray] = None
        self._upper_tris: Optional[np.ndarray] = None
        self._lower_verts: Optional[np.ndarray] = None
        self._lower_tris: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────────
    #  Upper electrode
    # ─────────────────────────────────────────────────────────────

    def generate_upper_mesh(
        self, electrode_gap_m: float,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate upper electrode assembly at the given gap.

        Components:
          1. Two perforated plates (modelled as solid slabs — perforation
             is a field correction, not a mesh feature)
          2. Silicon insulation supports
          3. Frame beams
          4. Copper feed strips

        The bottom of the plates sits at y = electrode_gap_m.

        Args:
            electrode_gap_m: Current electrode gap [m].

        Returns:
            (vertices, triangles, metadata)
        """
        p = self.params
        x0 = p.rf_zone_x_start_m
        z0 = p.belt_z0_m
        W = p.rf_zone_width_m
        pL = p.plate_length_m
        pt = p.plate_thickness_m
        sg = p.seam_gap_m
        y_plate_bot = electrode_gap_m

        parts = []

        # ── 1. Electrode plates ──────────────────────────────────
        for i in range(p.plate_count):
            px = x0 + i * (pL + sg)
            parts.append(box_mesh(px, y_plate_bot, z0, pL, pt, W))

        # ── 2. Silicon supports (between plates and frame) ───────
        y_sil_bot = y_plate_bot + pt
        sil_h = p.silicon_support_height_m
        sil_w = 0.020
        sil_d = 0.020
        n_sil = p.silicon_support_count
        sil_xs = np.linspace(x0 + 0.05, x0 + p.rf_zone_length_m - 0.05, n_sil)
        for sx in sil_xs:
            # Two per X position: near z0 and near z0+W
            parts.append(box_mesh(float(sx), y_sil_bot, z0 + 0.03,
                                  sil_w, sil_h, sil_d))
            parts.append(box_mesh(float(sx), y_sil_bot, z0 + W - 0.03 - sil_d,
                                  sil_w, sil_h, sil_d))

        # ── 3. Frame beams (longitudinal, above silicon supports) ─
        y_frame_bot = y_sil_bot + sil_h
        fb_h = p.frame_beam_height_m
        fb_w = p.frame_beam_width_m
        n_fb = p.frame_beam_count
        fb_zs = np.linspace(z0, z0 + W - fb_w, n_fb)
        for fz in fb_zs:
            parts.append(box_mesh(x0, y_frame_bot, float(fz),
                                  p.rf_zone_length_m, fb_h, fb_w))

        # ── 4. Copper feed strips ────────────────────────────────
        y_fs_bot = y_frame_bot + fb_h
        fs_w = p.feed_strip_width_m
        fs_t = p.feed_strip_thickness_m
        fs_h = p.feed_strip_height_m
        n_fs = p.feed_strip_count
        fs_xs = np.linspace(x0 + 0.08, x0 + p.rf_zone_length_m - 0.08, n_fs)
        for fx in fs_xs:
            # Vertical strip from frame top upward
            parts.append(box_mesh(float(fx), y_fs_bot, z0 + W / 2 - fs_w / 2,
                                  fs_t, fs_h, fs_w))

        self._upper_verts, self._upper_tris = concat_meshes(parts)
        return self._upper_verts, self._upper_tris, {
            "type": "upper_electrode",
            "gap_m": electrode_gap_m,
            "plate_count": p.plate_count,
            "plate_length_m": pL,
            "plate_y_bottom": y_plate_bot,
            "plate_y_top": y_plate_bot + pt,
            "frame_y_top": y_frame_bot + fb_h,
        }

    # ─────────────────────────────────────────────────────────────
    #  Lower electrode
    # ─────────────────────────────────────────────────────────────

    def generate_lower_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate lower electrode assembly at y = 0 (ground).

        Components:
          1. PET insulating supports (on deck plate)
          2. Two electrode trays with U-handle lips
          3. Copper chokes (at tray ends)

        The tray top surface is at y = 0 (this is where the top sheet,
        wear strips, belt, and product stack begins).

        Returns:
            (vertices, triangles, metadata)
        """
        p = self.params
        x0 = p.rf_zone_x_start_m
        z0 = p.belt_z0_m
        W = p.rf_zone_width_m
        tL_single = p.plate_length_m            # reuse same plate length for trays
        tt = p.tray_thickness_m
        sg = p.seam_gap_m

        parts = []

        # ── 1. PET supports (under trays, on deck plate) ────────
        pet_h = p.pet_support_height_m
        pet_w = 0.025
        pet_d = W
        y_pet_bot = -tt - pet_h                  # below tray bottom
        n_pet = p.pet_support_count
        pet_xs = np.linspace(x0 + 0.05, x0 + p.rf_zone_length_m - 0.05, n_pet)
        for px in pet_xs:
            parts.append(box_mesh(float(px), y_pet_bot, z0,
                                  pet_w, pet_h, pet_d))

        # ── 2. Electrode trays ───────────────────────────────────
        y_tray_bot = -tt                          # tray bottom; top at y=0
        for i in range(p.tray_count):
            tx = x0 + i * (tL_single + sg)
            # Main tray body
            parts.append(box_mesh(tx, y_tray_bot, z0, tL_single, tt, W))

            # U-handle lips (raised edges at X ends of each tray)
            lip_h = p.tray_lip_height_m
            lip_w = p.tray_lip_width_m
            # Infeed lip
            parts.append(box_mesh(tx, 0.0, z0,
                                  lip_w, lip_h, W))
            # Outfeed lip
            parts.append(box_mesh(tx + tL_single - lip_w, 0.0, z0,
                                  lip_w, lip_h, W))

        # ── 3. Copper chokes (at corners of tray assembly) ───────
        ch_d = p.choke_diameter_m
        ch_h = p.choke_height_m
        y_ch = y_tray_bot - ch_h
        choke_positions = [
            (x0 + 0.03, z0 + 0.03),
            (x0 + 0.03, z0 + W - 0.03 - ch_d),
            (x0 + p.rf_zone_length_m - 0.03 - ch_d, z0 + 0.03),
            (x0 + p.rf_zone_length_m - 0.03 - ch_d, z0 + W - 0.03 - ch_d),
        ]
        for cx, cz in choke_positions:
            parts.append(box_mesh(cx, y_ch, cz, ch_d, ch_h, ch_d))

        self._lower_verts, self._lower_tris = concat_meshes(parts)
        return self._lower_verts, self._lower_tris, {
            "type": "lower_electrode",
            "tray_count": p.tray_count,
            "tray_y_top": 0.0,
            "tray_y_bottom": y_tray_bot,
        }

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate lower electrode (compatibility)."""
        return self.generate_lower_mesh()

    # ─────────────────────────────────────────────────────────────
    #  Field correction maps (Phase 3)
    # ─────────────────────────────────────────────────────────────

    def get_perforation_correction_field(
        self, grid_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """2-D correction (nx, nz) for perforation + centre seam.

        Engineering guide §4.1.5::

            η_perf = 1 / (1 − π·d²/(4·p²))

        Centre seam creates ~5% dip at x = L/2.

        Returns:
            Array (nx, nz), values 0.92–1.08.
        """
        p = self.params
        nx, _, nz = grid_shape
        d = p.perforation_diameter_m
        pitch = p.perforation_pitch_m

        hole_frac = math.pi * d * d / (4.0 * pitch * pitch)
        eta = 1.0 / max(1.0 - hole_frac, 0.5)

        correction = np.full((nx, nz), eta, dtype=np.float32)

        # Centre seam dip
        i_seam = nx // 2
        sw = max(1, nx // 40)
        for di in range(-sw, sw + 1):
            ii = i_seam + di
            if 0 <= ii < nx:
                fade = 1.0 - 0.05 * max(0, 1.0 - abs(di) / max(sw, 1))
                correction[ii, :] *= fade

        return correction

    def get_fringe_field_correction(
        self, grid_shape: Tuple[int, int, int],
        electrode_gap_m: float,
    ) -> np.ndarray:
        """1-D fringe correction in Z (belt width).

        Palmer formula (§4.1.4)::

            C_fringe/C_uniform = 1 + (d/(π·w)) · [1 + ln(2π·w/d)]

        Returns:
            Array (nz,), mean ≈ 1.0, enhanced at edges.
        """
        _, _, nz = grid_shape
        w = self.params.rf_zone_width_m
        gap = max(electrode_gap_m, 0.01)

        ratio = gap / (math.pi * w)
        C_ratio = 1.0 + ratio * (1.0 + math.log(2.0 * math.pi * w / gap))

        z_norm = np.linspace(0, 1, nz)
        edge_dist = np.minimum(z_norm, 1.0 - z_norm)
        scale = gap / w
        enhancement = (C_ratio - 1.0) * 2.0
        profile = 1.0 + enhancement * np.exp(-edge_dist / max(scale, 0.01))
        profile /= np.mean(profile)

        return profile.astype(np.float32)

    @property
    def ports(self) -> Dict[str, "ConnectionPort"]:
        """Electrode is internal — no external ports."""
        return {}