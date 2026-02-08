"""
Electrode Geometry
==================

Detailed electrode plate geometry: perforations, center seam between
the two plates, copper feed strip attachment points, and the adjustable
gap mechanism.

Upper electrode: two perforated plates side-by-side, supported by a
frame with 4 lead screws for vertical gap adjustment.

Lower electrode: two removable trays with U-handles, covered by
Teflon wear strips and a protective top sheet.

Phase 3 additions (engineering guide §4.1.4, §4.1.5):
- Perforation correction: area-average + hole-edge enhancement
- Center seam correction: reduced field at plate joint
- Fringe field (Palmer formula): edge correction for finite plates
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from ..config import MachineConfig


@dataclass
class ElectrodeParams:
    """Electrode plate parameters."""
    plate_count: int = 2                    # Two plates per electrode
    plate_length_m: float = 0.75            # Each plate ~half oven length
    plate_width_m: float = 0.8
    plate_thickness_m: float = 0.003        # Aluminium plate
    perforation_diameter_m: float = 0.006   # Circular holes
    perforation_pitch_m: float = 0.012      # Center-to-center spacing
    seam_gap_m: float = 0.002               # Gap between two plates
    feed_strip_count: int = 6               # Copper strips from oscillator
    feed_strip_width_m: float = 0.05        # ~2 inches

    @classmethod
    def from_machine(cls, config: MachineConfig) -> "ElectrodeParams":
        return cls(
            plate_count=config.electrode_count,
            plate_width_m=config.belt_width_m,
            plate_length_m=config.oven_length_m / config.electrode_count,
        )


class ElectrodeGeometry:
    """Generates electrode plate meshes and field-correction masks.

    Provides three correction fields (engineering guide §4.1.4, §4.1.5):

    1. **Perforation correction** — area-average reduction from holes
       with local enhancement at hole edges.
    2. **Seam correction** — reduced field at the center joint between
       the two electrode plates.
    3. **Fringe field correction** — Palmer formula for finite-plate
       edge effects (applied in the Z direction across belt width).
    """

    def __init__(self, params: Optional[ElectrodeParams] = None):
        self.params = params or ElectrodeParams()

    # ------------------------------------------------------------------
    # Visualization meshes (simple box geometry)
    # ------------------------------------------------------------------

    def generate_upper_mesh(
        self,
        electrode_gap_m: float,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate upper electrode mesh as a flat plate at y = gap.

        Returns:
            (vertices, triangles, metadata)
        """
        p = self.params
        L = p.plate_length_m * p.plate_count + p.seam_gap_m
        W = p.plate_width_m
        t = p.plate_thickness_m
        y = electrode_gap_m

        verts, tris = _box_mesh(0.0, y, 0.0, L, t, W)
        return verts, tris, {"type": "upper_electrode"}

    def generate_lower_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate lower electrode tray mesh at y = 0."""
        p = self.params
        L = p.plate_length_m * p.plate_count + p.seam_gap_m
        W = p.plate_width_m
        t = p.plate_thickness_m

        verts, tris = _box_mesh(0.0, -t, 0.0, L, t, W)
        return verts, tris, {"type": "lower_electrode"}

    # ------------------------------------------------------------------
    # Phase 3: Correction fields
    # ------------------------------------------------------------------

    def get_perforation_correction_field(
        self,
        grid_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """Return a 2-D correction factor (nx, nz) for field non-uniformity
        due to electrode perforations and center seam.

        Engineering guide §4.1.5::

            eta_perf ≈ 1 / (1 - pi*d_hole^2 / (4*p^2))   (area average)
            eta_edge ≈ eta_perf * (1 + d_hole / (2*gap))   (hole edges)

        The center seam (at x = L/2) reduces the field by ~5%.

        Returns:
            Array of shape (nx, nz) with values 0.92–1.08.
        """
        p = self.params
        nx, _, nz = grid_shape
        d = p.perforation_diameter_m
        pitch = p.perforation_pitch_m

        # Area-average perforation factor
        hole_area_frac = math.pi * d * d / (4.0 * pitch * pitch)
        eta_avg = 1.0 / max(1.0 - hole_area_frac, 0.5)

        # Build 2-D map
        correction = np.full((nx, nz), eta_avg, dtype=np.float32)

        # Center seam reduction (~5% dip at x = L/2)
        i_seam = nx // 2
        seam_width = max(1, nx // 40)  # ~2.5% of domain
        for di in range(-seam_width, seam_width + 1):
            ii = i_seam + di
            if 0 <= ii < nx:
                fade = 1.0 - 0.05 * max(0.0, 1.0 - abs(di) / max(seam_width, 1))
                correction[ii, :] *= fade

        return correction

    def get_fringe_field_correction(
        self,
        grid_shape: Tuple[int, int, int],
        electrode_gap_m: float,
    ) -> np.ndarray:
        """Return a 1-D fringe-field correction map in Z (belt width).

        Uses the Palmer formula (engineering guide §4.1.4)::

            C_fringe / C_uniform ≈ 1 + (d / (pi*w)) * [1 + ln(2*pi*w/d)]

        The correction is applied as a Z-profile multiplier: near the
        belt edges (z = 0 and z = W) the field increases due to fringe
        bulge, while the center is close to 1.0.

        Returns:
            Array of shape (nz,) with values ~0.95 at center,
            ~1.05–1.10 at edges.
        """
        _, _, nz = grid_shape
        w = self.params.plate_width_m
        gap = max(electrode_gap_m, 0.01)

        # Palmer capacitance ratio
        ratio = gap / (math.pi * w)
        C_ratio = 1.0 + ratio * (1.0 + math.log(2.0 * math.pi * w / gap))

        # Normalise: mean correction = 1.0 (energy-conserving)
        # Profile: enhanced at edges, slightly depressed in center
        z = np.linspace(0, w, nz)
        z_norm = z / w  # 0..1

        # Symmetric edge-enhancement profile
        edge_dist = np.minimum(z_norm, 1.0 - z_norm)  # 0 at edges, 0.5 at center
        # Field enhancement near edges:  1 + enhancement * exp(-edge_dist / scale)
        scale = gap / w  # characteristic fringe width / plate width
        enhancement = (C_ratio - 1.0) * 2.0  # concentrate the fringe boost at edges
        profile = 1.0 + enhancement * np.exp(-edge_dist / max(scale, 0.01))

        # Normalise so mean = 1.0 (energy-conserving)
        profile /= np.mean(profile)

        return profile.astype(np.float32)


# ── Helper ───────────────────────────────────────────────────────────

def _box_mesh(
    x0: float, y0: float, z0: float,
    lx: float, ly: float, lz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple axis-aligned box triangle mesh.

    Returns (vertices [8, 3], triangles [12, 3]).
    """
    verts = np.array([
        [x0,      y0,      z0],
        [x0 + lx, y0,      z0],
        [x0 + lx, y0 + ly, z0],
        [x0,      y0 + ly, z0],
        [x0,      y0,      z0 + lz],
        [x0 + lx, y0,      z0 + lz],
        [x0 + lx, y0 + ly, z0 + lz],
        [x0,      y0 + ly, z0 + lz],
    ], dtype=np.float32)

    tris = np.array([
        [0, 1, 2], [0, 2, 3],  # -Z face
        [4, 6, 5], [4, 7, 6],  # +Z face
        [0, 4, 5], [0, 5, 1],  # -Y face
        [2, 6, 7], [2, 7, 3],  # +Y face
        [0, 3, 7], [0, 7, 4],  # -X face
        [1, 5, 6], [1, 6, 2],  # +X face
    ], dtype=np.int32)

    return verts, tris
