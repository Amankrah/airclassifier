"""
GP-15 Machine Geometry Builder
==============================

Assembles the complete 3D representation of the QMTI GP-15 RF dielectric
heating machine for PyVista viewport rendering by composing the geometry
components (oven, conveyor, electrode) and the machine envelope.

Follows the same pattern as the air classifier geometry: components
(oven.py, conveyor.py, electrode.py) provide parametric meshes; this
module assembles them into machine coordinates and adds envelope parts.

Machine dimensions from the engineering guide §2:
    Machine envelope: 5.5 × 2.9 × 2.2 m (L × W × H)
    Belt width: 0.8 m
    Active RF zone: 1.5 m (placeholder)
    Electrode gap range: 20–300 mm

Coordinate system (§3): Y-up
    X = conveyor direction (infeed → outfeed)
    Y = vertical (ground → ceiling)
    Z = across belt width

Assembled components:
    - Oven chamber (from geometry/oven.py)
    - Conveyor belt and material bed (from geometry/conveyor.py)
    - Upper and lower electrodes (from geometry/electrode.py)
    - Machine envelope: legs, housing, attenuation tunnels,
      infeed hopper, EMU extraction duct, control panel
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..config import MachineConfig, MaterialProperties
from .oven import OvenGeometry, OvenGeometryParams
from .conveyor import ConveyorGeometry, ConveyorParams
from .electrode import ElectrodeGeometry, ElectrodeParams


def build_gp15_machine_meshes(
    config: MachineConfig | None = None,
    material: MaterialProperties | None = None,
    electrode_gap_mm: float = 80.0,
) -> Dict[str, Dict]:
    """Build the complete GP-15 machine by assembling geometry components.

    Uses oven, conveyor, and electrode modules for the applicator;
    adds machine envelope (legs, housing, tunnels, hopper, EMU duct,
    control panel) in machine coordinates.

    Each component is a dict with:
        - ``vertices``: Nx3 float32 array
        - ``triangles``: Mx3 int32 array
        - ``color``: hex color string
        - ``opacity``: float 0-1

    Returns:
        Dict mapping component names to mesh dicts.
    """
    cfg = config or MachineConfig()
    mat = material or MaterialProperties()
    gap_m = electrode_gap_mm / 1000.0

    # Machine envelope dimensions
    L = cfg.machine_length_m        # 5.5 m (X)
    W = cfg.machine_width_m         # 2.9 m (Z)
    H = cfg.machine_height_m        # 2.2 m (Y)
    oven_L = cfg.oven_length_m      # 1.5 m
    belt_W = cfg.belt_width_m        # 0.8 m
    belt_stack = cfg.belt_stack_thickness_m
    bed_depth = mat.bed_depth_m

    # Oven and belt position in machine coordinates (engineering guide §2, §3)
    oven_x0 = (L - oven_L) / 2.0
    oven_x1 = oven_x0 + oven_L
    belt_z0 = (W - belt_W) / 2.0
    belt_z1 = belt_z0 + belt_W
    y_base = 0.85   # ~850 mm from floor (typical industrial conveyor height)

    meshes = {}

    # ── Assemble geometry components (oven, conveyor, electrode) ─────
    # All components use local coordinates; translate to machine frame.
    tx, ty, tz = oven_x0, y_base, belt_z0

    # 1. Oven chamber (from geometry/oven.py)
    oven_params = OvenGeometryParams.from_machine(cfg)
    oven_params.height = gap_m  # current gap for visualization
    oven_geom = OvenGeometry(oven_params)
    ov_verts, ov_tris, _ = oven_geom.generate_mesh()
    ov_verts, ov_tris = _translate_mesh(ov_verts, ov_tris, tx, ty, tz)
    meshes["oven"] = {
        "vertices": ov_verts, "triangles": ov_tris,
        "color": "#607080", "opacity": 0.2,
    }

    # 2. Conveyor belt and material bed (from geometry/conveyor.py)
    conv_params = ConveyorParams.from_machine(cfg)
    conv_geom = ConveyorGeometry(conv_params)
    belt_verts, belt_tris, _ = conv_geom.generate_belt_mesh()
    belt_verts, belt_tris = _translate_mesh(belt_verts, belt_tris, tx, ty, tz)
    meshes["belt"] = {
        "vertices": belt_verts, "triangles": belt_tris,
        "color": "#4A90D9", "opacity": 0.6,
    }
    bed_verts, bed_tris, _ = conv_geom.generate_bed_mesh(bed_depth)
    bed_verts, bed_tris = _translate_mesh(bed_verts, bed_tris, tx, ty, tz)
    meshes["material_bed"] = {
        "vertices": bed_verts, "triangles": bed_tris,
        "color": "#D4A76A", "opacity": 0.85,
    }

    # 3. Upper and lower electrodes (from geometry/electrode.py)
    elec_params = ElectrodeParams.from_machine(cfg)
    elec_geom = ElectrodeGeometry(elec_params)
    lower_verts, lower_tris, _ = elec_geom.generate_lower_mesh()
    lower_verts, lower_tris = _translate_mesh(lower_verts, lower_tris, tx, ty, tz)
    meshes["lower_electrode"] = {
        "vertices": lower_verts, "triangles": lower_tris,
        "color": "#A0A0A0", "opacity": 0.8,
    }
    upper_verts, upper_tris, _ = elec_geom.generate_upper_mesh(gap_m)
    upper_verts, upper_tris = _translate_mesh(upper_verts, upper_tris, tx, ty, tz)
    meshes["upper_electrode"] = {
        "vertices": upper_verts, "triangles": upper_tris,
        "color": "#C0C0C0", "opacity": 0.7,
    }

    # ── Machine envelope (legs, housing, tunnels, hopper, EMU, panel) ─

    # 4. Support legs (4 legs)
    leg_w = 0.10
    leg_positions = [
        (0.3, W * 0.2), (0.3, W * 0.8),
        (L - 0.3, W * 0.2), (L - 0.3, W * 0.8),
    ]
    legs_v, legs_t = _multi_box([
        (lx - leg_w, 0.0, lz - leg_w, leg_w * 2, y_base - 0.05, leg_w * 2)
        for lx, lz in leg_positions
    ])
    meshes["legs"] = {
        "vertices": legs_v, "triangles": legs_t,
        "color": "#555555", "opacity": 0.9,
    }

    # 5. Machine housing (outer cabinet)
    wall_t = 0.04
    v_bot, t_bot = _box(0, y_base - 0.05, 0, L, 0.05, W)
    v_top, t_top = _box(0, H - 0.05, 0, L, 0.05, W)
    v_lw, t_lw = _box(0, y_base, 0, L, H - y_base - 0.05, wall_t)
    v_rw, t_rw = _box(0, y_base, W - wall_t, L, H - y_base - 0.05, wall_t)
    housing_v, housing_t = _concat_meshes([
        (v_bot, t_bot), (v_top, t_top), (v_lw, t_lw), (v_rw, t_rw),
    ])
    meshes["housing"] = {
        "vertices": housing_v, "triangles": housing_t,
        "color": "#708090", "opacity": 0.15,
    }

    # 6. Attenuation tunnels (infeed + outfeed)
    tunnel_L = 0.6
    tunnel_H = 0.35
    tunnel_wall = 0.02
    v_in, t_in = _hollow_box(
        oven_x0 - tunnel_L, y_base, belt_z0 - 0.05,
        tunnel_L, tunnel_H, belt_W + 0.10, tunnel_wall,
    )
    v_out, t_out = _hollow_box(
        oven_x1, y_base, belt_z0 - 0.05,
        tunnel_L, tunnel_H, belt_W + 0.10, tunnel_wall,
    )
    tunnels_v, tunnels_t = _concat_meshes([(v_in, t_in), (v_out, t_out)])
    meshes["tunnels"] = {
        "vertices": tunnels_v, "triangles": tunnels_t,
        "color": "#607080", "opacity": 0.25,
    }

    # 7. Infeed hopper + sizing plate
    hopper_w = 0.4
    hopper_d = belt_W * 0.6
    hopper_h = 0.5
    hx = oven_x0 - tunnel_L - 0.2
    hy = y_base + 0.3
    hz = (W - hopper_d) / 2.0
    v_hop, t_hop = _box(hx, hy, hz, hopper_w, hopper_h, hopper_d)
    v_sp, t_sp = _box(hx, hy - 0.01, hz, hopper_w, 0.01, hopper_d)
    hopper_v, hopper_t = _concat_meshes([(v_hop, t_hop), (v_sp, t_sp)])
    meshes["infeed_hopper"] = {
        "vertices": hopper_v, "triangles": hopper_t,
        "color": "#888888", "opacity": 0.5,
    }

    # 8. EMU extraction duct (on top of housing)
    duct_d = 0.25
    duct_h = 0.4
    duct_x = L * 0.5
    duct_z = W * 0.5
    v_emu, t_emu = _box(
        duct_x - duct_d / 2, H - 0.05, duct_z - duct_d / 2,
        duct_d, duct_h, duct_d,
    )
    meshes["emu_duct"] = {
        "vertices": v_emu, "triangles": t_emu,
        "color": "#606060", "opacity": 0.5,
    }

    # 9. Control panel (side-mounted)
    panel_w = 0.6
    panel_h = 0.8
    panel_d = 0.15
    v_cp, t_cp = _box(L - 0.8, y_base + 0.3, W + 0.01, panel_w, panel_h, panel_d)
    meshes["control_panel"] = {
        "vertices": v_cp, "triangles": t_cp,
        "color": "#404050", "opacity": 0.7,
    }

    return meshes


def _translate_mesh(
    verts: np.ndarray, tris: np.ndarray, tx: float, ty: float, tz: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Translate vertices by (tx, ty, tz). Triangles unchanged."""
    offset = np.array([tx, ty, tz], dtype=np.float32)
    return verts + offset, np.array(tris, dtype=np.int32, copy=True)


# ── Envelope mesh helpers ─────────────────────────────────────────────

def _box(
    x0: float, y0: float, z0: float,
    lx: float, ly: float, lz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box → (8 verts, 12 triangles)."""
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
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
    ], dtype=np.int32)
    return verts, tris


def _hollow_box(
    x0: float, y0: float, z0: float,
    lx: float, ly: float, lz: float,
    wall: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hollow box (4 walls, open ends along X) → tunnel shape."""
    parts = [
        _box(x0, y0, z0, lx, wall, lz),
        _box(x0, y0 + ly - wall, z0, lx, wall, lz),
        _box(x0, y0, z0, lx, ly, wall),
        _box(x0, y0, z0 + lz - wall, lx, ly, wall),
    ]
    return _concat_meshes(parts)


def _multi_box(boxes: list) -> Tuple[np.ndarray, np.ndarray]:
    """Build multiple boxes and concatenate."""
    return _concat_meshes([_box(*b) for b in boxes])


def _concat_meshes(
    parts: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate multiple (verts, tris) pairs into one mesh."""
    if not parts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
    all_v = []
    all_t = []
    offset = 0
    for v, t in parts:
        all_v.append(v)
        all_t.append(t + offset)
        offset += len(v)
    return np.vstack(all_v), np.vstack(all_t)
