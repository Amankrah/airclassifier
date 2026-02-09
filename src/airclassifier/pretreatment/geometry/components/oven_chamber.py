"""
Oven Chamber Geometry
=====================

GP-15 oven / applicator: the rectangular chamber that sits on the
conveyor frame and houses the electrode system, belt section, and
material bed.

Physical reality (from Manual §Oven, Engineering Guide §2.2):

    The oven is a sheet-metal enclosure bolted to the top of the
    conveyor frame.  The belt passes through it via infeed and outfeed
    openings connected to attenuation ducts (RF shielding tunnels).
    Inside, the lower electrode trays sit on the deck plate, the belt
    rides over them, product sits on the belt, and the upper electrode
    hangs above on lead screws.

    Side view (X–Y):

        ┌──────────────────────────────────────────┐ ← extraction port
        │              OVEN CHAMBER                │
        │                                          │
        │   ┌──────────────────────────────────┐   │
        │   │     Upper electrode (movable)    │   │ lead screws
        │   └──────────────────────────────────┘   │
        │              air gap                     │
        │   ════════════════════════════════════   │ ← product bed
        │   ────────────────────────────────────   │ ← belt
        │   ┌──────────────────────────────────┐   │
        │   │    Lower electrode trays         │   │
        │   └──────────────────────────────────┘   │
      ──┤                                          ├── duct openings
        └──────────────────────────────────────────┘
        ========= conveyor frame deck plate =========

Coordinate system (Y-up, matches conveyor_belt.py):
    - y = 0 is the lower electrode surface / deck plate top
    - Belt carrying surface is at y ≈ belt_stack_thickness (3.5 mm)
    - Oven walls extend from y = -wall_thickness to y = oven_height
    - X and Z aligned to conveyor frame origin

Reference dimensions:
    Machine envelope: 5.5 × 2.9 × 2.2 m  (Manual)
    Active RF zone:   ~1.5 m (placeholder, TBD — MEASURE)
    Belt width:       800 mm (Manual)
    Electrode gap:    20–300 mm (placeholder, TBD — MEASURE)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..mesh_utils import box_mesh, concat_meshes

if TYPE_CHECKING:
    from airclassifier.geometry.connection_ports import ConnectionPort
    from ...config import MachineConfig


@dataclass
class OvenChamberParams:
    """GP-15 oven chamber geometry parameters.

    The oven is positioned on the conveyor frame.  Y = 0 is the lower
    electrode / deck-plate surface.  The oven origin in X is at
    ``oven_x_start`` along the conveyor.
    """

    # ── Active RF zone (between electrodes) ──────────────────────
    rf_zone_length_m: float = 1.80      # [m] active zone (increased)
    rf_zone_width_m: float = 0.80       # [m] = belt width (800 mm)
    electrode_gap_max_m: float = 0.300  # [m] max gap [TBD — MEASURE]

    # ── Oven chamber envelope ────────────────────────────────────
    # The oven is BIGGER than the RF zone — it includes clearance
    # for the lead screws, tuning structure, wiring, and doors.
    oven_length_m: float = 2.50         # [m] total chamber length (X)
    oven_width_m: float = 1.10          # [m] total chamber width (Z) = frame width
    oven_height_m: float = 1.10         # [m] floor-to-ceiling inside
    # RF zone is centred inside the oven chamber
    # Remaining space is for lead screw clearance, wiring, ducts

    # ── Wall construction ────────────────────────────────────────
    wall_thickness_m: float = 0.004     # [m] sheet metal (galv. steel)
    wall_flange_m: float = 0.020        # [m] stiffener flanges

    # ── Doors (heavy oven doors — Manual warning) ────────────────
    door_count: int = 2                 # one per Z side
    door_length_m: float = 1.40         # [m] along X
    door_height_m: float = 0.60         # [m] door panel height
    door_y_offset_m: float = 0.04       # [m] bottom of door above deck

    # ── Infeed / outfeed openings (connect to attenuation tunnels) ─
    opening_height_m: float = 0.258     # [m] matches feed tunnel height (25.8 cm)
    opening_width_m: float = 0.84       # [m] slightly > belt (820 mm)

    # ── EMU extraction port (top centre) ─────────────────────────
    extraction_diameter_m: float = 0.25 # [m] 250 mm (Manual)

    # ── Lead screw brackets (4 posts inside oven) ────────────────
    lead_screw_section_m: float = 0.030 # [m] square bracket section
    lead_screw_inset_x_m: float = 0.08  # [m] from RF-zone X edges
    lead_screw_inset_z_m: float = 0.04  # [m] from RF-zone Z edges

    # ── Positioning on conveyor frame ────────────────────────────
    # oven_x_start: distance from conveyor origin to oven front wall
    # The oven is roughly centred on the 5.5 m conveyor frame.
    oven_x_start_m: float = 1.65        # [m] → oven ends at 1.65+2.20 = 3.85 m

    # Belt Z position on conveyor (from conveyor_belt.py)
    conveyor_belt_z0_m: float = 0.15    # [m] = (frame_width - belt_width) / 2

    # ── Simulation ───────────────────────────────────────────────
    resolution: int = 32

    @classmethod
    def from_conveyor(cls, conv_params) -> "OvenChamberParams":
        """Create oven params aligned to conveyor params."""
        return cls(
            rf_zone_width_m=conv_params.belt_width_m,
            oven_width_m=conv_params.frame_width_m,
            conveyor_belt_z0_m=conv_params.belt_z0,
        )

    @property
    def rf_zone_x_start(self) -> float:
        """X position of RF zone start (inside oven)."""
        return self.oven_x_start_m + (self.oven_length_m - self.rf_zone_length_m) / 2

    @property
    def rf_zone_x_end(self) -> float:
        """X position of RF zone end."""
        return self.rf_zone_x_start + self.rf_zone_length_m

    @property
    def oven_x_end_m(self) -> float:
        return self.oven_x_start_m + self.oven_length_m

    @property
    def belt_z_center(self) -> float:
        return self.conveyor_belt_z0_m + self.rf_zone_width_m / 2


class OvenChamberGeometry:
    """Generates the GP-15 oven chamber mesh.

    Mesh components:
      1. Side walls (Z = 0 and Z = W_oven) with door cutouts
      2. Top panel with extraction port region
      3. End walls with infeed/outfeed openings
      4. Bottom flanges (frame attachment, not a full floor — belt
         passes through)
      5. Lead screw brackets (4 vertical posts)
      6. Stiffener angles at corners
    """

    def __init__(self, params: Optional[OvenChamberParams] = None):
        self.params = params or OvenChamberParams()
        self._vertices: Optional[np.ndarray] = None
        self._triangles: Optional[np.ndarray] = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the complete oven chamber wall mesh.

        Returns:
            (vertices, triangles, metadata)
        """
        if self._vertices is not None:
            return self._vertices, self._triangles, self._get_meta()

        p = self.params
        wt = p.wall_thickness_m
        fl = p.wall_flange_m

        # Oven position in conveyor coordinates
        x0 = p.oven_x_start_m
        x1 = x0 + p.oven_length_m
        z0 = 0.0                            # oven left wall = frame left edge
        z1 = p.oven_width_m                 # oven right wall = frame right edge
        y0 = 0.0                            # deck plate level
        y1 = p.oven_height_m                # oven ceiling
        oL = p.oven_length_m
        oW = p.oven_width_m
        oH = p.oven_height_m

        # Opening geometry
        oh = p.opening_height_m
        ow = p.opening_width_m
        oz_start = (oW - ow) / 2           # centred in Z

        parts = []

        # ── 1. Left side wall (z = z0) ───────────────────────────
        # Solid panel with door cutout
        dx_door = p.door_length_m
        dy_door = p.door_height_m
        door_x0 = x0 + (oL - dx_door) / 2
        door_y0 = y0 + p.door_y_offset_m

        # Below door
        parts.append(box_mesh(x0, y0, z0, oL, p.door_y_offset_m, wt))
        # Above door
        parts.append(box_mesh(x0, door_y0 + dy_door, z0,
                              oL, oH - door_y0 - dy_door, wt))
        # Left of door
        parts.append(box_mesh(x0, door_y0, z0,
                              door_x0 - x0, dy_door, wt))
        # Right of door
        parts.append(box_mesh(door_x0 + dx_door, door_y0, z0,
                              x1 - door_x0 - dx_door, dy_door, wt))

        # ── 2. Right side wall (z = z1 - wt) — mirror of left ───
        rz = z1 - wt
        parts.append(box_mesh(x0, y0, rz, oL, p.door_y_offset_m, wt))
        parts.append(box_mesh(x0, door_y0 + dy_door, rz,
                              oL, oH - door_y0 - dy_door, wt))
        parts.append(box_mesh(x0, door_y0, rz,
                              door_x0 - x0, dy_door, wt))
        parts.append(box_mesh(door_x0 + dx_door, door_y0, rz,
                              x1 - door_x0 - dx_door, dy_door, wt))

        # ── 3. Top panel ─────────────────────────────────────────
        parts.append(box_mesh(x0, y1, z0, oL, wt, oW))

        # ── 4. End walls with openings ───────────────────────────
        # Infeed wall (x = x0)
        # Above opening
        parts.append(box_mesh(x0 - wt, y0 + oh, z0, wt, oH - oh, oW))
        # Below opening — only sides
        parts.append(box_mesh(x0 - wt, y0, z0, wt, oh, oz_start))
        parts.append(box_mesh(x0 - wt, y0, oz_start + ow,
                              wt, oh, oW - oz_start - ow))

        # Outfeed wall (x = x1)
        parts.append(box_mesh(x1, y0 + oh, z0, wt, oH - oh, oW))
        parts.append(box_mesh(x1, y0, z0, wt, oh, oz_start))
        parts.append(box_mesh(x1, y0, oz_start + ow,
                              wt, oh, oW - oz_start - ow))

        # ── 5. Bottom flanges (L-angle at floor, both Z sides) ──
        parts.append(box_mesh(x0, y0 - wt, z0, oL, wt, fl))
        parts.append(box_mesh(x0, y0 - wt, z1 - fl, oL, wt, fl))

        # ── 6. Lead screw brackets ───────────────────────────────
        ls = p.lead_screw_section_m
        rf_x0 = p.rf_zone_x_start
        rf_x1 = p.rf_zone_x_end
        bz0 = p.conveyor_belt_z0_m
        bz1 = bz0 + p.rf_zone_width_m
        inx = p.lead_screw_inset_x_m
        inz = p.lead_screw_inset_z_m

        for lx in [rf_x0 + inx, rf_x1 - inx - ls]:
            for lz in [bz0 + inz, bz1 - inz - ls]:
                parts.append(box_mesh(lx, y0, lz, ls, oH, ls))

        # ── 7. Corner stiffener angles ───────────────────────────
        sa = fl
        corners = [(x0, z0), (x0, z1 - sa), (x1 - sa, z0), (x1 - sa, z1 - sa)]
        for cx, cz in corners:
            parts.append(box_mesh(cx, y0, cz, sa, oH, sa))

        # ── 8. Door frame trim (visual, recessed line) ───────────
        trim = 0.003
        # Left door frame
        parts.append(box_mesh(door_x0, door_y0, z0 - trim,
                              dx_door, trim, trim))       # bottom
        parts.append(box_mesh(door_x0, door_y0 + dy_door - trim, z0 - trim,
                              dx_door, trim, trim))       # top
        parts.append(box_mesh(door_x0, door_y0, z0 - trim,
                              trim, dy_door, trim))       # left
        parts.append(box_mesh(door_x0 + dx_door - trim, door_y0, z0 - trim,
                              trim, dy_door, trim))       # right
        # Right door frame (mirror on Z)
        parts.append(box_mesh(door_x0, door_y0, z1,
                              dx_door, trim, trim))
        parts.append(box_mesh(door_x0, door_y0 + dy_door - trim, z1,
                              dx_door, trim, trim))
        parts.append(box_mesh(door_x0, door_y0, z1,
                              trim, dy_door, trim))
        parts.append(box_mesh(door_x0 + dx_door - trim, door_y0, z1,
                              trim, dy_door, trim))

        self._vertices, self._triangles = concat_meshes(parts)
        return self._vertices, self._triangles, self._get_meta()

    def _get_meta(self) -> dict:
        p = self.params
        return {
            "type": "oven_chamber",
            "oven_length_m": p.oven_length_m,
            "oven_width_m": p.oven_width_m,
            "oven_height_m": p.oven_height_m,
            "rf_zone_length_m": p.rf_zone_length_m,
            "rf_zone_x_start": p.rf_zone_x_start,
            "rf_zone_x_end": p.rf_zone_x_end,
            "oven_x_start": p.oven_x_start_m,
            "oven_x_end": p.oven_x_end_m,
            "electrode_gap_max_m": p.electrode_gap_max_m,
        }

    # ── Simulation grid helpers ──────────────────────────────────

    def get_grid_shape(self) -> Tuple[int, int, int]:
        """Return (nx, ny, nz) for simulation grid over RF zone."""
        p = self.params
        r = p.resolution
        aspect_xz = p.rf_zone_length_m / p.rf_zone_width_m
        aspect_yz = p.electrode_gap_max_m / p.rf_zone_width_m
        nz = r
        nx = max(4, int(r * aspect_xz))
        ny = max(4, int(r * aspect_yz))
        return (nx, ny, nz)

    def get_cell_sizes(self) -> Tuple[float, float, float]:
        """Return (dx, dy, dz) cell sizes in metres."""
        nx, ny, nz = self.get_grid_shape()
        p = self.params
        return (
            p.rf_zone_length_m / nx,
            p.electrode_gap_max_m / ny,
            p.rf_zone_width_m / nz,
        )

    def build_material_mask(
        self,
        electrode_gap_m: float,
        bed_depth_m: float,
        belt_stack_m: float = 0.0035,
    ) -> np.ndarray:
        """Build 3-D zone mask for the simulation volume.

        Zone IDs:
            0 — air gap (above material, below upper electrode)
            1 — material bed (process zone)
            2 — belt / wear-strip / top-sheet stack

        Vertical layout (Y-axis) inside the electrode gap::

            y = gap       upper electrode (V = V_rf)
            y = d_bed     top of material
            y = d_belt    top of belt stack  (≈ 3.5 mm)
            y = 0         lower electrode (ground)

        Returns:
            np.ndarray shape (nx, ny, nz), dtype int32.
        """
        nx, ny, nz = self.get_grid_shape()
        dy = electrode_gap_m / ny
        mask = np.zeros((nx, ny, nz), dtype=np.int32)
        for j in range(ny):
            yc = (j + 0.5) * dy
            if yc < belt_stack_m:
                mask[:, j, :] = 2
            elif yc < belt_stack_m + bed_depth_m:
                mask[:, j, :] = 1
        return mask

    # ── Connection ports ─────────────────────────────────────────

    @property
    def ports(self) -> Dict[str, "ConnectionPort"]:
        """Ports for ducts and EMU.

        - inlet:      infeed duct connection
        - outlet:     outfeed duct connection
        - extraction: EMU exhaust at top centre
        """
        from airclassifier.geometry.connection_ports import ConnectionPort, PortType
        p = self.params
        z_mid = p.oven_width_m / 2
        x0 = p.oven_x_start_m
        return {
            'inlet': ConnectionPort(
                position=(x0, p.opening_height_m / 2, z_mid),
                direction=(-1.0, 0.0, 0.0),
                width=p.opening_width_m,
                height=p.opening_height_m,
                port_type=PortType.RECTANGULAR,
                name="oven_inlet",
            ),
            'outlet': ConnectionPort(
                position=(p.oven_x_end_m, p.opening_height_m / 2, z_mid),
                direction=(1.0, 0.0, 0.0),
                width=p.opening_width_m,
                height=p.opening_height_m,
                port_type=PortType.RECTANGULAR,
                name="oven_outlet",
            ),
            'extraction': ConnectionPort(
                position=(x0 + p.oven_length_m / 2, p.oven_height_m, z_mid),
                direction=(0.0, 1.0, 0.0),
                diameter=p.extraction_diameter_m,
                port_type=PortType.CIRCULAR,
                name="oven_extraction",
            ),
        }


# Backward compatibility
OvenGeometryParams = OvenChamberParams
OvenGeometry = OvenChamberGeometry