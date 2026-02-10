"""
GP-15 Machine Assembly
======================

Assembles the three core GP-15 geometry components into a single
coherent machine:

  1. **Conveyor belt** — structural frame, rollers, belt loop, deck plate
  2. **Oven chamber**  — sheet-metal enclosure that sits on the conveyor
  3. **Electrodes**    — upper (movable) and lower (fixed) electrode
                         assemblies inside the oven

Coordinate alignment (Engineering Guide §3):

    The conveyor provides the world coordinate frame.  Y-up.
    y = 0 is the lower electrode / deck-plate surface.
    The oven is centred on the conveyor at ``oven_x_start`` (default 1.65 m).
    Electrodes span the RF zone inside the oven.

Parameter chain:
    ConveyorBeltParams
        → OvenChamberParams.from_conveyor(conv_params)
            → ElectrodeParams.from_oven(oven_params)

This ensures belt width, frame width, and Z positions propagate
correctly through the entire assembly.

Usage::

    from airclassifier.pretreatment.geometry.assembly import (
        create_gp15_machine,
        build_gp15_machine_meshes,
    )

    machine = create_gp15_machine(electrode_gap_m=0.200, bed_depth_m=0.04)
    meshes = machine.generate_all_meshes()

    # Each entry: (vertices, triangles, metadata)
    for name, (v, t, meta) in meshes.items():
        print(f"{name}: {v.shape[0]} verts, {t.shape[0]} tris")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..components.conveyor_belt import ConveyorBeltGeometry, ConveyorBeltParams
from ..components.electrode import ElectrodeGeometry, ElectrodeParams
from ..components.emu import EMUGeometry, EMUParams
from ..components.generator import GeneratorGeometry, GeneratorParams
from ..components.hopper import InfeedHopperGeometry, InfeedHopperParams
from ..components.oven_chamber import OvenChamberGeometry, OvenChamberParams
from ..mesh_utils import concat_meshes


# ─────────────────────────────────────────────────────────────────────
#  Visual styling
# ─────────────────────────────────────────────────────────────────────

COMPONENT_COLORS: Dict[str, Dict[str, object]] = {
    "conveyor_frame": {
        "color": "#505060",
        "opacity": 0.22,
        "label": "Conveyor Frame",
    },
    "rollers": {
        "color": "#909090",
        "opacity": 0.85,
        "label": "Rollers",
    },
    "belt": {
        "color": "#4169E1",
        "opacity": 0.88,
        "label": "Belt (PTFE)",
    },
    "oven_chamber": {
        "color": "#8B6914",
        "opacity": 0.18,
        "label": "Oven Chamber",
    },
    "upper_electrode": {
        "color": "#A8A8B0",
        "opacity": 0.40,
        "label": "Upper Electrode",
    },
    "lower_electrode": {
        "color": "#909098",
        "opacity": 0.35,
        "label": "Lower Electrode",
    },
    "material_bed": {
        "color": "#DAA520",
        "opacity": 0.75,
        "label": "Material Bed",
    },
    "outfeed_tunnel": {
        "color": "#808088",
        "opacity": 0.55,
        "label": "Outfeed Tunnel",
    },
    "collection_bin": {
        "color": "#606068",
        "opacity": 0.70,
        "label": "Collection Bin",
    },
    "infeed_hopper": {
        "color": "#A0A0A8",
        "opacity": 0.82,
        "label": "Infeed Hopper",
    },
    "infeed_tunnel": {
        "color": "#808088",
        "opacity": 0.55,
        "label": "Feed Tunnel",
    },
    "emu_housing": {
        "color": "#B0B0B8",
        "opacity": 0.35,
        "label": "EMU Housing",
    },
    "generator": {
        "color": "#707880",
        "opacity": 0.45,
        "label": "RF Generator",
    },
    "rf_feed": {
        "color": "#CD7F32",
        "opacity": 0.90,
        "label": "RF Feed (copper)",
    },
}


# ─────────────────────────────────────────────────────────────────────
#  Assembly-level parameters
# ─────────────────────────────────────────────────────────────────────

@dataclass
class GP15MachineParams:
    """Top-level GP-15 machine assembly parameters.

    These control both the physical configuration and the parameter
    chain that flows from conveyor → oven → electrodes.

    Attributes:
        conveyor_params: Conveyor belt / frame geometry.
        oven_params: Oven chamber geometry (derived from conveyor if None).
        electrode_params: Electrode geometry (derived from oven if None).
        electrode_gap_m: Current electrode gap [m] (20–300 mm).
        bed_depth_m: Material bed depth on belt [m].
    """

    conveyor_params: ConveyorBeltParams = field(default_factory=ConveyorBeltParams)
    oven_params: Optional[OvenChamberParams] = None
    electrode_params: Optional[ElectrodeParams] = None
    hopper_params: Optional[InfeedHopperParams] = None
    emu_params: Optional[EMUParams] = None
    generator_params: Optional[GeneratorParams] = None

    # ── Adjustable operating parameters ───────────────────────────
    electrode_gap_m: float = 0.200       # [m] default 200 mm
    bed_depth_m: float = 0.040           # [m] default 40 mm

    def __post_init__(self) -> None:
        """Derive all component params from the conveyor → oven chain."""
        if self.oven_params is None:
            self.oven_params = OvenChamberParams.from_conveyor(self.conveyor_params)

        if self.electrode_params is None:
            self.electrode_params = ElectrodeParams.from_oven(self.oven_params)

        if self.hopper_params is None:
            self.hopper_params = InfeedHopperParams.from_oven(self.oven_params)

        if self.emu_params is None:
            self.emu_params = EMUParams.from_oven(self.oven_params)

        if self.generator_params is None:
            self.generator_params = GeneratorParams.from_oven(self.oven_params)

    # ── Validation ────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Check parameter consistency.  Returns list of warnings."""
        warnings: List[str] = []
        belt_stack = self.conveyor_params.belt_stack_thickness_m

        if self.bed_depth_m + belt_stack > self.electrode_gap_m:
            warnings.append(
                f"Bed ({self.bed_depth_m * 1000:.0f} mm) + belt stack "
                f"({belt_stack * 1000:.1f} mm) exceeds electrode gap "
                f"({self.electrode_gap_m * 1000:.0f} mm)."
            )

        if self.electrode_gap_m < 0.020:
            warnings.append(
                f"Electrode gap {self.electrode_gap_m * 1000:.0f} mm is below "
                f"the minimum 20 mm."
            )

        if self.electrode_gap_m > 0.300:
            warnings.append(
                f"Electrode gap {self.electrode_gap_m * 1000:.0f} mm exceeds "
                f"the maximum 300 mm."
            )

        return warnings


# ─────────────────────────────────────────────────────────────────────
#  Machine assembly
# ─────────────────────────────────────────────────────────────────────

class GP15MachineAssembly:
    """Complete GP-15 RF dielectric heating machine assembly.

    Assembles the conveyor belt, oven chamber, and electrode system
    into a single machine with consistent coordinate alignment.

    The conveyor is the base reference.  The oven sits on the conveyor
    frame.  The electrodes sit inside the oven in the RF zone.

    Typical usage::

        machine = GP15MachineAssembly()
        meshes = machine.generate_all_meshes()
        for name, (verts, tris, meta) in meshes.items():
            ...
    """

    def __init__(self, params: Optional[GP15MachineParams] = None) -> None:
        self.params = params or GP15MachineParams()

        # ── Instantiate component geometries ──────────────────────
        self._conveyor = ConveyorBeltGeometry(self.params.conveyor_params)
        self._oven = OvenChamberGeometry(self.params.oven_params)
        self._electrodes = ElectrodeGeometry(self.params.electrode_params)
        self._hopper = InfeedHopperGeometry(self.params.hopper_params)
        self._emu = EMUGeometry(self.params.emu_params)
        self._generator = GeneratorGeometry(self.params.generator_params)

        # ── Cached combined mesh ──────────────────────────────────
        self._combined_verts: Optional[np.ndarray] = None
        self._combined_tris: Optional[np.ndarray] = None

    # ─── Component accessors ──────────────────────────────────────

    @property
    def conveyor(self) -> ConveyorBeltGeometry:
        """Access the conveyor belt geometry component."""
        return self._conveyor

    @property
    def oven(self) -> OvenChamberGeometry:
        """Access the oven chamber geometry component."""
        return self._oven

    @property
    def electrodes(self) -> ElectrodeGeometry:
        """Access the electrode geometry component."""
        return self._electrodes

    @property
    def hopper(self) -> InfeedHopperGeometry:
        """Access the infeed hopper geometry component."""
        return self._hopper

    @property
    def emu(self) -> EMUGeometry:
        """Access the EMU housing geometry (infeed end)."""
        return self._emu

    @property
    def generator(self) -> GeneratorGeometry:
        """Access the RF generator cabinet geometry (behind oven)."""
        return self._generator

    # ─── Mesh generation (individual) ─────────────────────────────

    def generate_conveyor_frame_mesh(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the conveyor structural frame mesh."""
        return self._conveyor.generate_bed_structure_mesh()

    def generate_rollers_mesh(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the roller system (head, tail, tension, etc.)."""
        return self._conveyor.generate_wheels_mesh()

    def generate_belt_mesh(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the continuous belt loop.

        Note: generate_rollers_mesh() must be called first (or will be
        called internally) so the roller layout is available for the
        belt path computation.
        """
        return self._conveyor.generate_belt_mesh()

    def generate_oven_mesh(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the oven chamber walls, doors, and openings."""
        return self._oven.generate_mesh()

    def generate_upper_electrode_mesh(
        self,
        electrode_gap_m: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the upper electrode assembly at the current gap.

        Args:
            electrode_gap_m: Override gap (m).  Uses ``params.electrode_gap_m``
                             if *None*.
        """
        gap = electrode_gap_m if electrode_gap_m is not None else self.params.electrode_gap_m
        return self._electrodes.generate_upper_mesh(gap)

    def generate_lower_electrode_mesh(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the lower electrode trays, PET supports, chokes."""
        return self._electrodes.generate_lower_mesh()

    def generate_emu_mesh(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the EMU housing at oven infeed end."""
        return self._emu.generate_mesh()

    def generate_generator_mesh(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the RF generator cabinet behind the oven."""
        return self._generator.generate_mesh()

    def generate_rf_feed_mesh(
        self,
        electrode_gap_m: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the RF feed connection (generator → electrode).

        Copper busbars, tuning plates, and feed strip extensions
        running from the generator through the oven back wall and
        down to the upper electrode assembly.
        """
        gap = electrode_gap_m if electrode_gap_m is not None else self.params.electrode_gap_m
        return self._generator.generate_rf_feed_mesh(gap)

    def generate_hopper_mesh(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the infeed hopper bin."""
        return self._hopper.generate_hopper_mesh()

    def generate_infeed_tunnel_mesh(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the feed tunnel fitted to oven infeed wall."""
        return self._hopper.generate_tunnel_mesh()

    def generate_material_bed_mesh(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the material bed along the full belt carrying run.

        The belt moves in +X, carrying material from the hopper at the
        infeed end all the way to the head roller at the outfeed end,
        where it drops into the collection bin.

        From the manual (Illustration 3 / sizing plate design):
        the material is deposited from the hopper with a wedge profile
        that starts tall at the hopper discharge and tapers to the
        controlled bed depth set by the sizing gate.  Through the oven
        and the remaining belt the bed is at uniform depth.

        Side view::

            HOPPER                                      HEAD
              ╲                                         ROLLER
               ╲ slant (sizing gate)                      │
                ╲________________________________________╲│ → drops
                |               |              |       ╲ │   into
                | oven RF zone  | outfeed belt | → +X   ↓     bin
                |_______________|______________|___________
                ← slant →←── oven ──→←── post-oven ──→

        The bed spans from hopper discharge through the oven and along
        the remaining belt to the head roller (outfeed).
        """
        from ..mesh_utils import box_mesh as _box

        assert self.params.oven_params is not None
        assert self.params.hopper_params is not None

        op = self.params.oven_params
        hp = self.params.hopper_params
        cp = self.params.conveyor_params

        y_base = cp.belt_stack_thickness_m     # top of belt stack
        bed_depth = self.params.bed_depth_m    # uniform bed depth
        z0 = op.conveyor_belt_z0_m
        belt_w = op.rf_zone_width_m

        # ── Key X positions along the material flow (+X) ──────────
        x_hopper = hp.hopper_front_x           # where material lands
        x_oven_in = op.oven_x_start_m          # oven infeed wall
        x_head = cp.frame_length_m - cp.nose_length_m  # head roller

        slant_len = x_oven_in - x_hopper
        slant_top_height = bed_depth * 2.5     # pile height at hopper

        parts_v: List[np.ndarray] = []
        parts_t: List[np.ndarray] = []

        # ── 1. Slant section: hopper discharge → oven infeed ──────
        # Material piles up at the hopper and the sizing gate sets
        # the max height.  It tapers down to the uniform bed depth
        # over the distance from hopper to oven entry.
        if slant_len > 0.01:
            # Trapezoidal wedge: tall at hopper, tapers to bed_depth at oven
            slant_verts = np.array([
                # Left (z0)
                [x_hopper, y_base, z0],                                # 0
                [x_hopper, y_base + slant_top_height, z0],             # 1
                [x_oven_in, y_base + bed_depth, z0],                   # 2
                [x_oven_in, y_base, z0],                               # 3
                # Right (z0 + belt_w)
                [x_hopper, y_base, z0 + belt_w],                      # 4
                [x_hopper, y_base + slant_top_height, z0 + belt_w],   # 5
                [x_oven_in, y_base + bed_depth, z0 + belt_w],         # 6
                [x_oven_in, y_base, z0 + belt_w],                     # 7
            ], dtype=np.float32)
            slant_tris = np.array([
                # Left face
                [0, 1, 2], [0, 2, 3],
                # Right face
                [4, 6, 5], [4, 7, 6],
                # Bottom
                [0, 3, 7], [0, 7, 4],
                # Front (hopper face)
                [0, 4, 5], [0, 5, 1],
                # Back (oven face)
                [3, 2, 6], [3, 6, 7],
                # Top slope
                [1, 5, 6], [1, 6, 2],
            ], dtype=np.int32)
            parts_v.append(slant_verts)
            parts_t.append(slant_tris)

        # ── 2. Uniform section: oven infeed → head roller ─────────
        # The belt carries the material at uniform depth through
        # the oven (RF processing zone), out the outfeed tunnel,
        # and along the remaining belt to the head roller where
        # it drops into the collection bin.
        uniform_x0 = op.oven_x_start_m
        uniform_x1 = x_head                   # extends to head roller
        uniform_len = uniform_x1 - uniform_x0
        if uniform_len > 0.01:
            uv, ut = _box(
                uniform_x0, y_base, z0,
                uniform_len, bed_depth, belt_w,
            )
            # Offset triangle indices
            offset = sum(v.shape[0] for v in parts_v)
            parts_v.append(uv)
            parts_t.append(ut + offset)

        all_verts = np.vstack(parts_v).astype(np.float32)
        all_tris = np.vstack(parts_t).astype(np.int32)

        return all_verts, all_tris, {
            "type": "material_bed",
            "profile": "slant_plus_uniform",
            "slant_x_start": x_hopper,
            "uniform_x_start": op.oven_x_start_m,
            "uniform_x_end": float(x_head),
            "bed_depth_m": bed_depth,
            "flow_direction": "+X (hopper → head roller → collection bin)",
        }

    def generate_outfeed_tunnel_mesh(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the outfeed attenuation tunnel at the oven exit wall.

        Proportional to the infeed tunnel: same height, same length,
        same construction.  Sits at the oven outfeed wall (x = oven_x_end).
        """
        from ..mesh_utils import box_mesh as _box

        assert self.params.oven_params is not None
        assert self.params.hopper_params is not None

        op = self.params.oven_params
        hp = self.params.hopper_params

        # Same dimensions as infeed tunnel
        tL = hp.tunnel_length_m               # 0.248 m
        tH = hp.tunnel_height_m               # 0.258 m
        tw = hp.tunnel_wall_thickness_m        # 0.003 m

        # Position: starts at oven outfeed wall, extends in +X
        x0 = op.oven_x_end_m
        y0 = 0.0                               # deck level
        z_center = op.belt_z_center
        tunnel_z_width = hp.hopper_width_m + 0.04  # same as infeed tunnel
        tz0 = z_center - tunnel_z_width / 2.0

        parts = []
        # Bottom
        parts.append(_box(x0, y0 - tw, tz0, tL, tw, tunnel_z_width))
        # Top
        parts.append(_box(x0, y0 + tH, tz0, tL, tw, tunnel_z_width))
        # Left wall
        parts.append(_box(x0, y0, tz0, tL, tH, tw))
        # Right wall
        parts.append(_box(x0, y0, tz0 + tunnel_z_width - tw, tL, tH, tw))
        # End flange (outer end)
        fl = 0.015
        parts.append(_box(x0 + tL, y0 + tH, tz0 - fl, fl, tw, tunnel_z_width + 2 * fl))
        parts.append(_box(x0 + tL, y0 - tw, tz0 - fl, fl, tw, tunnel_z_width + 2 * fl))
        parts.append(_box(x0 + tL, y0, tz0 - fl, fl, tH, tw + fl))
        parts.append(_box(x0 + tL, y0, tz0 + tunnel_z_width, fl, tH, tw + fl))

        verts, tris = concat_meshes(parts)
        return verts, tris, {
            "type": "outfeed_tunnel",
            "x_start": x0,
            "x_end": x0 + tL,
            "tunnel_length_m": tL,
            "tunnel_height_m": tH,
        }

    def generate_collection_bin_mesh(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the collection bin at the outfeed end of the belt.

        A proportionate open-top stainless steel container that sits
        directly on the floor beneath and past the head roller.  Product
        rolls off the belt end and drops into the bin.  No legs — it
        rests flat on the floor like any real industrial collection bin.

        The bin's back wall extends slightly under the conveyor bed
        frame (into the open-bottom zone where the lower horizontal
        frame members have been shortened).  The back wall is taller
        than the other three walls to catch material falling backward
        off the head roller.

        Side view (X-Y)::

            upper frame ──────┐   belt →→→ ↓ falls off roller
                              │        │
                        ┌─────┤  ┌─────┼──────┐  ← back wall (taller)
                        │under│  │     ↓      │
                        │ bed │  │  product    │  ← front/side wall rim
                        │     │  │             │
                        │     │  │             │
                        └─────┴──┴─────────────┘  ← floor (no legs)
        """
        from ..mesh_utils import box_mesh as _box

        cp = self.params.conveyor_params

        L = cp.frame_length_m
        nose = cp.nose_length_m
        H = cp.frame_height_m
        lh = cp.leg_height_m
        W = cp.frame_width_m
        belt_w = cp.belt_width_m
        wt = 0.003  # 3 mm sheet metal

        # Floor level
        floor_y = -(H + lh)

        # Head roller position (where material falls off)
        head_x = L - nose

        # ── Bin dimensions (proportionate to machine) ─────────────
        # Most of the bin sits past the head roller (X+ direction).
        # A small portion extends under the bed to catch material
        # as it drops off the roller.
        bin_under_bed = 0.15                     # 15 cm extends under bed
        bin_past_end = 0.40                      # 40 cm past head roller
        bin_depth_x = bin_under_bed + bin_past_end   # ~55 cm total

        bin_x0 = head_x - bin_under_bed          # back wall (oven side)
        bin_x1 = head_x + bin_past_end           # front wall (past belt)

        bin_width_z = belt_w + 0.06              # slightly wider than belt
        bin_z0 = (W - bin_width_z) / 2           # centred on frame

        # Height: practical floor-standing bin (~60% of floor-to-deck)
        bin_height = abs(floor_y) * 0.60
        bin_bottom_y = floor_y                   # flat on the floor
        bin_top_y = floor_y + bin_height

        # Back wall is taller to catch falling material
        back_extra = 0.15                        # 15 cm above other walls

        parts = []

        # ── Walls ─────────────────────────────────────────────────
        # Back wall (toward oven, partially under bed) — taller
        parts.append(_box(bin_x0, bin_bottom_y, bin_z0,
                          wt, bin_height + back_extra, bin_width_z))
        # Front wall (past belt end)
        parts.append(_box(bin_x1 - wt, bin_bottom_y, bin_z0,
                          wt, bin_height, bin_width_z))
        # Left wall
        parts.append(_box(bin_x0, bin_bottom_y, bin_z0,
                          bin_depth_x, bin_height, wt))
        # Right wall
        parts.append(_box(bin_x0, bin_bottom_y, bin_z0 + bin_width_z - wt,
                          bin_depth_x, bin_height, wt))
        # Bottom (flat on the floor)
        parts.append(_box(bin_x0, bin_bottom_y, bin_z0,
                          bin_depth_x, wt, bin_width_z))

        # ── Top rim flange (folded edge for rigidity) ─────────────
        # Rim around the three shorter walls (back wall is taller)
        rim = 0.015
        # Front rim
        parts.append(_box(bin_x1 - wt - rim, bin_top_y, bin_z0 - rim,
                          rim + wt, rim, bin_width_z + 2 * rim))
        # Left rim
        parts.append(_box(bin_x0, bin_top_y, bin_z0 - rim,
                          bin_depth_x, rim, rim + wt))
        # Right rim
        parts.append(_box(bin_x0, bin_top_y, bin_z0 + bin_width_z - wt,
                          bin_depth_x, rim, rim + wt))

        verts, tris = concat_meshes(parts)
        return verts, tris, {
            "type": "collection_bin",
            "bin_x_start": float(bin_x0),
            "bin_x_end": float(bin_x1),
            "bin_top_y": float(bin_top_y),
            "bin_bottom_y": float(bin_bottom_y),
            "bin_height_m": float(bin_height),
            "back_wall_top_y": float(bin_top_y + back_extra),
            "under_bed_m": float(bin_under_bed),
            "past_end_m": float(bin_past_end),
            "floor_standing": True,
        }

    # ─── Mesh generation (all at once) ────────────────────────────

    def generate_all_meshes(
        self,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, dict]]:
        """Generate all component meshes in the correct order.

        The order matters: rollers must be generated before the belt
        so the belt path can reference the roller layout.

        Returns:
            Ordered dict of ``{name: (vertices, triangles, metadata)}``.
        """
        meshes: Dict[str, Tuple[np.ndarray, np.ndarray, dict]] = {}

        # 1. Conveyor frame (base structure)
        meshes["conveyor_frame"] = self.generate_conveyor_frame_mesh()

        # 2. Rollers (must come before belt — belt reads roller layout)
        meshes["rollers"] = self.generate_rollers_mesh()

        # 3. Belt loop (continuous PTFE loop around rollers)
        meshes["belt"] = self.generate_belt_mesh()

        # 4. Oven chamber (sits on conveyor frame)
        meshes["oven_chamber"] = self.generate_oven_mesh()

        # 5. Upper electrode (movable, inside oven)
        meshes["upper_electrode"] = self.generate_upper_electrode_mesh()

        # 6. Lower electrode (fixed, on deck plate)
        meshes["lower_electrode"] = self.generate_lower_electrode_mesh()

        # 7. Material bed (hopper → oven → head roller → drops into bin)
        meshes["material_bed"] = self.generate_material_bed_mesh()

        # 8. Infeed hopper (before oven, feeds belt)
        meshes["infeed_hopper"] = self.generate_hopper_mesh()

        # 9. Feed tunnel (connects hopper zone to oven infeed wall)
        meshes["infeed_tunnel"] = self.generate_infeed_tunnel_mesh()

        # 10. Outfeed tunnel (oven exit wall, proportional to infeed)
        meshes["outfeed_tunnel"] = self.generate_outfeed_tunnel_mesh()

        # 11. Collection bin (outfeed end, catches product off head roller)
        meshes["collection_bin"] = self.generate_collection_bin_mesh()

        # 12. EMU housing (heater/extraction, behind oven)
        meshes["emu_housing"] = self.generate_emu_mesh()

        # 13. RF Generator cabinet (behind oven, +Z side)
        meshes["generator"] = self.generate_generator_mesh()

        # 14. RF feed connection (generator → oven → electrode)
        meshes["rf_feed"] = self.generate_rf_feed_mesh()

        return meshes

    def generate_combined_mesh(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Concatenate all component meshes into a single mesh.

        Useful for bounding-box calculations or simple renders.

        Returns:
            (combined_vertices, combined_triangles, summary_metadata)
        """
        all_meshes = self.generate_all_meshes()

        parts = [(v, t) for v, t, _meta in all_meshes.values()]
        verts, tris = concat_meshes(parts)

        meta = {
            "type": "gp15_machine_assembly",
            "component_count": len(all_meshes),
            "total_vertices": int(verts.shape[0]),
            "total_triangles": int(tris.shape[0]),
            "electrode_gap_m": self.params.electrode_gap_m,
            "bed_depth_m": self.params.bed_depth_m,
            "components": list(all_meshes.keys()),
        }
        return verts, tris, meta

    # ─── Derived information ──────────────────────────────────────

    def get_assembly_info(self) -> dict:
        """Return a summary of the machine assembly configuration."""
        p = self.params
        cp = p.conveyor_params
        op = p.oven_params
        ep = p.electrode_params

        assert op is not None
        assert ep is not None

        return {
            "machine": "GP-15 RF Dielectric Heating Machine",
            "frame_length_m": cp.frame_length_m,
            "frame_width_m": cp.frame_width_m,
            "belt_width_m": cp.belt_width_m,
            "belt_stack_thickness_m": cp.belt_stack_thickness_m,
            "oven_length_m": op.oven_length_m,
            "oven_width_m": op.oven_width_m,
            "oven_height_m": op.oven_height_m,
            "oven_x_start_m": op.oven_x_start_m,
            "oven_x_end_m": op.oven_x_end_m,
            "rf_zone_length_m": op.rf_zone_length_m,
            "rf_zone_x_start_m": op.rf_zone_x_start,
            "rf_zone_x_end_m": op.rf_zone_x_end,
            "electrode_gap_m": p.electrode_gap_m,
            "bed_depth_m": p.bed_depth_m,
            "air_gap_m": max(
                0.0,
                p.electrode_gap_m - p.bed_depth_m - cp.belt_stack_thickness_m,
            ),
        }

    def set_electrode_gap(self, gap_m: float) -> None:
        """Update the electrode gap and invalidate upper electrode cache.

        This allows animating the gap without rebuilding the entire machine.

        Args:
            gap_m: New electrode gap in metres (0.020 – 0.300).
        """
        self.params.electrode_gap_m = gap_m
        # Invalidate upper electrode cache
        self._electrodes._upper_verts = None
        self._electrodes._upper_tris = None
        # Invalidate combined cache
        self._combined_verts = None
        self._combined_tris = None

    def set_bed_depth(self, depth_m: float) -> None:
        """Update the material bed depth.

        Args:
            depth_m: New bed depth in metres.
        """
        self.params.bed_depth_m = depth_m
        self._combined_verts = None
        self._combined_tris = None


# ─────────────────────────────────────────────────────────────────────
#  Factory functions
# ─────────────────────────────────────────────────────────────────────

def create_gp15_machine(
    electrode_gap_m: float = 0.200,
    bed_depth_m: float = 0.040,
    conveyor_params: Optional[ConveyorBeltParams] = None,
    oven_params: Optional[OvenChamberParams] = None,
    electrode_params: Optional[ElectrodeParams] = None,
    hopper_params: Optional[InfeedHopperParams] = None,
    emu_params: Optional[EMUParams] = None,
) -> GP15MachineAssembly:
    """Create a standard GP-15 machine assembly.

    This is the recommended entry point.  It builds the full parameter
    chain (conveyor → oven → electrodes) and returns a ready-to-use
    assembly.

    Args:
        electrode_gap_m: Electrode gap (default 200 mm).
        bed_depth_m: Material bed depth on belt (default 40 mm).
        conveyor_params: Override conveyor parameters.
        oven_params: Override oven parameters (derived from conveyor
                     if *None*).
        electrode_params: Override electrode parameters (derived from
                         oven if *None*).

    Returns:
        Configured GP15MachineAssembly.

    Example::

        machine = create_gp15_machine(electrode_gap_m=0.150, bed_depth_m=0.06)
        meshes = machine.generate_all_meshes()
    """
    conv = conveyor_params or ConveyorBeltParams()

    params = GP15MachineParams(
        conveyor_params=conv,
        oven_params=oven_params,
        electrode_params=electrode_params,
        hopper_params=hopper_params,
        emu_params=emu_params,
        electrode_gap_m=electrode_gap_m,
        bed_depth_m=bed_depth_m,
    )

    warnings = params.validate()
    for w in warnings:
        import warnings as _warnings
        _warnings.warn(w, stacklevel=2)

    return GP15MachineAssembly(params)


def build_gp15_machine_meshes(
    electrode_gap_m: float = 0.200,
    bed_depth_m: float = 0.040,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, dict]]:
    """One-shot helper: create machine and return all meshes.

    Convenience wrapper around ``create_gp15_machine`` +
    ``generate_all_meshes`` for scripts that just want the mesh data.

    Args:
        electrode_gap_m: Electrode gap (default 200 mm).
        bed_depth_m: Material bed depth on belt (default 40 mm).

    Returns:
        Dict of ``{component_name: (vertices, triangles, metadata)}``.

    Example::

        meshes = build_gp15_machine_meshes(electrode_gap_m=0.150)
        for name, (v, t, meta) in meshes.items():
            print(f"{name}: {v.shape[0]:,} verts, {t.shape[0]:,} tris")
    """
    machine = create_gp15_machine(
        electrode_gap_m=electrode_gap_m,
        bed_depth_m=bed_depth_m,
    )
    return machine.generate_all_meshes()
