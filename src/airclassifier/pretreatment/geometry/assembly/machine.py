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
        "color": "#C0392B",
        "opacity": 0.85,
        "label": "Upper Electrode",
    },
    "lower_electrode": {
        "color": "#7B2D8E",
        "opacity": 0.80,
        "label": "Lower Electrode",
    },
    "material_bed": {
        "color": "#DAA520",
        "opacity": 0.75,
        "label": "Material Bed",
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
        bed_depth_m: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Generate the material bed (product on belt inside RF zone).

        The bed is a flat slab of material sitting on the belt stack
        inside the RF zone.  It spans the full RF zone length and belt
        width.

        Args:
            bed_depth_m: Override depth (m).  Uses ``params.bed_depth_m``
                         if *None*.
        """
        from ..mesh_utils import box_mesh as _box

        depth = bed_depth_m if bed_depth_m is not None else self.params.bed_depth_m
        assert self.params.oven_params is not None

        op = self.params.oven_params
        cp = self.params.conveyor_params
        y_base = cp.belt_stack_thickness_m  # top of belt stack

        verts, tris = _box(
            op.rf_zone_x_start,            # x: RF zone start
            y_base,                         # y: on top of belt stack
            op.conveyor_belt_z0_m,          # z: belt left edge
            op.rf_zone_length_m,            # dx: RF zone length
            depth,                          # dy: bed depth
            op.rf_zone_width_m,             # dz: belt width
        )
        return verts, tris, {
            "type": "material_bed",
            "x_start": op.rf_zone_x_start,
            "x_end": op.rf_zone_x_end,
            "depth_m": depth,
        }

    # ─── Mesh generation (all at once) ────────────────────────────

    def generate_all_meshes(
        self,
        include_bed: bool = True,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, dict]]:
        """Generate all component meshes in the correct order.

        The order matters: rollers must be generated before the belt
        so the belt path can reference the roller layout.

        Args:
            include_bed: Whether to include the material bed mesh.

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

        # 7. Material bed (on belt, inside RF zone)
        if include_bed:
            meshes["material_bed"] = self.generate_material_bed_mesh()

        # 8. Infeed hopper (before oven, feeds belt)
        meshes["infeed_hopper"] = self.generate_hopper_mesh()

        # 9. Feed tunnel (connects hopper zone to oven infeed wall)
        meshes["infeed_tunnel"] = self.generate_infeed_tunnel_mesh()

        # 10. EMU housing (heater/extraction, at oven infeed end)
        meshes["emu_housing"] = self.generate_emu_mesh()

        # 11. RF Generator cabinet (behind oven, +Z side)
        meshes["generator"] = self.generate_generator_mesh()

        # 12. RF feed connection (generator → oven → electrode)
        meshes["rf_feed"] = self.generate_rf_feed_mesh()

        return meshes

    def generate_combined_mesh(
        self,
        include_bed: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Concatenate all component meshes into a single mesh.

        Useful for bounding-box calculations or simple renders.

        Returns:
            (combined_vertices, combined_triangles, summary_metadata)
        """
        all_meshes = self.generate_all_meshes(include_bed=include_bed)

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
    include_bed: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, dict]]:
    """One-shot helper: create machine and return all meshes.

    Convenience wrapper around ``create_gp15_machine`` +
    ``generate_all_meshes`` for scripts that just want the mesh data.

    Args:
        electrode_gap_m: Electrode gap (default 200 mm).
        bed_depth_m: Material bed depth on belt (default 40 mm).
        include_bed: Whether to include the material bed mesh.

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
    return machine.generate_all_meshes(include_bed=include_bed)
