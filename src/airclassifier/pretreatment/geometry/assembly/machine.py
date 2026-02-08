"""
GP-15 Machine Assembly
======================

Complete GP-15 RF dielectric heating machine assembly.

Combines all components with proper port-based alignment:
- Generator (RF oscillator cabinet)
- Oven chamber with electrodes
- Conveyor belt with material bed
- Infeed/outfeed attenuation tunnels
- Infeed hopper with sizing plate
- EMU (extraction duct, heater banks)
- HMI control panel
- Outer housing and support legs

Follows the Air Classifier Designer's assembly pattern with:
- ``build_mesh()`` for combined mesh
- ``get_component_meshes()`` for individual meshes with colors
- ``to_legacy_format()`` for backward compatibility
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...config import MachineConfig, MaterialProperties
from ..components import (
    GeneratorGeometry, GeneratorParams,
    OvenChamberGeometry, OvenChamberParams,
    ConveyorBeltGeometry, ConveyorBeltParams,
    ElectrodeGeometry, ElectrodeParams,
    TunnelGeometry, TunnelParams,
    InfeedHopperGeometry, InfeedHopperParams,
    EMUGeometry, EMUParams,
    HMIPanelGeometry, HMIPanelParams,
    HousingGeometry, HousingParams,
    SupportLegsGeometry, SupportLegsParams,
)
from ..mesh_utils import concat_meshes, translate_mesh


# Component colors for visualization
COMPONENT_COLORS = {
    'oven': ('#607080', 0.2),
    'belt': ('#4A90D9', 0.6),
    'material_bed': ('#D4A76A', 0.85),
    'lower_electrode': ('#A0A0A0', 0.8),
    'upper_electrode': ('#C0C0C0', 0.7),
    'legs': ('#555555', 0.9),
    'housing': ('#708090', 0.15),
    'infeed_tunnel': ('#607080', 0.25),
    'outfeed_tunnel': ('#607080', 0.25),
    'infeed_hopper': ('#888888', 0.5),
    'emu': ('#606060', 0.5),
    'control_panel': ('#404050', 0.7),
    'generator': ('#505050', 0.6),
}


@dataclass
class GP15MachineParams:
    """Complete GP-15 machine geometry parameters."""

    # Machine envelope
    machine_length_m: float = 5.5
    machine_width_m: float = 2.9
    machine_height_m: float = 2.2

    # Oven parameters
    oven_length_m: float = 1.5
    belt_width_m: float = 0.8

    # Heights
    conveyor_height_m: float = 0.85  # Belt surface from floor

    # Current electrode gap (for visualization)
    electrode_gap_m: float = 0.08

    # Material bed depth
    bed_depth_m: float = 0.05

    # Extraction duct
    extraction_duct_diameter_m: float = 0.25

    # Resolution
    resolution: int = 32

    @classmethod
    def from_machine(
        cls,
        config: MachineConfig,
        material: Optional[MaterialProperties] = None,
        electrode_gap_mm: float = 80.0,
    ) -> "GP15MachineParams":
        """Create params from MachineConfig and MaterialProperties."""
        mat = material or MaterialProperties()
        return cls(
            machine_length_m=config.machine_length_m,
            machine_width_m=config.machine_width_m,
            machine_height_m=config.machine_height_m,
            oven_length_m=config.oven_length_m,
            belt_width_m=config.belt_width_m,
            electrode_gap_m=electrode_gap_mm / 1000.0,
            bed_depth_m=mat.bed_depth_m,
            extraction_duct_diameter_m=config.extraction_duct_diameter_m,
        )


class GP15MachineAssembly:
    """
    Complete GP-15 RF dielectric heating machine assembly.

    Combines all components with proper positioning based on
    the machine layout shown in the engineering guide and
    machine photographs.

    Layout (Y-up, X = conveyor direction):

        Generator ─── Infeed Tunnel ─── Oven ─── Outfeed Tunnel
                            │           │
                         Hopper    EMU (on top)
                                        │
                                   Electrodes
                                        │
                                     Belt
                                        │
                                   HMI Panel (side)
                                        │
                               Housing + Legs

    Example::

        assembly = GP15MachineAssembly.from_config(config, material)
        meshes = assembly.get_component_meshes()
        combined = assembly.build_mesh()
    """

    def __init__(
        self,
        params: Optional[GP15MachineParams] = None,
        config: Optional[MachineConfig] = None,
        material: Optional[MaterialProperties] = None,
        device: str = "cpu",
    ):
        """Initialize GP-15 machine assembly.

        Args:
            params: GP15MachineParams (overrides config/material if provided)
            config: MachineConfig for component creation
            material: MaterialProperties for bed depth
            device: Device for mesh operations
        """
        if params is None:
            config = config or MachineConfig()
            material = material or MaterialProperties()
            params = GP15MachineParams.from_machine(config, material)

        self.params = params
        self.config = config or MachineConfig()
        self.material = material or MaterialProperties()
        self.device = device

        # Component instances
        self.generator: Optional[GeneratorGeometry] = None
        self.oven: Optional[OvenChamberGeometry] = None
        self.conveyor: Optional[ConveyorBeltGeometry] = None
        self.electrodes: Optional[ElectrodeGeometry] = None
        self.infeed_tunnel: Optional[TunnelGeometry] = None
        self.outfeed_tunnel: Optional[TunnelGeometry] = None
        self.hopper: Optional[InfeedHopperGeometry] = None
        self.emu: Optional[EMUGeometry] = None
        self.hmi: Optional[HMIPanelGeometry] = None
        self.housing: Optional[HousingGeometry] = None
        self.legs: Optional[SupportLegsGeometry] = None

        # Component positions (world coordinates)
        self._positions: Dict[str, Tuple[float, float, float]] = {}

        # Mesh cache
        self._combined_vertices: Optional[np.ndarray] = None
        self._combined_indices: Optional[np.ndarray] = None
        self._mesh_built: bool = False

        # Create and position components
        self._create_components()
        self._calculate_positions()

    @classmethod
    def from_config(
        cls,
        config: MachineConfig,
        material: Optional[MaterialProperties] = None,
        electrode_gap_mm: float = 80.0,
        device: str = "cpu",
    ) -> "GP15MachineAssembly":
        """Create assembly from MachineConfig.

        Args:
            config: Machine configuration
            material: Material properties
            electrode_gap_mm: Current electrode gap for visualization
            device: Compute device

        Returns:
            GP15MachineAssembly instance
        """
        params = GP15MachineParams.from_machine(config, material, electrode_gap_mm)
        return cls(params, config, material, device)

    def _create_components(self):
        """Create all machine components.

        Layout reference (from GP-15 machine images):
        - Generator: Large cabinet at BACK of machine (high Z side)
        - Housing: Main oven enclosure, FRONT of machine (low Z side)
        - Tunnels: Extend beyond housing at infeed/outfeed
        - Conveyor: Runs full length through tunnels
        - HMI: On side of generator cabinet
        - EMU: Extraction on top of oven housing
        """
        p = self.params

        # Key layout dimensions
        generator_width = 1.2   # Z dimension of generator
        generator_depth = 0.8   # X dimension
        generator_height = 1.9  # Y dimension (floor to top)
        housing_width = p.machine_width_m - generator_width  # Front portion
        tunnel_length = 0.6

        # Oven chamber (inside housing)
        self.oven = OvenChamberGeometry(OvenChamberParams(
            length=p.oven_length_m,
            width=p.belt_width_m,
            height=p.electrode_gap_m,
            extraction_port_diameter=p.extraction_duct_diameter_m,
        ))

        # Conveyor belt - extends through tunnels
        conveyor_length = p.oven_length_m + 2 * tunnel_length + 0.4  # Extra for visibility
        self.conveyor = ConveyorBeltGeometry(ConveyorBeltParams(
            belt_width_m=p.belt_width_m,
            belt_length_m=conveyor_length,
        ))

        # Electrodes (inside oven)
        self.electrodes = ElectrodeGeometry(ElectrodeParams(
            plate_width_m=p.belt_width_m,
            plate_length_m=p.oven_length_m / 2,
        ))

        # Tunnels (attenuation tunnels extending from housing)
        self.infeed_tunnel = TunnelGeometry(TunnelParams(
            tunnel_type="infeed",
            length=tunnel_length,
            width=p.belt_width_m + 0.1,
            height=0.35,
        ))
        self.outfeed_tunnel = TunnelGeometry(TunnelParams(
            tunnel_type="outfeed",
            length=tunnel_length,
            width=p.belt_width_m + 0.1,
            height=0.35,
        ))

        # Generator - LARGE cabinet at back (high Z)
        self.generator = GeneratorGeometry(GeneratorParams(
            width=generator_width,     # Z: full width at back
            height=generator_height,   # Y: floor to near ceiling
            depth=generator_depth,     # X: depth into machine
        ))

        # Hopper (before infeed tunnel)
        self.hopper = InfeedHopperGeometry(InfeedHopperParams(
            depth=p.belt_width_m * 0.6,
            sizing_plate_gap=p.bed_depth_m,
        ))

        # EMU (on top of housing over oven)
        self.emu = EMUGeometry(EMUParams(
            duct_diameter=p.extraction_duct_diameter_m,
        ))

        # HMI Panel (on generator cabinet)
        self.hmi = HMIPanelGeometry(HMIPanelParams(
            width=0.5,
            height=0.6,
            depth=0.12,
        ))

        # Housing - main oven enclosure (FRONT portion, not full width)
        self.housing = HousingGeometry(HousingParams(
            length=p.oven_length_m + 0.4,  # Slightly larger than oven
            width=housing_width,            # Front portion only
            height=p.machine_height_m - 0.2,
            base_height=p.conveyor_height_m,
        ))

        # Support legs
        self.legs = SupportLegsGeometry(SupportLegsParams(
            machine_length=p.machine_length_m,
            machine_width=p.machine_width_m,
            leg_height=p.conveyor_height_m,
        ))

    def _calculate_positions(self):
        """Calculate world positions for all components.

        Machine coordinate system (Y-up, matching reference images):
        - X: Conveyor direction (infeed → outfeed)
        - Y: Vertical (floor → ceiling)
        - Z: Across belt width (front → back)

        Layout from reference image:
        - Generator at BACK (high Z), spanning most of X
        - Housing/Oven at FRONT (low Z)
        - Tunnels extend beyond housing on infeed/outfeed sides
        - Conveyor runs full length
        """
        p = self.params

        # Key dimensions
        L = p.machine_length_m       # 5.5m total X
        W = p.machine_width_m        # 2.9m total Z
        H = p.machine_height_m       # 2.2m total Y
        oven_L = p.oven_length_m     # 1.5m oven length
        belt_W = p.belt_width_m      # 0.8m belt width
        y_base = p.conveyor_height_m # 0.85m conveyor height

        # Generator dimensions (large cabinet at back)
        gen_width = 1.2    # Z
        gen_depth = 0.8    # X
        gen_height = 1.9   # Y

        # Housing dimensions (front portion)
        housing_width = W - gen_width  # Z: front portion
        housing_length = oven_L + 0.4  # X: slightly larger than oven

        # Tunnel length
        tunnel_L = 0.6

        # Calculate key X positions
        # Tunnels extend beyond housing, oven centered
        housing_x0 = (L - housing_length) / 2.0
        oven_x0 = (L - oven_L) / 2.0
        infeed_tunnel_x0 = housing_x0 - tunnel_L
        outfeed_tunnel_x1 = housing_x0 + housing_length

        # Calculate key Z positions (belt centered in front portion)
        housing_z0 = 0.0  # Housing at front
        belt_z0 = (housing_width - belt_W) / 2.0
        gen_z0 = W - gen_width  # Generator at back

        # Store positions
        self._positions = {
            # Core processing components (inside housing)
            'oven': (oven_x0, y_base, belt_z0),
            'belt': (infeed_tunnel_x0 - 0.2, y_base, belt_z0),  # Conveyor extends through tunnels
            'material_bed': (oven_x0, y_base, belt_z0),  # Material only in oven
            'lower_electrode': (oven_x0, y_base, belt_z0),
            'upper_electrode': (oven_x0, y_base, belt_z0),

            # Tunnels (extend beyond housing)
            'infeed_tunnel': (infeed_tunnel_x0, y_base, belt_z0 - 0.05),
            'outfeed_tunnel': (outfeed_tunnel_x1, y_base, belt_z0 - 0.05),

            # Generator (BACK of machine, floor level)
            'generator': ((L - gen_depth) / 2.0, 0.0, gen_z0),

            # Hopper (before infeed tunnel, above belt)
            'infeed_hopper': (infeed_tunnel_x0 - 0.6, y_base + 0.2, belt_z0 + belt_W * 0.2),

            # EMU (on top of housing, centered over oven)
            'emu': (oven_x0 + oven_L / 2, H - 0.3, belt_z0 + belt_W / 2),

            # HMI panel (on generator cabinet side, facing front)
            'control_panel': ((L - gen_depth) / 2.0 + gen_depth + 0.02, y_base, gen_z0 + 0.3),

            # Housing (FRONT portion of machine)
            'housing': (housing_x0, 0.0, housing_z0),

            # Legs (at machine corners)
            'legs': (0.0, 0.0, 0.0),
        }

    def build_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build combined mesh for all components.

        Returns:
            Tuple of (vertices, indices) for complete machine
        """
        if self._mesh_built:
            return self._combined_vertices, self._combined_indices

        parts = []

        for name, (component, position) in self._get_component_list():
            verts, tris, _ = component.generate_mesh()
            verts, tris = translate_mesh(verts, tris, *position)
            parts.append((verts, tris))

        self._combined_vertices, self._combined_indices = concat_meshes(parts)
        self._mesh_built = True

        return self._combined_vertices, self._combined_indices

    def get_component_meshes(self) -> Dict[str, Dict[str, Any]]:
        """Get individual component meshes with colors.

        Returns:
            Dict mapping component names to mesh dicts:
            {
                'oven': {'vertices': ..., 'triangles': ..., 'color': '#...', 'opacity': ...},
                'belt': {...},
                ...
            }
        """
        meshes = {}
        p = self.params

        for name, (component, position) in self._get_component_list():
            # Get mesh with special handling for electrode gap
            if name == 'upper_electrode':
                verts, tris, _ = self.electrodes.generate_upper_mesh(p.electrode_gap_m)
            elif name == 'material_bed':
                verts, tris, _ = self.conveyor.generate_bed_mesh(p.bed_depth_m)
            else:
                verts, tris, _ = component.generate_mesh()

            # Translate to world position
            verts, tris = translate_mesh(verts, tris, *position)

            # Get color
            color, opacity = COMPONENT_COLORS.get(name, ('#808080', 0.5))

            meshes[name] = {
                'vertices': verts,
                'triangles': tris,
                'color': color,
                'opacity': opacity,
            }

        return meshes

    def to_legacy_format(self) -> Dict[str, Dict[str, Any]]:
        """Return mesh dict compatible with original build_gp15_machine_meshes().

        Maintains backward compatibility with existing visualization code.
        """
        meshes = self.get_component_meshes()

        # Combine tunnels into single entry for legacy format
        if 'infeed_tunnel' in meshes and 'outfeed_tunnel' in meshes:
            infeed = meshes.pop('infeed_tunnel')
            outfeed = meshes.pop('outfeed_tunnel')

            # Combine tunnel meshes
            combined_verts = np.vstack([infeed['vertices'], outfeed['vertices']])
            combined_tris = np.vstack([
                infeed['triangles'],
                outfeed['triangles'] + len(infeed['vertices'])
            ])

            meshes['tunnels'] = {
                'vertices': combined_verts.astype(np.float32),
                'triangles': combined_tris.astype(np.int32),
                'color': infeed['color'],
                'opacity': infeed['opacity'],
            }

        # Rename for legacy compatibility
        if 'emu' in meshes:
            meshes['emu_duct'] = meshes.pop('emu')

        return meshes

    def _get_component_list(self) -> List[Tuple[str, Tuple[Any, Tuple[float, float, float]]]]:
        """Return list of (name, (component, position)) tuples."""
        p = self.params

        components = [
            ('oven', (self.oven, self._positions['oven'])),
            ('belt', (self.conveyor, self._positions['belt'])),
            ('material_bed', (self.conveyor, self._positions['material_bed'])),
            ('lower_electrode', (self.electrodes, self._positions['lower_electrode'])),
            ('upper_electrode', (self.electrodes, self._positions['upper_electrode'])),
            ('infeed_tunnel', (self.infeed_tunnel, self._positions['infeed_tunnel'])),
            ('outfeed_tunnel', (self.outfeed_tunnel, self._positions['outfeed_tunnel'])),
            ('generator', (self.generator, self._positions['generator'])),
            ('infeed_hopper', (self.hopper, self._positions['infeed_hopper'])),
            ('emu', (self.emu, self._positions['emu'])),
            ('control_panel', (self.hmi, self._positions['control_panel'])),
            ('housing', (self.housing, self._positions['housing'])),
            ('legs', (self.legs, self._positions['legs'])),
        ]

        # Filter out None components
        return [(name, data) for name, data in components if data[0] is not None]

    def get_oven_geometry(self) -> OvenChamberGeometry:
        """Access oven for simulation (backward compat)."""
        return self.oven

    def get_electrode_geometry(self) -> ElectrodeGeometry:
        """Access electrodes for field corrections."""
        return self.electrodes

    def get_conveyor_geometry(self) -> ConveyorBeltGeometry:
        """Access conveyor for advection."""
        return self.conveyor

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box."""
        p = self.params
        return (
            np.array([0, 0, 0]),
            np.array([p.machine_length_m, p.machine_height_m, p.machine_width_m])
        )

    def print_summary(self):
        """Print assembly summary."""
        p = self.params
        print("=" * 60)
        print("GP-15 MACHINE ASSEMBLY")
        print("=" * 60)
        print(f"Machine envelope: {p.machine_length_m:.1f} x {p.machine_width_m:.1f} x {p.machine_height_m:.1f} m")
        print(f"Oven length: {p.oven_length_m:.1f} m")
        print(f"Belt width: {p.belt_width_m:.1f} m")
        print(f"Electrode gap: {p.electrode_gap_m * 1000:.0f} mm")
        print(f"Bed depth: {p.bed_depth_m * 1000:.0f} mm")
        print()
        print("Components:")
        for name, (comp, pos) in self._get_component_list():
            print(f"  - {name}: at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        print("=" * 60)


def create_gp15_machine(
    config: Optional[MachineConfig] = None,
    material: Optional[MaterialProperties] = None,
    electrode_gap_mm: float = 80.0,
    device: str = "cpu",
) -> GP15MachineAssembly:
    """Create a standard GP-15 machine assembly.

    Factory function for creating GP15MachineAssembly with standard
    configuration.

    Args:
        config: Machine configuration
        material: Material properties
        electrode_gap_mm: Current electrode gap for visualization
        device: Compute device

    Returns:
        GP15MachineAssembly instance
    """
    config = config or MachineConfig()
    material = material or MaterialProperties()
    params = GP15MachineParams.from_machine(config, material, electrode_gap_mm)
    return GP15MachineAssembly(params, config, material, device)


def build_gp15_machine_meshes(
    config: Optional[MachineConfig] = None,
    material: Optional[MaterialProperties] = None,
    electrode_gap_mm: float = 80.0,
) -> Dict[str, Dict[str, Any]]:
    """Build GP-15 machine meshes (backward compatible).

    DEPRECATED: Use GP15MachineAssembly or create_gp15_machine() instead.

    This function maintains backward compatibility with existing code
    that uses the original build_gp15_machine_meshes() function.

    Args:
        config: Machine configuration
        material: Material properties
        electrode_gap_mm: Current electrode gap [mm]

    Returns:
        Dict mapping component names to mesh dicts
    """
    assembly = create_gp15_machine(config, material, electrode_gap_mm)
    return assembly.to_legacy_format()
