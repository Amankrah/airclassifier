"""
Classification System Assembly Module
======================================

This module provides the complete classification system assembly for protein separation.
It combines all Phase 1 components into an integrated particle separation train.

SYSTEM OVERVIEW
===============

The classification system separates particles by size using air classification:
- Venturi Eductor: Entrains particles into airstream
- Zigzag Classifier: Primary separation by particle size
- Multi-Cyclone System: Staged collection of fines
- Bag Filter: Final fine particle capture

MATERIAL FLOW PATH
==================

    AIR SUPPLY (from blower, +Y direction upward)
           │
           ▼
    ┌──────────────────┐
    │ VENTURI EDUCTOR  │◄────── Feed (from de-agglomerator)
    │   (entrainment)  │        enters via solids inlet (+X side)
    │    axis='y'      │
    └────────┬─────────┘
             │ (air + particles, +Y upward)
    ┌────────┴─────────┐
    │   ROUND DUCT     │  ← Connects venturi outlet to zigzag air inlet
    └────────┬─────────┘
             ▼
    ┌──────────────────┐
    │ ZIGZAG CLASSIFIER│  ← Air flows upward, particles separate
    │   (separation)   │
    └────┬────────┬────┘
         │        │
    (fines)    (coarse)
    +Y up      -Y down
         │        │
         │        ▼
         │   ┌─────────┐
         │   │ STARCH  │  (heavy particles fall out, collected below)
         │   │ OUTLET  │
         │   └─────────┘
         │
    ┌────┴─────────┐
    │   ELBOW      │  ← Turns flow from +Y to +X (horizontal)
    │  (90° turn)  │
    └────┬─────────┘
         │
    ┌────┴─────────┐
    │  HORIZ DUCT  │  ← Horizontal duct to cyclone inlet
    └────┬─────────┘
         ▼
    ┌──────────────────┐
    │ MULTI-CYCLONE    │  ← Tangential inlet on +X side
    │  (series stages) │
    │  Primary→Second→ │────► Dust outlets (staged product collection)
    │  →Tertiary       │
    └────────┬─────────┘
             │ (overflow from last cyclone, +Y direction)
    ┌────────┴─────────┐
    │   ELBOW          │  ← Turns flow from +Y to +X
    └────────┬─────────┘
         │
    ┌────┴─────────┐
    │  HORIZ DUCT  │  ← To bag filter inlet
    └────┬─────────┘
         ▼
    ┌──────────────────┐
    │   BAG FILTER     │  ← Dirty air inlet on +X side
    │  (fine capture)  │────► Dust outlet (protein-rich fines)
    └────────┬─────────┘
             │ (clean air, +Y direction)
             ▼
        CLEAN AIR EXHAUST (to exhaust system)
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import warp as wp

from ..connection_ports import (
    ConnectionPort, PortType, calculate_alignment,
    validate_assembly_connections, print_connection_report
)


@dataclass
class ClassificationSystemParams:
    """
    Parameters for complete classification system.

    Combines all Phase 1 components into a protein separation system.

    Industry Standard Pilot-Scale Proportions (100-500 kg/h):
    - Zigzag channel: 100-200mm width, 5-7 stages
    - Cyclones: Primary 250-400mm, Secondary 150-250mm, Tertiary 100-150mm
    - Bag filter: sized for 0.1-0.3 m³/s (360-1080 m³/h)
    - Venturi: 80-150mm inlet diameter
    """

    # Zigzag classifier parameters (pilot scale)
    zigzag_channel_width: float = 0.12      # [m] 120mm - typical pilot scale
    zigzag_num_stages: int = 5              # 5 stages for good separation
    zigzag_channel_depth: float = 0.20      # [m] 200mm depth

    # Venturi eductor parameters
    venturi_inlet_diameter: float = 0.08    # [m] 80mm - matches air supply
    venturi_throat_ratio: float = 0.5       # Throat = 40mm

    # Multi-cyclone parameters (pilot scale)
    primary_cyclone_diameter: float = 0.30   # [m] 300mm - coarse/starch
    secondary_cyclone_diameter: float = 0.20 # [m] 200mm - medium
    tertiary_cyclone_diameter: float = 0.12  # [m] 120mm - fine/protein

    # Bag filter parameters (pilot scale: ~0.15 m³/s = 540 m³/h)
    bag_filter_flow_rate: float = 0.15      # [m³/s] Pilot scale air flow
    bag_filter_air_to_cloth: float = 2.5    # [m³/min/m²] Typical for food dust

    # Layout parameters
    flange_gap: float = 0.005               # [m] Gap between flanges (5mm gasket space)
    duct_spacing: float = 0.08              # [m] Minimum duct length between components
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Mesh resolution
    resolution: int = 24


class ClassificationSystemAssembly:
    """
    Complete protein separation/classification system assembly.

    Combines all Phase 1 components with proper port-to-port connections:
    - Venturi Eductor: Particle entrainment (vertical, axis='y')
    - Zigzag Classifier: Primary separation (vertical)
    - Multi-Cyclone System: Staged collection (vertical cyclones, series arrangement)
    - Bag Filter: Fine particle collection

    Flow path connections:
    1. Venturi outlet (+Y) → Duct → Zigzag air_inlet (-Y facing)
    2. Zigzag fines_outlet (+Y) → Elbow (90°) → Duct → Cyclone inlet (tangential +X)
    3. Cyclone overflow (+Y) → Elbow (90°) → Duct → Bag filter dirty_air_inlet (+X)
    4. Bag filter clean_air_outlet (+Y) → To exhaust system

    Coordinate system:
    - Origin at venturi air inlet (bottom of system)
    - X-axis: Horizontal (right)
    - Y-axis: Vertical (up) - main flow direction through venturi/zigzag
    - Z-axis: Depth (into page)
    """

    def __init__(self, params: ClassificationSystemParams = None, device: str = "cpu"):
        """
        Initialize classification system assembly.

        Args:
            params: ClassificationSystemParams (uses defaults if None)
            device: Warp device for mesh operations
        """
        self.params = params or ClassificationSystemParams()
        self.device = device

        # Component storage
        self.venturi = None
        self.zigzag = None
        self.multi_cyclone = None
        self.bag_filter = None

        # Component positions (world coordinates of component origins)
        self._component_positions: Dict[str, np.ndarray] = {}

        # Duct sections: list of (duct_component, world_position) tuples
        self._duct_sections: List[Tuple[Any, Tuple[float, float, float]]] = []

        # Mesh data
        self._combined_vertices = None
        self._combined_indices = None
        self._mesh_built = False

        # Create components with proper geometry-based positioning
        self._create_components()

    def _create_components(self):
        """
        Create all system components with proper geometry-based positioning.

        Layout: Vertical venturi/zigzag with horizontal runs to cyclones and bag filter.

        Flow path (following port-to-port connections):
        1. Venturi air_inlet (bottom) receives air from blower
        2. Venturi outlet (+Y) → Duct → Zigzag air_inlet
        3. Zigzag fines_outlet (+Y) → Elbow → Duct → Cyclone inlet (tangential)
        4. Cyclone overflow (+Y) → Elbow → Duct → Bag filter dirty_air_inlet
        5. Bag filter clean_air_outlet (+Y) → To exhaust

        All positions calculated from port connection points.
        """
        from ..components import (
            MultiCycloneSystem, MultiCycloneParams, CycloneStageParams,
            create_standard_bag_filter,
        )
        from ..components.zigzag_classifier import ZigzagClassifierParams, ZigzagClassifier
        from ..components.venturi_eductor import VenturiEducatorParams, VenturiEducator
        from ..components.ductwork import (
            RoundDuct, RoundDuctParams,
            DuctElbow, DuctElbowParams,
            RectToRoundTransition, RectToRoundTransitionParams,
        )

        p = self.params
        gap = p.flange_gap

        # ============================================================
        # 1. VENTURI EDUCTOR - Positioned at origin, vertical (axis='y')
        # ============================================================
        # Venturi creates vertical upward flow for particle entrainment
        # air_inlet at bottom (Y=0), outlet at top (Y=total_length)
        #
        # COORDINATE SYSTEM:
        # - X+: From air filter toward deagglomerator (horizontal)
        # - Y+: Vertical (upward toward bag filter top outlet)
        # - Z+: Distance away from classification system (toward feed)
        #
        # SOLIDS INLET CONFIGURATION:
        # - Angular position: π/2 (90°) puts inlet on +Z side (facing feed system)
        # - Entry angle: 15° tilts inlet upward to receive 15-degree descending shaft
        throat_d = p.venturi_inlet_diameter * p.venturi_throat_ratio
        venturi_params = VenturiEducatorParams(
            inlet_diameter=p.venturi_inlet_diameter,
            throat_diameter=throat_d,
            outlet_diameter=p.venturi_inlet_diameter * 0.9,
            convergent_angle=np.radians(12),
            divergent_angle=np.radians(5),
            solids_inlet_diameter=throat_d * 0.8,
            solids_inlet_angle=np.radians(15),  # 15° tilt upward to match feed shaft angle
            solids_inlet_position=throat_d * 0.3,
            solids_inlet_angular_position=np.pi / 2,  # π/2 = +Z side (facing feed system)
            center=(0.0, 0.0, 0.0),  # Origin at air_inlet
            axis="y"  # Vertical axis for upward flow
        )
        self.venturi = VenturiEducator(venturi_params)

        # Venturi position: air_inlet at system origin
        self._component_positions['venturi'] = np.array([
            p.center[0], p.center[1], p.center[2]
        ])

        # Get venturi outlet port in world coordinates
        venturi_outlet = self.venturi.ports['outlet']
        venturi_outlet_world = self._get_port_world_pos('venturi', venturi_outlet)

        # ============================================================
        # 2. ZIGZAG CLASSIFIER - Above venturi
        # ============================================================
        # Zigzag receives air+particles from venturi
        # air_inlet at bottom (receives from venturi), fines_outlet at top
        zigzag_params = ZigzagClassifierParams(
            channel_width=p.zigzag_channel_width,
            channel_depth=p.zigzag_channel_depth,
            num_stages=p.zigzag_num_stages,
            stage_height=p.zigzag_channel_width * 1.5,
            zigzag_angle=np.radians(120),
            feed_stage=(p.zigzag_num_stages + 1) // 2,
            feed_width=p.zigzag_channel_width * 0.5,
            air_inlet_width=p.zigzag_channel_width,
            air_inlet_height=p.zigzag_channel_width * 0.5,
            fines_outlet_width=p.zigzag_channel_width,
            fines_outlet_height=p.zigzag_channel_width * 0.5,
            coarse_outlet_width=p.zigzag_channel_width * 0.5,
            coarse_outlet_height=p.zigzag_channel_width * 0.3,
            center=(0.0, 0.0, 0.0)  # Will offset for port alignment
        )
        self.zigzag = ZigzagClassifier(zigzag_params)

        # Get zigzag air_inlet port (in local coords relative to zigzag center)
        zigzag_inlet = self.zigzag.ports['air_inlet']

        # Get dimensions for the connection
        venturi_d = self.venturi.params.outlet_diameter
        zigzag_inlet_w = zigzag_inlet.width   # X dimension (air_inlet_width)
        zigzag_inlet_h = zigzag_inlet.height  # Z dimension (channel_depth)
        
        # Calculate proper transition length based on size change
        # Use gradual expansion angle (max 15 degrees) for smooth flow
        max_expansion_angle = np.radians(12)
        max_dim = max(zigzag_inlet_w, zigzag_inlet_h)
        min_transition_length = (max_dim - venturi_d) / (2 * np.tan(max_expansion_angle))
        min_transition_length = max(min_transition_length, 0.1)  # At least 100mm
        
        # Round duct length - proportional to venturi diameter
        duct1a_length = venturi_d * 0.5  # 50% of diameter
        
        # Total connection length
        trans1_length = min_transition_length
        total_duct1_length = gap + duct1a_length + gap + trans1_length + gap

        # Position zigzag so its air_inlet aligns with venturi outlet + duct + transition
        zigzag_y = venturi_outlet_world[1] + total_duct1_length - zigzag_inlet.position[1]
        self._component_positions['zigzag'] = np.array([
            venturi_outlet_world[0] - zigzag_inlet.position[0],  # Align X
            zigzag_y,
            venturi_outlet_world[2] - zigzag_inlet.position[2],  # Align Z
        ])

        # Create duct + transition: venturi outlet (round) to zigzag inlet (rect)
        # Round duct from venturi outlet
        duct1a_start = (
            venturi_outlet_world[0],
            venturi_outlet_world[1] + gap,
            venturi_outlet_world[2],
        )
        duct1a = RoundDuct(RoundDuctParams(
            diameter=venturi_d,
            length=duct1a_length,
            wall_thickness=0.002,
            direction=(0.0, 1.0, 0.0),  # Vertical +Y
            center=(0, 0, 0),
            flanged=True,
        ))
        self._duct_sections.append((duct1a, duct1a_start))

        # Round-to-rect transition from venturi diameter to zigzag inlet
        # For vertical +Y direction: perp1=+X, perp2=-Z
        # outlet_dimensions[0] maps to X direction (zigzag width)
        # outlet_dimensions[1] maps to Z direction (zigzag depth)
        from ..components.transitions import Transition, TransitionParams
        trans1_start = (
            duct1a_start[0],
            duct1a_start[1] + duct1a_length + gap,
            duct1a_start[2],
        )
        trans1 = Transition(TransitionParams(
            transition_type="round_to_rect",
            inlet_dimensions=(venturi_d,),
            outlet_dimensions=(zigzag_inlet_w, zigzag_inlet_h),
            length=trans1_length,
            concentric=True,
            wall_thickness=0.002,
            direction=(0.0, 1.0, 0.0),  # Vertical +Y
            center=(0, 0, 0),
            flanged=True,
        ))
        self._duct_sections.append((trans1, trans1_start))

        # Get zigzag fines_outlet in world coordinates
        zigzag_fines = self.zigzag.ports['fines_outlet']
        zigzag_fines_world = self._get_port_world_pos('zigzag', zigzag_fines)

        # ============================================================
        # 3. MULTI-CYCLONE SYSTEM - To the right (+X) of zigzag top
        # ============================================================
        # Connection: zigzag fines (+Y) → 90° elbow → horizontal duct (+X) → cyclone inlet
        # Cyclone inlet is on -X side of primary cyclone (tangential entry)
        cyclone_stages = [
            CycloneStageParams(
                name="primary",
                diameter=p.primary_cyclone_diameter,
                design_d50=40e-6,  # 40 μm - coarse/starch fraction
            ),
            CycloneStageParams(
                name="secondary",
                diameter=p.secondary_cyclone_diameter,
                design_d50=20e-6,  # 20 μm - medium fraction
            ),
            CycloneStageParams(
                name="tertiary",
                diameter=p.tertiary_cyclone_diameter,
                design_d50=10e-6,  # 10 μm - fine/protein fraction
            ),
        ]
        # Spacing between cyclones = larger diameter * 0.3 for clearance
        cyclone_spacing = p.primary_cyclone_diameter * 0.3
        cyclone_params = MultiCycloneParams(
            stages=cyclone_stages,
            arrangement="series",
            spacing=cyclone_spacing,
            center=(0.0, 0.0, 0.0),
            resolution=p.resolution,
        )
        self.multi_cyclone = MultiCycloneSystem(cyclone_params)

        # Get cyclone inlet port dimensions
        cyclone_inlet = self.multi_cyclone.ports['inlet']
        cyclone_inlet_w = cyclone_inlet.width
        cyclone_inlet_h = cyclone_inlet.height
        cyclone_inlet_area = cyclone_inlet_w * cyclone_inlet_h
        cyclone_inlet_equiv_d = np.sqrt(4 * cyclone_inlet_area / np.pi)

        # Calculate zigzag fines equivalent diameter
        zigzag_fines_area = zigzag_fines.width * zigzag_fines.height
        zigzag_fines_equiv_d = np.sqrt(4 * zigzag_fines_area / np.pi)

        # ============================================================
        # Connection: Zigzag fines (rect) → rect-to-round transition →
        #             elbow → duct → round-to-rect transition → cyclone inlet (rect)
        # ============================================================

        # Use cyclone inlet equivalent diameter for the duct
        # (flow accelerates into cyclone for better separation)
        duct2_diameter = cyclone_inlet_equiv_d
        elbow2_bend_radius = duct2_diameter * 1.5  # R/D = 1.5

        # STEP 1: Rect-to-round transition from zigzag fines to round duct
        trans2a_length = 0.15  # 150mm transition
        trans2a_start = (
            zigzag_fines_world[0],
            zigzag_fines_world[1] + gap,
            zigzag_fines_world[2],
        )
        trans2a = Transition(TransitionParams(
            transition_type="rect_to_round",
            inlet_dimensions=(zigzag_fines.width, zigzag_fines.height),
            outlet_dimensions=(duct2_diameter,),
            length=trans2a_length,
            concentric=True,
            wall_thickness=0.002,
            direction=(0.0, 1.0, 0.0),  # Vertical +Y
            center=(0, 0, 0),
        ))
        self._duct_sections.append((trans2a, trans2a_start))

        # STEP 2: 90° elbow after transition (turns from +Y to +X)
        elbow2_inlet_pos = (
            trans2a_start[0],
            trans2a_start[1] + trans2a_length + gap,
            trans2a_start[2],
        )
        
        # Create elbow with inlet at origin, will be positioned at elbow2_inlet_pos
        # Note: To turn from +Y to +X, we need rotation_axis=(0, 0, 1)
        # Right-hand rule: thumb points +Z, fingers curl from +Y to +X (90° counterclockwise viewed from +Z)
        elbow2 = DuctElbow(DuctElbowParams(
            diameter=duct2_diameter,
            bend_radius=elbow2_bend_radius,
            angle=90.0,
            wall_thickness=0.002,
            flanged=True,
            center=(0, 0, 0),
            inlet_direction=(0.0, 1.0, 0.0),   # Flow enters from +Y (vertical up)
            rotation_axis=(0.0, 0.0, 1.0),     # Bend around +Z to turn toward +X
        ))
        
        # Use elbow's calculated outlet position for accuracy
        elbow2_local_outlet = elbow2.get_outlet_position()
        elbow2_outlet_pos = (
            elbow2_inlet_pos[0] + elbow2_local_outlet[0],
            elbow2_inlet_pos[1] + elbow2_local_outlet[1],
            elbow2_inlet_pos[2] + elbow2_local_outlet[2],
        )
        self._duct_sections.append((elbow2, elbow2_inlet_pos))

        # STEP 3: Horizontal round duct
        duct2_length = p.primary_cyclone_diameter * 0.5  # Shorter duct
        duct2_start = (
            elbow2_outlet_pos[0] + gap,
            elbow2_outlet_pos[1],
            elbow2_outlet_pos[2],
        )
        duct2 = RoundDuct(RoundDuctParams(
            diameter=duct2_diameter,
            length=duct2_length,
            wall_thickness=0.002,
            direction=(1.0, 0.0, 0.0),
            center=(0, 0, 0),
            flanged=True,
        ))
        self._duct_sections.append((duct2, duct2_start))

        # STEP 4: Round-to-rect transition to cyclone inlet
        # Note: For horizontal +X direction, the transition's coordinate system maps:
        #   outlet_dimensions[0] → perp1 = -Y (vertical)
        #   outlet_dimensions[1] → perp2 = +Z (horizontal depth)
        # Cyclone inlet has: height=vertical (Y), width=tangent (Z)
        # So we pass (height, width) to match the cyclone inlet orientation
        trans2b_length = 0.10  # 100mm transition
        trans2b_start = (
            duct2_start[0] + duct2_length + gap,
            duct2_start[1],
            duct2_start[2],
        )
        trans2b = Transition(TransitionParams(
            transition_type="round_to_rect",
            inlet_dimensions=(duct2_diameter,),
            outlet_dimensions=(cyclone_inlet_h, cyclone_inlet_w),  # (height, width) to match inlet
            length=trans2b_length,
            concentric=True,
            wall_thickness=0.002,
            direction=(1.0, 0.0, 0.0),  # Horizontal +X
            center=(0, 0, 0),
        ))
        self._duct_sections.append((trans2b, trans2b_start))

        # Position cyclone so its inlet aligns with end of transition
        cyclone_inlet_target_x = trans2b_start[0] + trans2b_length + gap
        cyclone_inlet_target_y = trans2b_start[1]
        cyclone_inlet_target_z = trans2b_start[2]

        self._component_positions['multi_cyclone'] = np.array([
            cyclone_inlet_target_x - cyclone_inlet.position[0],
            cyclone_inlet_target_y - cyclone_inlet.position[1],
            cyclone_inlet_target_z - cyclone_inlet.position[2],
        ])

        # Get cyclone overflow port in world coordinates
        cyclone_overflow = self.multi_cyclone.ports['overflow']
        cyclone_overflow_world = self._get_port_world_pos('multi_cyclone', cyclone_overflow)

        # ============================================================
        # 4. BAG FILTER - To the right (+X) of cyclone system
        # ============================================================
        # Connection: cyclone overflow (+Y) → 90° elbow → horizontal duct (+X) →
        #             expansion transition → bag filter inlet
        # Bag filter is sized for pilot-scale flow rate
        self.bag_filter = create_standard_bag_filter(
            flow_rate_m3s=p.bag_filter_flow_rate,
            air_to_cloth=p.bag_filter_air_to_cloth
        )

        # Bag filter inlet port
        bag_inlet = self.bag_filter.ports['dirty_air_inlet']
        bag_inlet_diameter = bag_inlet.diameter

        # Elbow parameters for cyclone-to-bagfilter connection
        # Size elbow to match cyclone overflow
        elbow3_diameter = cyclone_overflow.diameter
        elbow3_bend_radius = elbow3_diameter * 1.5

        # Elbow inlet receives from cyclone overflow (+Y), turns to +X
        elbow3_inlet_pos = (
            cyclone_overflow_world[0],
            cyclone_overflow_world[1] + gap,
            cyclone_overflow_world[2],
        )

        # Create elbow3 now (will add to duct_sections later)
        # Note: To turn from +Y to +X, we need rotation_axis=(0, 0, 1)
        elbow3 = DuctElbow(DuctElbowParams(
            diameter=elbow3_diameter,
            bend_radius=elbow3_bend_radius,
            angle=90.0,
            wall_thickness=0.002,
            flanged=True,
            center=(0, 0, 0),
            inlet_direction=(0.0, 1.0, 0.0),   # Air comes from +Y (up from cyclone)
            rotation_axis=(0.0, 0.0, 1.0),     # Rotate around +Z to turn toward +X
        ))
        elbow3_local_outlet = elbow3.get_outlet_position()
        
        # After 90° elbow (Y→X), outlet position:
        elbow3_outlet_pos = (
            elbow3_inlet_pos[0] + elbow3_local_outlet[0],
            elbow3_inlet_pos[1] + elbow3_local_outlet[1],
            elbow3_inlet_pos[2] + elbow3_local_outlet[2],
        )

        # Calculate clearance: bag filter should be positioned so it doesn't
        # overlap with cyclones. We need to know the cyclone system extent.
        mc_min, mc_max = self.multi_cyclone.get_system_bounds()
        cyclone_max_x = self._component_positions['multi_cyclone'][0] + mc_max[0]

        # We need: elbow → short duct → expansion transition → bag filter inlet
        # Short duct from elbow maintains cyclone overflow diameter
        duct3a_length = 0.1  # 100mm short duct section

        # Expansion transition from cyclone overflow diameter to bag filter inlet diameter
        # Expansion angle should be ≤15° to avoid flow separation
        # Length = (D_out - D_in) / (2 * tan(angle))
        expansion_angle = np.radians(12)  # 12° half-angle for gradual expansion
        transition_length = (bag_inlet_diameter - elbow3_diameter) / (2 * np.tan(expansion_angle))
        transition_length = max(transition_length, 0.15)  # Minimum 150mm

        # Calculate total path length
        total_duct_length = duct3a_length + transition_length
        min_total_length = max(0.3, cyclone_max_x - elbow3_outlet_pos[0] + 0.15)
        if total_duct_length < min_total_length:
            # Extend duct3a to meet minimum length
            duct3a_length = min_total_length - transition_length

        # Position bag filter so its inlet aligns with end of transition
        bag_inlet_target_x = elbow3_outlet_pos[0] + gap + duct3a_length + gap + transition_length + gap
        bag_inlet_target_y = elbow3_outlet_pos[1]
        bag_inlet_target_z = elbow3_outlet_pos[2]

        self._component_positions['bag_filter'] = np.array([
            bag_inlet_target_x - bag_inlet.position[0],
            bag_inlet_target_y - bag_inlet.position[1],
            bag_inlet_target_z - bag_inlet.position[2],
        ])

        # Add elbow3 (created earlier) to duct sections
        self._duct_sections.append((elbow3, elbow3_inlet_pos))

        # Create short duct section from elbow (maintains cyclone diameter)
        duct3a_start = (
            elbow3_outlet_pos[0] + gap,
            elbow3_outlet_pos[1],
            elbow3_outlet_pos[2],
        )
        duct3a = RoundDuct(RoundDuctParams(
            diameter=elbow3_diameter,
            length=duct3a_length,
            wall_thickness=0.002,
            direction=(1.0, 0.0, 0.0),  # Horizontal +X
            center=(0, 0, 0),
            flanged=True,
        ))
        self._duct_sections.append((duct3a, duct3a_start))

        # Create expansion transition from cyclone diameter to bag filter diameter
        transition_start = (
            duct3a_start[0] + duct3a_length + gap,
            duct3a_start[1],
            duct3a_start[2],
        )
        # Use round-to-round Transition for concentric expansion
        from ..components.transitions import Transition, TransitionParams
        expansion = Transition(TransitionParams(
            transition_type="round_to_round",
            inlet_dimensions=(elbow3_diameter,),
            outlet_dimensions=(bag_inlet_diameter,),
            length=transition_length,
            concentric=True,
            wall_thickness=0.002,
            direction=(1.0, 0.0, 0.0),  # Horizontal +X
            center=(0, 0, 0),
        ))
        self._duct_sections.append((expansion, transition_start))

    def _get_port_world_pos(self, component_name: str, port: ConnectionPort) -> np.ndarray:
        """Helper to get port position in world coordinates."""
        comp_pos = self._component_positions[component_name]
        return np.array([
            comp_pos[0] + port.position[0],
            comp_pos[1] + port.position[1],
            comp_pos[2] + port.position[2],
        ])



    def build_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build combined mesh for all components including duct sections.

        Returns:
            Tuple of (vertices, indices)
        """
        all_vertices = []
        all_indices = []
        vertex_offset = 0

        def add_component_mesh(component, position):
            """Add component mesh with position offset."""
            nonlocal vertex_offset
            verts, idx, _ = component.generate_mesh()

            # Apply position offset
            offset = np.array(position)
            verts_offset = verts + offset

            all_vertices.append(verts_offset)
            all_indices.append(idx + vertex_offset)
            vertex_offset += len(verts)

        # Add main components
        add_component_mesh(self.venturi, self._component_positions['venturi'])
        add_component_mesh(self.zigzag, self._component_positions['zigzag'])
        add_component_mesh(self.multi_cyclone, self._component_positions['multi_cyclone'])
        add_component_mesh(self.bag_filter, self._component_positions['bag_filter'])

        # Add duct sections (each is a tuple of (duct_component, world_position))
        for duct, position in self._duct_sections:
            add_component_mesh(duct, position)

        self._combined_vertices = np.vstack(all_vertices).astype(np.float32)
        self._combined_indices = np.concatenate(all_indices).astype(np.int32)
        self._mesh_built = True

        return self._combined_vertices, self._combined_indices

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box of the entire system."""
        if not self._mesh_built:
            self.build_mesh()

        min_corner = self._combined_vertices.min(axis=0)
        max_corner = self._combined_vertices.max(axis=0)

        return min_corner, max_corner

    def get_system_extent(self) -> np.ndarray:
        """Get system extent (dimensions) in each axis."""
        min_c, max_c = self.get_bounds()
        return max_c - min_c

    def get_component(self, name: str) -> Any:
        """Get a specific component by name."""
        components = {
            'venturi': self.venturi,
            'zigzag': self.zigzag,
            'multi_cyclone': self.multi_cyclone,
            'bag_filter': self.bag_filter,
        }
        if name not in components:
            raise KeyError(f"Unknown component: {name}. Available: {list(components.keys())}")
        return components[name]

    def get_component_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """Get positions of all components."""
        return {name: tuple(pos) for name, pos in self._component_positions.items()}

    def get_port_world_position(self, component_name: str, port_name: str) -> np.ndarray:
        """
        Get the world position of a specific port.

        Args:
            component_name: Name of the component
            port_name: Name of the port on that component

        Returns:
            World position as numpy array [x, y, z]
        """
        component = self.get_component(component_name)
        port = component.ports[port_name]
        return self._get_port_world_pos(component_name, port)

    def validate_system_configuration(
        self,
        air_flow_m3_h: float,
        particle_density: float = 1420.0,
        min_particle_um: float = 5.0,
        max_particle_um: float = 100.0,
    ) -> dict:
        """
        Validate entire classification system configuration.

        Should be called before running simulation to check that operating
        conditions (air flow) match the component geometries for the intended
        particle size range.

        Args:
            air_flow_m3_h: Air flow rate [m³/h]
            particle_density: Particle density [kg/m³]
            min_particle_um: Smallest particle to separate [µm]
            max_particle_um: Largest particle to separate [µm]

        Returns:
            Dictionary with valid, warnings, errors, components.zigzag,
            components.cyclones, and optional recommendation.
        """
        Q = air_flow_m3_h / 3600.0  # m³/s

        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "components": {},
        }

        zigzag_val = self.zigzag.validate_operating_conditions(
            Q,
            min_particle_um * 1e-6,
            max_particle_um * 1e-6,
            particle_density,
        )
        result["components"]["zigzag"] = zigzag_val
        if not zigzag_val["valid"]:
            result["valid"] = False
            result["errors"].extend(zigzag_val["errors"])

        cyclone_val = self.multi_cyclone.validate_staging(Q, particle_density)
        result["components"]["cyclones"] = cyclone_val
        if not cyclone_val["valid"]:
            result["valid"] = False
            result["errors"].extend(cyclone_val["errors"])

        if not result["valid"] and "recommended_flow_m3_h" in cyclone_val:
            result["recommendation"] = (
                f"Current flow ({air_flow_m3_h:.0f} m³/h) is incompatible with classification. "
                f"Recommended: {cyclone_val['recommended_flow_m3_h']:.0f} m³/h for design cut sizes."
            )

        return result

    def to_warp_mesh(self) -> wp.Mesh:
        """Create a Warp mesh from the system geometry."""
        if not self._mesh_built:
            self.build_mesh()

        points = wp.array(self._combined_vertices, dtype=wp.vec3, device=self.device)
        indices = wp.array(self._combined_indices, dtype=wp.int32, device=self.device)

        return wp.Mesh(points=points, indices=indices)

    def print_summary(self):
        """Print summary of the classification system."""
        p = self.params

        print("=" * 70)
        print("Classification System Assembly Summary")
        print("=" * 70)
        print("\nFLOW PATH:")
        print("  Air Supply -> Venturi -> Zigzag -> Cyclones -> Bag Filter -> Clean Air")
        print("                            |")
        print("                      Coarse (starch) out")

        print("\n" + "-" * 70)
        print("1. VENTURI EDUCTOR (Particle Entrainment)")
        print("-" * 70)
        pos = self._component_positions['venturi']
        print(f"   Position:       ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m")
        print(f"   Axis:           Y (vertical, upward flow)")
        print(f"   Inlet diameter: {p.venturi_inlet_diameter * 1000:.0f} mm")
        print(f"   Throat ratio:   {p.venturi_throat_ratio:.2f}")
        print(f"   Total length:   {self.venturi.params.total_length * 1000:.0f} mm")

        print("\n   Ports:")
        for port_name, port in self.venturi.ports.items():
            world_pos = self.get_port_world_position('venturi', port_name)
            dim = f"D={port.diameter*1000:.0f}mm"
            print(f"     {port_name:15s} pos=({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}) "
                  f"dir={port.direction} {dim}")

        print("\n" + "-" * 70)
        print("2. ZIGZAG CLASSIFIER (Primary Separation)")
        print("-" * 70)
        pos = self._component_positions['zigzag']
        print(f"   Position:       ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m")
        print(f"   Channel width:  {p.zigzag_channel_width * 1000:.0f} mm")
        print(f"   Channel depth:  {p.zigzag_channel_depth * 1000:.0f} mm")
        print(f"   Stages:         {p.zigzag_num_stages}")
        print(f"   Total height:   {self.zigzag.params.total_height * 1000:.0f} mm")

        print("\n   Ports:")
        for port_name, port in self.zigzag.ports.items():
            world_pos = self.get_port_world_position('zigzag', port_name)
            dim = f"D={port.diameter*1000:.0f}mm" if port.diameter > 0 else f"W={port.width*1000:.0f}mm"
            print(f"     {port_name:15s} pos=({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}) "
                  f"dir={port.direction} {dim}")

        print("\n" + "-" * 70)
        print("3. MULTI-CYCLONE SYSTEM (Staged Collection)")
        print("-" * 70)
        pos = self._component_positions['multi_cyclone']
        print(f"   Position:       ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m")
        print(f"   Arrangement:    {self.multi_cyclone.params.arrangement}")
        for info in self.multi_cyclone.get_stage_info():
            print(f"     {info['name'].title():12s} D={info['diameter']:.0f}mm, "
                  f"d50={info['design_d50']:.0f}um, H={info['total_height']:.0f}mm")

        print("\n   Ports:")
        for port_name, port in self.multi_cyclone.ports.items():
            world_pos = self.get_port_world_position('multi_cyclone', port_name)
            dim = f"D={port.diameter*1000:.0f}mm" if port.diameter > 0 else f"W={port.width*1000:.0f}mm"
            print(f"     {port_name:20s} pos=({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}) {dim}")

        print("\n" + "-" * 70)
        print("4. BAG FILTER (Fine Particle Collection)")
        print("-" * 70)
        pos = self._component_positions['bag_filter']
        print(f"   Position:       ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m")
        print(f"   Number of bags: {self.bag_filter.params.num_bags}")
        print(f"   Filter area:    {self.bag_filter.params.total_filter_area:.1f} m2")
        print(f"   Air-to-cloth:   {self.bag_filter.params.air_to_cloth_ratio:.1f} m3/min/m2")

        print("\n   Ports:")
        for port_name, port in self.bag_filter.ports.items():
            world_pos = self.get_port_world_position('bag_filter', port_name)
            dim = f"D={port.diameter*1000:.0f}mm" if port.diameter > 0 else f"W={port.width*1000:.0f}mm"
            print(f"     {port_name:20s} pos=({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}) {dim}")

        print("\n" + "-" * 70)
        print("5. CONNECTING DUCTWORK")
        print("-" * 70)
        for i, (duct, position) in enumerate(self._duct_sections):
            duct_type = type(duct).__name__
            if hasattr(duct, 'params'):
                if hasattr(duct.params, 'diameter'):
                    dim = f"D={duct.params.diameter*1000:.0f}mm"
                else:
                    dim = ""
                if hasattr(duct.params, 'length'):
                    length = f"L={duct.params.length*1000:.0f}mm"
                elif hasattr(duct.params, 'bend_radius'):
                    length = f"R={duct.params.bend_radius*1000:.0f}mm"
                else:
                    length = ""
                print(f"   [{i+1}] {duct_type:20s} at ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) "
                      f"{length} {dim}")
            else:
                print(f"   [{i+1}] {duct_type} at ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")

        print("\n" + "=" * 70)
        extent = self.get_system_extent()
        print(f"System extent: {extent[0]*1000:.0f} x {extent[1]*1000:.0f} x {extent[2]*1000:.0f} mm")
        print(f"               ({extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m)")

        if self._mesh_built:
            n_verts = len(self._combined_vertices)
            n_tris = len(self._combined_indices) // 3
            print(f"Total mesh:    {n_verts:,} vertices, {n_tris:,} triangles")
        print("=" * 70)

    @property
    def vertices(self) -> np.ndarray:
        """Get combined mesh vertices."""
        if not self._mesh_built:
            self.build_mesh()
        return self._combined_vertices

    @property
    def indices(self) -> np.ndarray:
        """Get combined mesh indices."""
        if not self._mesh_built:
            self.build_mesh()
        return self._combined_indices


def create_standard_classification_system(device: str = "cpu") -> ClassificationSystemAssembly:
    """
    Create a standard classification system with default parameters.

    Args:
        device: Warp device

    Returns:
        ClassificationSystemAssembly instance
    """
    return ClassificationSystemAssembly(device=device)


def create_protein_separation_system(
    throughput_kg_h: float = 100,
    device: str = "cpu"
) -> ClassificationSystemAssembly:
    """
    Create a protein separation system sized for given throughput.

    Args:
        throughput_kg_h: Design throughput [kg/h]
        device: Warp device

    Returns:
        ClassificationSystemAssembly configured for protein separation
    """
    # Scale parameters based on throughput
    air_flow_m3s = throughput_kg_h * 2.5 / 3600  # m3/s
    scale = (throughput_kg_h / 100) ** 0.5  # Square root scaling

    params = ClassificationSystemParams(
        zigzag_channel_width=0.15 * scale,
        zigzag_num_stages=5,
        zigzag_channel_depth=0.30 * scale,
        venturi_inlet_diameter=0.10 * scale,
        venturi_throat_ratio=0.5,
        primary_cyclone_diameter=0.40 * scale,
        secondary_cyclone_diameter=0.25 * scale,
        tertiary_cyclone_diameter=0.15 * scale,
        bag_filter_flow_rate=max(air_flow_m3s, 0.5),
        bag_filter_air_to_cloth=2.0,
    )

    return ClassificationSystemAssembly(params, device=device)
