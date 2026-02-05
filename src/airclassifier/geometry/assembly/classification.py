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
    # Cross-section sized to match venturi outlet area (~41 cm²) so flow
    # velocity is maintained through the transition (avoids 6× expansion
    # that kills particle transport).  60×80mm = 48 cm².
    zigzag_channel_width: float = 0.060     # [m] 60mm width
    zigzag_num_stages: int = 5              # 5 stages for good separation
    zigzag_channel_depth: float = 0.080     # [m] 80mm depth

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

    # Coarse dropout hopper (integrated into venturi-to-zigzag transition)
    include_dropout: bool = True          # Whether to include dropout hopper on transition
    dropout_hopper_height: float = 0.15   # [m] Hopper cone height
    dropout_hopper_outlet_d: float = 0.04 # [m] Hopper discharge opening diameter

    # Coarse collection hardware (rotary airlocks below discharge points)
    include_coarse_collection: bool = True   # Airlock below zigzag coarse outlet
    coarse_airlock_rotor_d: float = 0.05     # [m] 50mm rotor diameter (sized for 30mm coarse outlet)

    # Dropout collection hardware (only active if include_dropout=True)
    include_dropout_collection: bool = True  # Airlock below dropout hopper discharge
    dropout_airlock_rotor_d: float = 0.06    # [m] 60mm rotor diameter

    # Bypass duct (splits flow before venturi, merges before cyclones)
    include_bypass: bool = False          # Whether to build bypass geometry
    bypass_diameter: float = 0.06         # [m] Bypass duct diameter (or auto-size)
    bypass_stub_length: float = 0.05      # [m] Tee branch stub length

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
        # Collection hardware (coarse/dropout airlocks + routing) - separate from main flow
        self._collection_duct_sections: List[Tuple[Any, Tuple[float, float, float]]] = []

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

        # Bypass split tee below venturi (if bypass enabled)
        if p.include_bypass:
            from ..components.tee_junction import TeeJunction, TeeJunctionParams
            split_tee_main_length = 0.10
            split_tee = TeeJunction(TeeJunctionParams(
                main_diameter=p.venturi_inlet_diameter,
                branch_diameter=p.bypass_diameter,
                main_length=split_tee_main_length,
                branch_stub_length=p.bypass_stub_length,
                main_direction=(0.0, 1.0, 0.0),
                branch_direction=(1.0, 0.0, 0.0),
                center=(0, 0, 0),
            ))
            # Position so tee outlet aligns with venturi air_inlet (with gap)
            # Tee outlet port (local): (0, main_length/2, 0)
            split_tee_pos = (
                p.center[0],
                p.center[1] - gap - split_tee_main_length / 2,
                p.center[2],
            )
            self._duct_sections.append((split_tee, split_tee_pos))
            self._bypass_split_tee = split_tee
            self._bypass_split_tee_pos = split_tee_pos

        # Get venturi outlet port in world coordinates
        venturi_outlet = self.venturi.ports['outlet']
        venturi_outlet_world = self._get_port_world_pos('venturi', venturi_outlet)

        # ============================================================
        # 2. ZIGZAG CLASSIFIER - Above venturi
        # ============================================================
        # Zigzag receives air+particles from venturi
        # air_inlet at bottom (receives from venturi), fines_outlet at top
        # Uses deflector plate geometry for proper separation physics
        zigzag_params = ZigzagClassifierParams(
            channel_width=p.zigzag_channel_width,
            channel_depth=p.zigzag_channel_depth,
            num_stages=p.zigzag_num_stages,
            stage_height=p.zigzag_channel_width * 1.5,
            # Deflector plate parameters (replaces old zigzag_angle)
            plate_angle=np.radians(45),       # 45° from vertical - good separation/pressure balance
            plate_length_ratio=0.5,            # Plate extends 50% across channel
            plate_thickness=0.003,
            feed_stage=(p.zigzag_num_stages + 1) // 2,
            feed_width=p.zigzag_channel_width * 0.5,
            feed_angle=0.0,
            air_inlet_width=p.zigzag_channel_width,
            air_inlet_height=p.zigzag_channel_width * 0.5,
            fines_outlet_width=p.zigzag_channel_width,
            fines_outlet_height=p.zigzag_channel_width * 0.5,
            coarse_outlet_width=p.zigzag_channel_width * 0.5,
            coarse_outlet_height=p.zigzag_channel_width * 0.3,
            wall_thickness=0.003,
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

        if p.include_dropout:
            # Expanding transition with integrated dropout hopper
            from ..components.expanding_transition import (
                ExpandingTransitionWithDropout, ExpandingTransitionParams,
            )
            trans1 = ExpandingTransitionWithDropout(ExpandingTransitionParams(
                inlet_diameter=venturi_d,
                outlet_width=zigzag_inlet_w,
                outlet_depth=zigzag_inlet_h,
                transition_length=trans1_length,
                hopper_height=p.dropout_hopper_height,
                hopper_outlet_diameter=p.dropout_hopper_outlet_d,
                wall_thickness=0.002,
                direction=(0.0, 1.0, 0.0),
                center=(0, 0, 0),
                flanged=True,
            ))
        else:
            trans1 = Transition(TransitionParams(
                transition_type="round_to_rect",
                inlet_dimensions=(venturi_d,),
                outlet_dimensions=(zigzag_inlet_w, zigzag_inlet_h),
                length=trans1_length,
                concentric=True,
                wall_thickness=0.002,
                direction=(0.0, 1.0, 0.0),
                center=(0, 0, 0),
                flanged=True,
            ))
        self._duct_sections.append((trans1, trans1_start))

        # ============================================================
        # 2a. COARSE COLLECTION - Routed to -X side of zigzag
        # ============================================================
        # The coarse outlet is directly above the air supply duct, so
        # collection hardware must route SIDEWAYS (-X) to clear it.
        # Route: coarse_outlet → horizontal transition (-X) → 90° elbow → airlock
        if p.include_coarse_collection:
            from ..components import RotaryAirlock, RotaryAirlockParams

            zigzag_coarse = self.zigzag.ports['coarse_outlet']
            zigzag_coarse_world = self._get_port_world_pos('zigzag', zigzag_coarse)

            # Rect-to-round transition oriented HORIZONTALLY (-X) from coarse outlet
            coarse_round_d = zigzag_coarse.width  # 60mm
            diag = np.sqrt(zigzag_coarse.width**2 + zigzag_coarse.height**2) / 2
            r = coarse_round_d / 2
            coarse_trans_length = abs(diag - r) / np.tan(np.radians(15))
            coarse_trans_length = max(coarse_trans_length, 0.10)

            coarse_trans = Transition(TransitionParams(
                transition_type="rect_to_round",
                inlet_dimensions=(zigzag_coarse.width, zigzag_coarse.height),
                outlet_dimensions=(coarse_round_d,),
                length=coarse_trans_length,
                concentric=True,
                wall_thickness=0.002,
                direction=(-1.0, 0.0, 0.0),  # Horizontal to -X (away from cyclones)
                center=(0, 0, 0),
            ))
            coarse_trans_start = (
                zigzag_coarse_world[0] - gap,
                zigzag_coarse_world[1],
                zigzag_coarse_world[2],
            )
            self._collection_duct_sections.append((coarse_trans, coarse_trans_start))

            # 90° elbow turning from -X to -Y (downward)
            coarse_elbow_R = coarse_round_d * 1.5
            coarse_elbow = DuctElbow(DuctElbowParams(
                diameter=coarse_round_d,
                bend_radius=coarse_elbow_R,
                angle=90.0,
                wall_thickness=0.002,
                flanged=True,
                center=(0, 0, 0),
                inlet_direction=(-1.0, 0.0, 0.0),
                rotation_axis=(0.0, 0.0, -1.0),  # Turns -X → -Y
            ))
            coarse_elbow_inlet = (
                coarse_trans_start[0] - coarse_trans_length - gap,
                coarse_trans_start[1],
                coarse_trans_start[2],
            )
            coarse_elbow_outlet_local = coarse_elbow.get_outlet_position()
            coarse_elbow_outlet = (
                coarse_elbow_inlet[0] + coarse_elbow_outlet_local[0],
                coarse_elbow_inlet[1] + coarse_elbow_outlet_local[1],
                coarse_elbow_inlet[2] + coarse_elbow_outlet_local[2],
            )
            self._collection_duct_sections.append((coarse_elbow, coarse_elbow_inlet))

            # Rotary airlock below elbow outlet
            coarse_airlock = RotaryAirlock(RotaryAirlockParams(
                rotor_diameter=p.coarse_airlock_rotor_d,
                rotor_length=p.coarse_airlock_rotor_d * 0.6,
                num_vanes=8,
                vane_thickness=0.004,
                vane_tip_clearance=0.0003,
                inlet_diameter=coarse_round_d,
                outlet_diameter=coarse_round_d * 0.85,
            ))
            airlock_inlet_port = coarse_airlock.ports['inlet']
            coarse_airlock_pos = (
                coarse_elbow_outlet[0],
                coarse_elbow_outlet[1] - gap - airlock_inlet_port.position[1],
                coarse_elbow_outlet[2],
            )
            self._collection_duct_sections.append((coarse_airlock, coarse_airlock_pos))

        # ============================================================
        # 2b. DROPOUT COLLECTION - Routed to -X side to clear venturi
        # ============================================================
        # The dropout hopper discharges directly above the venturi, so
        # we route sideways (-X) with an elbow before the airlock.
        # Route: dropout → 90° elbow (-Y to -X) → short duct → 90° elbow (-X to -Y) → airlock
        if p.include_dropout and p.include_dropout_collection:
            from ..components import RotaryAirlock, RotaryAirlockParams

            if hasattr(trans1, 'ports') and 'dropout' in trans1.ports:
                dropout_port = trans1.ports['dropout']
                dropout_world = np.array(trans1_start) + np.array(dropout_port.position)
                dropout_d = p.dropout_hopper_outlet_d  # 40mm

                # Elbow 1: turns from -Y (down) to -X (left)
                dropout_elbow_R = dropout_d * 2.0  # Generous bend for powder flow
                dropout_elbow1 = DuctElbow(DuctElbowParams(
                    diameter=dropout_d,
                    bend_radius=dropout_elbow_R,
                    angle=90.0,
                    wall_thickness=0.002,
                    flanged=True,
                    center=(0, 0, 0),
                    inlet_direction=(0.0, -1.0, 0.0),
                    rotation_axis=(0.0, 0.0, 1.0),  # Turns -Y → -X
                ))
                dropout_elbow1_inlet = (
                    dropout_world[0],
                    dropout_world[1] - gap,
                    dropout_world[2],
                )
                elbow1_outlet_local = dropout_elbow1.get_outlet_position()
                elbow1_outlet = (
                    dropout_elbow1_inlet[0] + elbow1_outlet_local[0],
                    dropout_elbow1_inlet[1] + elbow1_outlet_local[1],
                    dropout_elbow1_inlet[2] + elbow1_outlet_local[2],
                )
                self._collection_duct_sections.append((dropout_elbow1, dropout_elbow1_inlet))

                # Short horizontal duct going -X to clear venturi envelope
                dropout_duct_length = 0.10  # 100mm
                dropout_duct = RoundDuct(RoundDuctParams(
                    diameter=dropout_d,
                    length=dropout_duct_length,
                    wall_thickness=0.002,
                    direction=(-1.0, 0.0, 0.0),
                    center=(0, 0, 0),
                    flanged=True,
                ))
                dropout_duct_start = (
                    elbow1_outlet[0] - gap,
                    elbow1_outlet[1],
                    elbow1_outlet[2],
                )
                self._collection_duct_sections.append((dropout_duct, dropout_duct_start))

                # Elbow 2: turns from -X to -Y (downward to airlock)
                dropout_elbow2 = DuctElbow(DuctElbowParams(
                    diameter=dropout_d,
                    bend_radius=dropout_elbow_R,
                    angle=90.0,
                    wall_thickness=0.002,
                    flanged=True,
                    center=(0, 0, 0),
                    inlet_direction=(-1.0, 0.0, 0.0),
                    rotation_axis=(0.0, 0.0, -1.0),  # Turns -X → -Y
                ))
                dropout_elbow2_inlet = (
                    dropout_duct_start[0] - dropout_duct_length - gap,
                    dropout_duct_start[1],
                    dropout_duct_start[2],
                )
                elbow2_outlet_local = dropout_elbow2.get_outlet_position()
                elbow2_outlet = (
                    dropout_elbow2_inlet[0] + elbow2_outlet_local[0],
                    dropout_elbow2_inlet[1] + elbow2_outlet_local[1],
                    dropout_elbow2_inlet[2] + elbow2_outlet_local[2],
                )
                self._collection_duct_sections.append((dropout_elbow2, dropout_elbow2_inlet))

                # Rotary airlock below elbow 2
                dropout_airlock = RotaryAirlock(RotaryAirlockParams(
                    rotor_diameter=p.dropout_airlock_rotor_d,
                    rotor_length=p.dropout_airlock_rotor_d * 0.6,
                    num_vanes=6,
                    vane_thickness=0.003,
                    vane_tip_clearance=0.0003,
                    inlet_diameter=dropout_d,
                    outlet_diameter=dropout_d * 0.85,
                ))
                dropout_inlet_port = dropout_airlock.ports['inlet']
                dropout_airlock_pos = (
                    elbow2_outlet[0],
                    elbow2_outlet[1] - gap - dropout_inlet_port.position[1],
                    elbow2_outlet[2],
                )
                self._collection_duct_sections.append((dropout_airlock, dropout_airlock_pos))

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

        # STEP 3: Horizontal round duct (with optional merge tee for bypass)
        duct2_length = p.primary_cyclone_diameter * 0.5  # Shorter duct
        next_x = elbow2_outlet_pos[0] + gap

        if p.include_bypass:
            # Insert merge tee for bypass flow to rejoin the fines path
            from ..components.tee_junction import TeeJunction, TeeJunctionParams
            merge_tee_main_length = 0.10
            merge_tee = TeeJunction(TeeJunctionParams(
                main_diameter=duct2_diameter,
                branch_diameter=p.bypass_diameter,
                main_length=merge_tee_main_length,
                branch_stub_length=p.bypass_stub_length,
                main_direction=(1.0, 0.0, 0.0),
                branch_direction=(0.0, -1.0, 0.0),  # Bypass arrives from below
                center=(0, 0, 0),
            ))
            # Position so tee inlet aligns with elbow2 outlet
            # Tee inlet port (local): (-main_length/2, 0, 0)
            merge_tee_pos = (
                next_x + merge_tee_main_length / 2,
                elbow2_outlet_pos[1],
                elbow2_outlet_pos[2],
            )
            self._duct_sections.append((merge_tee, merge_tee_pos))
            self._bypass_merge_tee = merge_tee
            self._bypass_merge_tee_pos = merge_tee_pos

            # Duct2 starts after merge tee
            duct2_start = (
                next_x + merge_tee_main_length + gap,
                elbow2_outlet_pos[1],
                elbow2_outlet_pos[2],
            )
        else:
            duct2_start = (
                next_x,
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

        # ============================================================
        # 5. BYPASS ROUTING (if enabled)
        # ============================================================
        # Route: split_tee.branch → horizontal duct (+X) → 90° elbow →
        #        vertical duct (+Y) → merge_tee.branch (from below)
        if p.include_bypass:
            bypass_d = p.bypass_diameter
            bypass_elbow_R = bypass_d * 1.5

            # Split tee branch port world position
            split_branch = self._bypass_split_tee.ports['branch']
            split_branch_world = (
                np.array(self._bypass_split_tee_pos)
                + np.array(split_branch.position)
            )

            # Merge tee branch port world position
            merge_branch = self._bypass_merge_tee.ports['branch']
            merge_branch_world = (
                np.array(self._bypass_merge_tee_pos)
                + np.array(merge_branch.position)
            )

            # The vertical duct must arrive at merge_branch_world from below.
            # The elbow turns +X → +Y, so elbow outlet is at:
            #   (elbow_inlet_X + R, elbow_inlet_Y + R, Z)
            # We need elbow_outlet_X = merge_branch_world_X
            # So: elbow_inlet_X = merge_branch_world_X - R
            target_elbow_inlet_x = merge_branch_world[0] - bypass_elbow_R

            # Horizontal bypass duct from split tee branch going +X
            bp_horiz_start = (
                split_branch_world[0] + gap,
                split_branch_world[1],
                split_branch_world[2],
            )
            bp_horiz_length = target_elbow_inlet_x - gap - bp_horiz_start[0]
            bp_horiz_length = max(bp_horiz_length, 0.02)  # Minimum 20mm

            bp_horiz_duct = RoundDuct(RoundDuctParams(
                diameter=bypass_d,
                length=bp_horiz_length,
                wall_thickness=0.002,
                direction=(1.0, 0.0, 0.0),
                center=(0, 0, 0),
                flanged=True,
            ))
            self._duct_sections.append((bp_horiz_duct, bp_horiz_start))

            # 90° elbow turning from +X to +Y
            bp_elbow_inlet = (
                bp_horiz_start[0] + bp_horiz_length + gap,
                bp_horiz_start[1],
                bp_horiz_start[2],
            )
            bp_elbow = DuctElbow(DuctElbowParams(
                diameter=bypass_d,
                bend_radius=bypass_elbow_R,
                angle=90.0,
                wall_thickness=0.002,
                flanged=True,
                center=(0, 0, 0),
                inlet_direction=(1.0, 0.0, 0.0),
                rotation_axis=(0.0, 0.0, -1.0),  # Turns +X → +Y
            ))
            bp_elbow_local_outlet = bp_elbow.get_outlet_position()
            bp_elbow_outlet = (
                bp_elbow_inlet[0] + bp_elbow_local_outlet[0],
                bp_elbow_inlet[1] + bp_elbow_local_outlet[1],
                bp_elbow_inlet[2] + bp_elbow_local_outlet[2],
            )
            self._duct_sections.append((bp_elbow, bp_elbow_inlet))

            # Vertical bypass duct going +Y from elbow outlet to merge tee branch
            bp_vert_start = (
                bp_elbow_outlet[0],
                bp_elbow_outlet[1] + gap,
                bp_elbow_outlet[2],
            )
            bp_vert_end_y = merge_branch_world[1] - gap
            bp_vert_length = bp_vert_end_y - bp_vert_start[1]
            bp_vert_length = max(bp_vert_length, 0.05)

            bp_vert_duct = RoundDuct(RoundDuctParams(
                diameter=bypass_d,
                length=bp_vert_length,
                wall_thickness=0.002,
                direction=(0.0, 1.0, 0.0),
                center=(0, 0, 0),
                flanged=True,
            ))
            self._duct_sections.append((bp_vert_duct, bp_vert_start))

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

        # Add collection hardware (coarse/dropout airlocks and routing)
        for duct, position in self._collection_duct_sections:
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
        *,
        classification_flow_m3_h: float | None = None,
        cyclone_flow_m3_h: float | None = None,
    ) -> dict:
        """
        Validate entire classification system configuration.

        Should be called before running simulation to check that operating
        conditions (air flow) match the component geometries for the intended
        particle size range.

        When bypass is used, pass classification_flow_m3_h (flow through
        venturi+zigzag) and cyclone_flow_m3_h (total flow after bypass merge)
        so zigzag d50 and cyclone staging use the correct flows.

        Args:
            air_flow_m3_h: Air flow rate [m³/h] (used for both if optional flows not set)
            particle_density: Particle density [kg/m³]
            min_particle_um: Smallest particle to separate [µm]
            max_particle_um: Largest particle to separate [µm]
            classification_flow_m3_h: Flow through zigzag [m³/h]; if set, used for zigzag validation.
            cyclone_flow_m3_h: Flow through cyclones (after bypass merge) [m³/h]; if set, used for cyclone validation.

        Returns:
            Dictionary with valid, warnings, errors, components.zigzag,
            components.cyclones, and optional recommendation.
        """
        Q_zigzag = (classification_flow_m3_h if classification_flow_m3_h is not None else air_flow_m3_h) / 3600.0
        Q_cyclone = (cyclone_flow_m3_h if cyclone_flow_m3_h is not None else air_flow_m3_h) / 3600.0

        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "components": {},
        }

        zigzag_val = self.zigzag.validate_operating_conditions(
            Q_zigzag,
            min_particle_um * 1e-6,
            max_particle_um * 1e-6,
            particle_density,
        )
        result["components"]["zigzag"] = zigzag_val
        if not zigzag_val["valid"]:
            result["valid"] = False
            result["errors"].extend(zigzag_val["errors"])

        cyclone_val = self.multi_cyclone.validate_staging(Q_cyclone, particle_density)
        result["components"]["cyclones"] = cyclone_val
        if not cyclone_val["valid"]:
            result["valid"] = False
            result["errors"].extend(cyclone_val["errors"])

        flow_for_recommendation = cyclone_flow_m3_h if cyclone_flow_m3_h is not None else air_flow_m3_h
        if not result["valid"] and "recommended_flow_m3_h" in cyclone_val:
            result["recommendation"] = (
                f"Current flow ({flow_for_recommendation:.0f} m³/h) is incompatible with classification. "
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

        if p.include_coarse_collection:
            coarse_port = self.zigzag.ports['coarse_outlet']
            print(f"\n   Coarse collection hardware:")
            print(f"     Transition:     rect {coarse_port.width*1000:.0f}x{coarse_port.height*1000:.0f}mm "
                  f"-> round {coarse_port.width*1000:.0f}mm")
            print(f"     Airlock rotor:  D={p.coarse_airlock_rotor_d*1000:.0f}mm")

        if p.include_dropout and p.include_dropout_collection:
            print(f"\n   Dropout collection hardware:")
            print(f"     Hopper outlet:  D={p.dropout_hopper_outlet_d*1000:.0f}mm")
            print(f"     Airlock rotor:  D={p.dropout_airlock_rotor_d*1000:.0f}mm")

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
        print("5. CONNECTING DUCTWORK (main flow path)")
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

        if self._collection_duct_sections:
            print(f"\n   Collection hardware ({len(self._collection_duct_sections)} sections):")
            for i, (duct, position) in enumerate(self._collection_duct_sections):
                duct_type = type(duct).__name__
                if hasattr(duct, 'params'):
                    if hasattr(duct.params, 'diameter'):
                        dim = f"D={duct.params.diameter*1000:.0f}mm"
                    elif hasattr(duct.params, 'rotor_diameter'):
                        dim = f"D_rotor={duct.params.rotor_diameter*1000:.0f}mm"
                    else:
                        dim = ""
                    if hasattr(duct.params, 'length'):
                        length = f"L={duct.params.length*1000:.0f}mm"
                    elif hasattr(duct.params, 'bend_radius'):
                        length = f"R={duct.params.bend_radius*1000:.0f}mm"
                    else:
                        length = ""
                    print(f"   [C{i+1}] {duct_type:20s} at ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) "
                          f"{length} {dim}")
                else:
                    print(f"   [C{i+1}] {duct_type} at ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")

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


def create_standard_classification_system(
    device: str = "cpu",
    params: ClassificationSystemParams = None,
) -> ClassificationSystemAssembly:
    """
    Create a classification system with default or custom parameters.

    Args:
        device: Warp device
        params: Optional custom parameters (uses defaults if None)

    Returns:
        ClassificationSystemAssembly instance
    """
    if params is not None:
        return ClassificationSystemAssembly(params=params, device=device)
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
