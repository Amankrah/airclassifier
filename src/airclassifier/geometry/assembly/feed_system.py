"""
Feed System Assembly Module
===========================

This module provides the complete feed system assembly for powder/flour handling
in air classification systems. It combines all Phase 2 components into an integrated
material handling train that prepares powder for the classification process.

SYSTEM OVERVIEW
===============

The feed system is the starting point of the air classification process. Its purpose
is to:
1. Store bulk powder material (Feed Hopper)
2. Meter and seal the material flow (Rotary Airlock)
3. Provide controlled, consistent feed rate (Screw Feeder)
4. Break up agglomerates/lumps before classification (Deagglomerator)

MATERIAL FLOW PATH
==================

    ┌─────────────────────────────────────────────────────────────────────┐
    │                         FEED HOPPER                                 │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │                                                             │   │
    │  │   ╔═══════════════════════════════════════════════════╗    │   │
    │  │   ║              BULK POWDER STORAGE                  ║    │   │
    │  │   ║          (Cylindrical + Conical Sections)         ║    │   │
    │  │   ║                                                   ║    │   │
    │  │   ║   - Capacity: 500 kg (configurable)              ║    │   │
    │  │   ║   - Mass flow design (cone angle > angle of      ║    │   │
    │  │   ║     repose + 10-15°)                             ║    │   │
    │  │   ║   - Hinged lid with T-bar handle                 ║    │   │
    │  │   ║   - Inner skirt for dust-tight seal              ║    │   │
    │  │   ╚═══════════════════════════════════════════════════╝    │   │
    │  │                         │                                   │   │
    │  │                         ▼ (Gravity)                         │   │
    │  │              ┌──────────────────────┐                       │   │
    │  │              │   DISCHARGE RING     │                       │   │
    │  │              │   (150mm diameter)   │                       │   │
    │  └──────────────┴──────────┬───────────┴───────────────────────┘   │
    │                            │                                        │
    │                 ┌──────────┴──────────┐                             │
    │                 │ TRANSITION CONNECTOR │  ← Dust-tight seal         │
    │                 │   (flanged pipe)     │                            │
    │                 └──────────┬──────────┘                             │
    └────────────────────────────┼────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                       ROTARY AIRLOCK                                │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │              ┌───────────────────┐                          │   │
    │  │              │    INLET NECK     │  ← Saddle joint on       │   │
    │  │              │   with flange     │    cylindrical housing   │   │
    │  │              └─────────┬─────────┘                          │   │
    │  │                        │                                    │   │
    │  │   ╔════════════════════╧════════════════════╗              │   │
    │  │   ║          CYLINDRICAL HOUSING            ║              │   │
    │  │   ║    ┌─────────────────────────────┐     ║              │   │
    │  │   ║    │   ╱ ╲     ROTOR     ╱ ╲     │     ║              │   │
    │  │   ║    │  ╱   ╲   (8 vanes) ╱   ╲    │     ║   ROTATION   │   │
    │  │   ║    │ ╱     ╲    ●     ╱     ╲   │     ║   AXIS: Z    │   │
    │  │   ║    │╱       ╲  HUB   ╱       ╲  │     ║              │   │
    │  │   ║    │╲       ╱       ╲       ╱  │     ║              │   │
    │  │   ║    │ ╲     ╱         ╲     ╱   │     ║              │   │
    │  │   ║    │  ╲   ╱           ╲   ╱    │     ║              │   │
    │  │   ║    │   ╲ ╱             ╲ ╱     │     ║              │   │
    │  │   ║    └─────────────────────────────┘     ║              │   │
    │  │   ╚════════════════════╤════════════════════╝              │   │
    │  │                        │                                    │   │
    │  │              ┌─────────┴─────────┐                          │   │
    │  │              │   OUTLET NECK     │  ← Saddle joint          │   │
    │  │              │   with flange     │                          │   │
    │  │              └───────────────────┘                          │   │
    │  │                                                             │   │
    │  │   Function: Pressure seal + volumetric metering             │   │
    │  │   - Vane tip clearance: 0.3mm (prevents jamming)           │   │
    │  │   - RPM: 20 (adjustable for feed rate control)             │   │
    │  │   - Prevents air backflow into hopper                       │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │                            │                                        │
    │                 ┌──────────┴──────────┐                             │
    │                 │ TRANSITION CONNECTOR │                            │
    │                 └──────────┬──────────┘                             │
    └────────────────────────────┼────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        SCREW FEEDER                                 │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │                                                             │   │
    │  │    INLET FLANGE (from airlock)                              │   │
    │  │         │                                                   │   │
    │  │         ▼                                                   │   │
    │  │   ╔═══════════════════════════════════════════════════╗    │   │
    │  │   ║         FULLY ENCLOSED CYLINDRICAL TUBE           ║    │   │
    │  │   ║   ┌─────────────────────────────────────────┐    ║    │   │
    │  │   ║   │  ╭──╮   ╭──╮   ╭──╮   ╭──╮   ╭──╮     │    ║    │   │
    │  │   ║   │  │  │   │  │   │  │   │  │   │  │     │    ║    │   │
    │  │   ║   │  ╰──╯   ╰──╯   ╰──╯   ╰──╯   ╰──╯     │    ║    │   │
    │  │   ║   │        HELICAL SCREW FLIGHTS           │    ║    │   │
    │  │   ║   │     (pushed by rotation → → →)        │    ║    │   │
    │  │   ║   └─────────────────────────────────────────┘    ║    │   │
    │  │   ╚══════════════════════════════════════════════╤══╝    │   │
    │  │                                                   │       │   │
    │  │   Function: Controlled volumetric feed rate      ▼       │   │
    │  │   - ENCLOSED TUBE (dust-tight, no particle escape)       │   │
    │  │   - Screw diameter: 100mm                    OUTLET      │   │
    │  │   - Pitch: 80mm (0.8 × diameter)            FLANGE       │   │
    │  │   - Target rate: 500 kg/h                   (to deagg)   │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │                            │                                        │
    │                 ┌──────────┴──────────┐                             │
    │                 │ FLANGED TRANSITION  │  ← Conical if diameters    │
    │                 │   (dust-tight)      │    differ, with flanges    │
    │                 └──────────┬──────────┘                             │
    └────────────────────────────┼────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      DEAGGLOMERATOR                                 │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │                                                             │   │
    │  │    INLET (from feeder)                                      │   │
    │  │         │                                                   │   │
    │  │         ▼                                                   │   │
    │  │   ╔═══════════════════════════════════════════════════╗    │   │
    │  │   ║              CYLINDRICAL HOUSING                  ║    │   │
    │  │   ║   ┌─────────────────────────────────────────┐    ║    │   │
    │  │   ║   │         ┌─────────────────┐             │    ║    │   │
    │  │   ║   │    ▪    │    PIN ROTOR    │    ▪        │    ║    │   │
    │  │   ║   │   ▪     │   ═══●═══       │     ▪       │    ║    │   │
    │  │   ║   │    ▪    │   (high speed)  │    ▪        │    ║    │   │
    │  │   ║   │         └─────────────────┘             │    ║    │   │
    │  │   ║   │    ▪  ▪  ▪  IMPACT PINS  ▪  ▪  ▪       │    ║    │   │
    │  │   ║   │         (break up lumps/agglomerates)   │    ║    │   │
    │  │   ║   └─────────────────────────────────────────┘    ║    │   │
    │  │   ║                      │                           ║    │   │
    │  │   ║             ╔════════╧════════╗                  ║    │   │
    │  │   ║             ║   SCREEN        ║  ← Only particles║    │   │
    │  │   ║             ║   (2mm mesh)    ║    < aperture    ║    │   │
    │  │   ║             ║                 ║    pass through  ║    │   │
    │  │   ║             ╚════════╤════════╝                  ║    │   │
    │  │   ╚══════════════════════╪═══════════════════════════╝    │   │
    │  │                          │                                │   │
    │  │                          ▼                                │   │
    │  │                    ┌───────────┐                          │   │
    │  │                    │  OUTLET   │                          │   │
    │  │                    └─────┬─────┘                          │   │
    │  │                          │                                │   │
    │  │   Function: Break agglomerates for uniform classification │   │
    │  │   - Rotor diameter: 200mm                                 │   │
    │  │   - 3 rows × 6 pins = 18 impact pins                     │   │
    │  │   - Screen aperture: 2mm (configurable)                  │   │
    │  │   - 40% open area for material passage                   │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │                            │                                        │
    └────────────────────────────┼────────────────────────────────────────┘
                                 │
                                 ▼
                    ════════════════════════
                      TO AIR CLASSIFIER
                    ════════════════════════


COMPONENT CONNECTIONS
=====================

All connections use a standardized port system for precise alignment:

    ┌─────────────────────────────────────────────────────────────┐
    │                    CONNECTION PORT SYSTEM                   │
    │                                                             │
    │   Each component has defined ports with:                    │
    │   - Position (x, y, z) relative to component origin        │
    │   - Direction vector (normal to port face)                 │
    │   - Diameter (for circular ports)                          │
    │   - Port type (FLANGED, CIRCULAR, GRAVITY)                 │
    │                                                             │
    │   Assembly uses calculate_alignment() to:                   │
    │   1. Match source port to target port                      │
    │   2. Align port directions (opposing)                      │
    │   3. Apply configurable gap (default 5mm)                  │
    │                                                             │
    │   Connection validation checks:                             │
    │   - Distance between mating surfaces                       │
    │   - Direction alignment (should be opposing)               │
    │   - Diameter compatibility                                 │
    └─────────────────────────────────────────────────────────────┘


TRANSITION CONNECTORS
=====================

Between each component, sealed transition connectors prevent particle escape:

    ┌─────────────┐
    │   FLANGE    │  ← Bolts to upstream component
    ├─────────────┤
    │             │
    │    PIPE     │  ← Cylindrical (or conical if diameters differ)
    │   SECTION   │
    │             │
    ├─────────────┤
    │   FLANGE    │  ← Bolts to downstream component
    └─────────────┘

Features:
- Match inlet/outlet diameters to adjacent components
- Flanged ends for bolted connections
- Optional bellows section for vibration isolation
- Dust-tight sealing


COORDINATE SYSTEM
=================

    Y (up)
    │
    │    Feed Hopper (top)
    │         │
    │         ▼
    │    Rotary Airlock
    │         │
    │         ▼
    │    Screw Feeder ──────► X (horizontal feed direction)
    │         │
    │         ▼
    │    Deagglomerator (bottom)
    │
    └────────────────────► Z

- Origin: Center of system (at hopper discharge level)
- Y-axis: Vertical (gravity direction is -Y)
- X-axis: Horizontal feed direction (screw feeder axis)
- Z-axis: Lateral (perpendicular to flow)


DESIGN PARAMETERS
=================

Default Configuration (500 kg/h flour processing):

    Component           Parameter                   Value
    ─────────────────────────────────────────────────────────
    Feed Hopper         Capacity                    500 kg
                        Discharge diameter          150 mm
                        Bulk density               500 kg/m³

    Rotary Airlock      Rotor diameter              200 mm
                        Rotor length                120 mm
                        Number of vanes             8
                        Vane tip clearance          0.3 mm
                        Inlet diameter              150 mm
                        Outlet diameter             135 mm

    Screw Feeder        Screw diameter              100 mm
                        Screw pitch                 80 mm
                        Trough length               240 mm
                        Target feed rate            500 kg/h

    Deagglomerator      Rotor diameter              200 mm
                        Housing diameter            260 mm
                        Pin rows × pins/row         3 × 6
                        Screen aperture             2 mm
                        Screen open area            40%

    Assembly            Component spacing           5 mm


USAGE
=====

    from airclassifier.geometry.assembly import (
        FeedSystemAssembly,
        FeedSystemParams,
        create_standard_feed_system
    )

    # Create with default parameters
    feed_system = create_standard_feed_system()

    # Or customize parameters
    params = FeedSystemParams(
        hopper_capacity_kg=1000,
        hopper_discharge_diameter=0.20,
        feeder_target_rate_kg_h=800,
    )
    feed_system = FeedSystemAssembly(params)

    # Build mesh for visualization/export
    vertices, indices = feed_system.build_mesh()

    # Access individual components
    hopper = feed_system.hopper
    airlock = feed_system.airlock
    feeder = feed_system.feeder
    deagglomerator = feed_system.deagglomerator

    # Validate connections
    report = feed_system.validate_connections()
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import numpy as np
import warp as wp

from ..connection_ports import (
    ConnectionPort, PortType, calculate_alignment, 
    validate_assembly_connections, print_connection_report
)


@dataclass
class FeedSystemParams:
    """
    Parameters for complete feed system assembly.

    This dataclass defines all configurable parameters for the feed system,
    organized by component. The parameters control sizing, capacity, and
    layout of the entire material handling train.

    Attributes
    ----------
    hopper_capacity_kg : float
        Target storage capacity of the feed hopper in kilograms.
        This determines the hopper volume based on bulk_density.
        Default: 500 kg (typical for pilot-scale operations)

    hopper_discharge_diameter : float
        Diameter of the hopper discharge opening in meters.
        This sets the flow rate potential and must match the airlock inlet.
        Default: 0.15 m (150 mm)

    airlock_rotor_diameter : float
        Diameter of the rotary airlock rotor in meters.
        Larger rotors = higher volumetric capacity.
        Default: 0.20 m (200 mm)

    feeder_screw_diameter : float
        Diameter of the screw feeder helical screw in meters.
        Determines the volumetric feed rate per revolution.
        Default: 0.10 m (100 mm)

    feeder_target_rate_kg_h : float
        Target feed rate in kilograms per hour.
        Used for screw speed calculations (not geometry).
        Default: 500 kg/h

    deagg_rotor_diameter : float
        Diameter of the deagglomerator rotor in meters.
        Larger = more throughput capacity.
        Default: 0.20 m (200 mm)

    deagg_screen_aperture : float
        Screen mesh opening size in meters.
        Controls maximum particle size to classifier.
        Default: 0.002 m (2 mm)

    component_spacing : float
        Gap between mating flanges in meters.
        Represents gasket/seal space in real systems.
        Default: 0.005 m (5 mm)

    center : Tuple[float, float, float]
        World coordinates of system center (hopper discharge level).
        Default: (0.0, 0.0, 0.0)

    bulk_density : float
        Material bulk density in kg/m³.
        Used for hopper sizing calculations.
        Default: 500.0 kg/m³ (typical for flour)

    Example
    -------
    >>> params = FeedSystemParams(
    ...     hopper_capacity_kg=1000,
    ...     feeder_target_rate_kg_h=800,
    ...     deagg_screen_aperture=0.003,  # 3mm screen
    ... )
    >>> system = FeedSystemAssembly(params)
    """

    # Feed hopper parameters
    hopper_capacity_kg: float = 500       # [kg] Storage capacity
    hopper_discharge_diameter: float = 0.15  # [m] Discharge opening diameter

    # Rotary airlock parameters  
    airlock_rotor_diameter: float = 0.20  # [m] Rotor diameter

    # Screw feeder parameters
    feeder_screw_diameter: float = 0.10 # [m] Screw diameter
    feeder_target_rate_kg_h: float = 500  # [kg/h] Target mass flow rate

    # De-agglomerator parameters
    deagg_rotor_diameter: float = 0.20    # [m] Rotor diameter
    deagg_screen_aperture: float = 0.001 # [m] Screen mesh size (1mm)

    # Layout parameters
    component_spacing: float = 0.002      # [m] Gap between flanges (2mm - tight fit)
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # System origin

    # Material properties
    bulk_density: float = 500.0           # [kg/m³] Material bulk density


class FeedSystemAssembly:
    """
    Complete feed system assembly for air classification.

    This class creates and manages an integrated material handling system
    consisting of four primary components connected in series. The assembly
    handles component instantiation, spatial positioning, port alignment,
    and mesh generation for visualization/simulation.

    Components (in flow order)
    --------------------------
    1. Feed Hopper (FeedHopper)
       - Bulk powder storage with mass flow design
       - Cylindrical body + conical discharge section
       - Hinged lid with T-bar handle for filling
       - Inner skirt seal to prevent dust escape

    2. Rotary Airlock (RotaryAirlock)
       - Pressure seal between hopper and downstream equipment
       - 8-vane rotor for volumetric metering
       - Saddle-joint inlet/outlet connections
       - Prevents air backflow into hopper

    3. Screw Feeder (ScrewFeeder)
       - Controlled volumetric dosing
       - Helical screw in U-trough design
       - Consistent feed rate independent of hopper level
       - Variable speed for rate adjustment

    4. Deagglomerator (Deagglomerator)
       - Breaks up lumps and agglomerates
       - High-speed pin rotor (3 rows × 6 pins)
       - Screen at outlet controls max particle size
       - Ensures uniform particle distribution

    Connection System
    -----------------
    Components are connected via a standardized port system:
    - Each component defines inlet/outlet ConnectionPorts
    - Ports have position, direction, and diameter
    - calculate_alignment() positions components for proper mating
    - Transition connectors seal gaps between components

    Process Flow
    ------------
        HOPPER → AIRLOCK → FEEDER → DEAGGLOMERATOR → (to classifier)
           ↓        ↓         ↓            ↓
        Gravity  Pressure   Controlled   Lump
        feed     seal       metering     breaking

    Coordinate System
    -----------------
    - Origin: At hopper discharge center (Y=0)
    - Y-axis: Vertical (positive up, gravity is -Y)
    - X-axis: Horizontal (screw feeder direction)
    - Z-axis: Lateral (hopper axis for z-rotation airlock)

    Attributes
    ----------
    params : FeedSystemParams
        Configuration parameters for the assembly
    hopper : FeedHopper
        Feed hopper component instance
    airlock : RotaryAirlock
        Rotary airlock component instance
    feeder : ScrewFeeder
        Screw feeder component instance
    deagglomerator : Deagglomerator
        Deagglomerator component instance

    Methods
    -------
    build_mesh()
        Generate combined mesh for all components
    validate_connections()
        Check all port alignments and report issues
    get_component(name)
        Access individual component by name
    get_bounds()
        Get bounding box of entire assembly

    Example
    -------
    >>> from airclassifier.geometry.assembly import create_standard_feed_system
    >>> 
    >>> # Create with defaults
    >>> feed_system = create_standard_feed_system()
    >>> 
    >>> # Build mesh for visualization
    >>> vertices, indices = feed_system.build_mesh()
    >>> 
    >>> # Validate connections
    >>> report = feed_system.validate_connections()
    >>> 
    >>> # Access components
    >>> hopper_volume = feed_system.hopper.params.total_volume
    >>> airlock_capacity = feed_system.airlock.params.volumetric_capacity
    """

    def __init__(self, params: FeedSystemParams = None, device: str = "cpu"):
        """
        Initialize feed system assembly.

        Args:
            params: FeedSystemParams (uses defaults if None)
            device: Warp device for mesh operations
        """
        self.params = params or FeedSystemParams()
        self.device = device

        # Create components
        self._create_components()

        # Mesh data
        self._combined_vertices = None
        self._combined_indices = None
        self._mesh_built = False

    def _create_components(self):
        """
        Create all system components with proper port-to-port positioning.
        
        DYNAMIC PROPORTIONAL SIZING (like classification.py):
        1. Calculate port diameters based on connected components
        2. Calculate transition lengths FIRST based on diameter changes
        3. Position components with proper gaps for transitions
        4. Create transitions that exactly fill the gaps
        """
        # Lazy imports to avoid circular dependency
        from ..components import (
            create_standard_feed_hopper,
            create_standard_rotary_airlock,
            create_standard_screw_feeder,
            create_standard_deagglomerator,
        )

        p = self.params
        flange_gap = p.component_spacing  # Small gap for flange gaskets (5mm)
        
        # ============================================================
        # HELPER: Calculate transition length based on diameter change
        # ============================================================
        def calc_transition_length(d_in: float, d_out: float, min_length: float = 0.05) -> float:
            """
            Calculate proper transition length for smooth flow.
            Uses max 12° half-angle for expansion/contraction.
            """
            max_half_angle = np.radians(12)  # 12° max half-angle
            d_diff = abs(d_in - d_out)
            
            if d_diff > 0.001:
                # Conical transition - length based on angle
                length = (d_diff / 2) / np.tan(max_half_angle)
                return max(length, min_length)
            else:
                # Same diameter - minimum visible length
                return min_length

        # ============================================================
        # 1. FEED HOPPER (top of system) - positioned first as reference
        # ============================================================
        self.hopper = create_standard_feed_hopper(
            capacity_kg=p.hopper_capacity_kg,
            bulk_density=p.bulk_density,
            discharge_diameter=p.hopper_discharge_diameter
        )
        
        self._hopper_position = (
            p.center[0], 
            p.center[1],
            p.center[2]
        )

        # ============================================================
        # 2. ROTARY AIRLOCK - below hopper
        # ============================================================
        from ..components import RotaryAirlockParams, RotaryAirlock
        
        airlock_params = RotaryAirlockParams(
            rotor_diameter=p.airlock_rotor_diameter,
            rotor_length=p.airlock_rotor_diameter * 0.6,
            num_vanes=8,
            vane_thickness=0.005,
            vane_tip_clearance=0.0003,
            inlet_diameter=p.hopper_discharge_diameter,
            outlet_diameter=p.hopper_discharge_diameter * 0.9,
        )
        self.airlock = RotaryAirlock(airlock_params)
        
        # Transition 1: Hopper discharge -> Airlock inlet
        # Both same diameter, use minimum length
        trans1_length = calc_transition_length(
            p.hopper_discharge_diameter,
            p.hopper_discharge_diameter,
            min_length=0.015  # 15mm minimum - tight cylindrical connection
        )
        gap1 = flange_gap + trans1_length + flange_gap
        
        alignment = calculate_alignment(
            source_port=self.hopper.ports['discharge'],
            target_port=self.airlock.ports['inlet'],
            source_position=self._hopper_position,
            gap=gap1,
            align_directions=True
        )
        self._airlock_position = tuple(alignment.position_offset)

        # ============================================================
        # 3. SCREW FEEDER - below airlock
        # ============================================================
        from ..components import ScrewFeederParams, ScrewFeeder

        # Calculate feeder tube geometry first to properly size inlet
        screw_pitch = p.feeder_screw_diameter * 0.8
        trough_clearance = 0.003
        wall_thickness = 0.003
        feeder_trough_radius = p.feeder_screw_diameter / 2 + trough_clearance
        feeder_tube_outer_radius = feeder_trough_radius + wall_thickness

        # CRITICAL: Inlet diameter must fit ON TOP of the tube, not larger than tube
        # Maximum inlet = ~80% of tube diameter for proper visual fit
        max_inlet_dia = feeder_tube_outer_radius * 2 * 0.75
        feeder_inlet_dia = min(airlock_params.outlet_diameter, max_inlet_dia)

        # Outlet diameter for deagglomerator connection
        feeder_outlet_dia = p.deagg_rotor_diameter * 0.4

        feeder_params = ScrewFeederParams(
            screw_diameter=p.feeder_screw_diameter,
            shaft_diameter=p.feeder_screw_diameter * 0.3,
            screw_pitch=screw_pitch,
            flight_thickness=0.003,
            trough_length=screw_pitch * 3,
            trough_clearance=trough_clearance,
            inlet_diameter=feeder_inlet_dia,
            outlet_diameter=feeder_outlet_dia,
        )
        self.feeder = ScrewFeeder(feeder_params)
        
        # Transition 2: Airlock outlet -> Feeder inlet
        # CONICAL REDUCER: airlock outlet (larger) to feeder inlet (smaller)
        trans2_length = calc_transition_length(
            airlock_params.outlet_diameter,
            feeder_inlet_dia,
            min_length=0.025  # 25mm minimum - tight conical reducer
        )
        gap2 = flange_gap + trans2_length + flange_gap
        
        alignment = calculate_alignment(
            source_port=self.airlock.ports['outlet'],
            target_port=self.feeder.ports['inlet'],
            source_position=self._airlock_position,
            gap=gap2,
            align_directions=True
        )
        self._feeder_position = tuple(alignment.position_offset)

        # ============================================================
        # 4. DE-AGGLOMERATOR - below screw feeder
        # ============================================================
        from ..components import DeagglomeratorParams, Deagglomerator
        
        deagg_params = DeagglomeratorParams(
            rotor_diameter=p.deagg_rotor_diameter,
            rotor_length=p.deagg_rotor_diameter * 0.6,
            shaft_diameter=p.deagg_rotor_diameter * 0.2,
            num_pin_rows=3,
            pins_per_row=6,
            pin_diameter=p.deagg_rotor_diameter * 0.05,
            pin_length=p.deagg_rotor_diameter * 0.35,
            housing_diameter=p.deagg_rotor_diameter * 1.3,
            housing_length=p.deagg_rotor_diameter * 0.8,
            screen_diameter=p.deagg_rotor_diameter * 1.1,
            screen_aperture=p.deagg_screen_aperture,
            screen_open_area=0.40,
            inlet_diameter=feeder_outlet_dia,
            outlet_diameter=feeder_outlet_dia * 1.2,
        )
        self.deagglomerator = Deagglomerator(deagg_params)
        
        # Transition 3: Feeder outlet -> Deagglomerator inlet
        # Same diameter - short cylindrical connection
        trans3_length = calc_transition_length(
            feeder_outlet_dia,
            feeder_outlet_dia,  # Same - deagg inlet matches feeder outlet
            min_length=0.020  # 20mm minimum - tight connection
        )
        gap3 = flange_gap + trans3_length + flange_gap
        
        alignment = calculate_alignment(
            source_port=self.feeder.ports['outlet'],
            target_port=self.deagglomerator.ports['inlet'],
            source_position=self._feeder_position,
            gap=gap3,
            align_directions=True
        )
        self._deagglomerator_position = tuple(alignment.position_offset)
        
        # Store transition lengths for connector creation
        self._transition_lengths = {
            'hopper_to_airlock': trans1_length,
            'airlock_to_feeder': trans2_length,
            'feeder_to_deagglomerator': trans3_length,
        }
        
        # 5. Create transition connectors that exactly fill the gaps
        self._create_transition_connectors(flange_gap)

    def _create_transition_connectors(self, flange_gap: float):
        """
        Create proper flanged transition connectors between all component connections.
        
        INDUSTRIAL DESIGN - DYNAMIC PROPORTIONAL SIZING:
        Transitions are created with pre-calculated lengths from _create_components
        to exactly fill the gaps between components.
        
        Each transition:
        1. Uses the pre-calculated length that accounts for diameter changes
        2. Handles diameter differences with CONICAL transitions
        3. Provides flanged ends for bolted, dust-tight connections
        4. Follows the port direction for proper 3D orientation
        
        CONNECTION CHAIN:
        HOPPER → [Transition] → AIRLOCK → [Transition] → FEEDER → [Transition] → DEAGGLOMERATOR
        """
        from ..components.transitions import Transition, TransitionParams
        
        self._transition_connectors = []
        
        def create_transition_connector(
            source_port, source_position,
            target_port, target_position,
            name: str,
            transition_length: float
        ):
            """
            Create a transition connector with the specified length.
            
            The transition starts at the source port (offset by flange_gap)
            and extends for the pre-calculated transition_length.
            """
            # Calculate world positions of ports
            source_world = np.array([
                source_position[i] + source_port.position[i] for i in range(3)
            ])
            target_world = np.array([
                target_position[i] + target_port.position[i] for i in range(3)
            ])
            
            # Get port diameters
            d_source = source_port.diameter
            d_target = target_port.diameter
            
            # Direction from source to target (normalized)
            connection_vec = target_world - source_world
            actual_distance = np.linalg.norm(connection_vec)
            
            if actual_distance > 0.001:
                direction = tuple(connection_vec / actual_distance)
            else:
                # Default to -Y direction (downward)
                direction = (0.0, -1.0, 0.0)
            
            # Transition starts after flange_gap from source port
            dir_array = np.array(direction)
            transition_start = source_world + dir_array * flange_gap
            
            # Create transition with proper orientation and sizing
            transition_params = TransitionParams(
                transition_type="round_to_round",
                inlet_dimensions=(d_source,),
                outlet_dimensions=(d_target,),
                length=transition_length,
                direction=direction,
                center=tuple(transition_start),
                concentric=True,
                flanged=True,
                flange_width=0.015,  # 15mm flange
                flange_thickness=0.006,  # 6mm flange
                wall_thickness=0.003,  # 3mm wall
            )
            
            transition = Transition(transition_params)
            
            return (transition, tuple(transition_start), name)
        
        # ============================================================
        # 1. HOPPER → AIRLOCK Transition
        # ============================================================
        trans1 = create_transition_connector(
            self.hopper.ports['discharge'], self._hopper_position,
            self.airlock.ports['inlet'], self._airlock_position,
            "hopper_to_airlock",
            self._transition_lengths['hopper_to_airlock']
        )
        self._transition_connectors.append(trans1)
        
        # ============================================================
        # 2. AIRLOCK → SCREW FEEDER Transition  
        # ============================================================
        trans2 = create_transition_connector(
            self.airlock.ports['outlet'], self._airlock_position,
            self.feeder.ports['inlet'], self._feeder_position,
            "airlock_to_feeder",
            self._transition_lengths['airlock_to_feeder']
        )
        self._transition_connectors.append(trans2)
        
        # ============================================================
        # 3. SCREW FEEDER → DEAGGLOMERATOR Transition
        # ============================================================
        trans3 = create_transition_connector(
            self.feeder.ports['outlet'], self._feeder_position,
            self.deagglomerator.ports['inlet'], self._deagglomerator_position,
            "feeder_to_deagglomerator",
            self._transition_lengths['feeder_to_deagglomerator']
        )
        self._transition_connectors.append(trans3)

    def build_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build combined mesh for all components including transition connectors.

        Returns:
            Tuple of (vertices, indices)
        """
        all_vertices = []
        all_indices = []
        vertex_offset = 0

        # Helper to add component mesh with position offset
        def add_component_mesh(component, position):
            nonlocal vertex_offset
            verts, idx, _ = component.generate_mesh()

            # Apply position offset
            offset = np.array(position)
            verts_offset = verts + offset

            all_vertices.append(verts_offset)
            all_indices.append(idx + vertex_offset)
            vertex_offset += len(verts)

        # Add main components
        add_component_mesh(self.hopper, self._hopper_position)
        add_component_mesh(self.airlock, self._airlock_position)
        add_component_mesh(self.feeder, self._feeder_position)
        add_component_mesh(self.deagglomerator, self._deagglomerator_position)
        
        # Add flanged transition connectors for dust-tight sealing
        # Format: (transition, position, name)
        for connector_data in self._transition_connectors:
            connector = connector_data[0]
            # Position is already baked into the Transition center parameter
            # so we use (0, 0, 0) offset
            add_component_mesh(connector, (0.0, 0.0, 0.0))

        self._combined_vertices = np.vstack(all_vertices).astype(np.float32)
        self._combined_indices = np.concatenate(all_indices).astype(np.int32)
        self._mesh_built = True

        return self._combined_vertices, self._combined_indices

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get axis-aligned bounding box of the entire system.

        Returns:
            Tuple of (min_corner, max_corner) as numpy arrays
        """
        if not self._mesh_built:
            self.build_mesh()

        min_corner = self._combined_vertices.min(axis=0)
        max_corner = self._combined_vertices.max(axis=0)

        return min_corner, max_corner

    def get_system_extent(self) -> np.ndarray:
        """
        Get system extent (dimensions) in each axis.

        Returns:
            Array of [width, height, depth]
        """
        min_c, max_c = self.get_bounds()
        return max_c - min_c

    def get_component(self, name: str) -> Any:
        """
        Get a specific component by name.

        Args:
            name: Component name ('hopper', 'airlock', 'feeder', 'deagglomerator')

        Returns:
            Component instance
        """
        components = {
            'hopper': self.hopper,
            'airlock': self.airlock,
            'feeder': self.feeder,
            'deagglomerator': self.deagglomerator,
        }
        if name not in components:
            raise KeyError(f"Unknown component: {name}. Available: {list(components.keys())}")
        return components[name]

    def get_component_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Get positions of all components.

        Returns:
            Dictionary of component names to positions
        """
        return {
            'hopper': self._hopper_position,
            'airlock': self._airlock_position,
            'feeder': self._feeder_position,
            'deagglomerator': self._deagglomerator_position,
        }

    def get_feed_rate(self, rpm: float = None) -> float:
        """
        Get feed rate based on feeder settings.

        Args:
            rpm: Feeder RPM (uses default if None)

        Returns:
            Feed rate [kg/h]
        """
        return self.feeder.get_feed_rate(rpm=rpm, bulk_density=self.params.bulk_density)

    def to_warp_mesh(self) -> wp.Mesh:
        """
        Create a Warp mesh from the system geometry.

        Returns:
            wp.Mesh object
        """
        if not self._mesh_built:
            self.build_mesh()

        points = wp.array(self._combined_vertices, dtype=wp.vec3, device=self.device)
        indices = wp.array(self._combined_indices, dtype=wp.int32, device=self.device)

        return wp.Mesh(points=points, indices=indices)

    def validate_connections(self, tolerance: float = None) -> List[Dict[str, Any]]:
        """
        Validate that all component connections are properly aligned.
        
        Args:
            tolerance: Position tolerance [m] (defaults to component_spacing + 0.01)
            
        Returns:
            List of validation results for each connection
        """
        if tolerance is None:
            # Allow for the configured component spacing plus small margin
            tolerance = self.params.component_spacing + 0.01
        
        components = [self.hopper, self.airlock, self.feeder, self.deagglomerator]
        positions = {
            0: self._hopper_position,
            1: self._airlock_position,
            2: self._feeder_position,
            3: self._deagglomerator_position,
        }
        
        # Define expected connections: (comp_a_idx, port_a, comp_b_idx, port_b)
        connections = [
            (0, 'discharge', 1, 'inlet'),    # Hopper -> Airlock
            (1, 'outlet', 2, 'inlet'),       # Airlock -> Feeder
            (2, 'outlet', 3, 'inlet'),       # Feeder -> Deagglomerator
        ]
        
        return validate_assembly_connections(components, positions, connections, tolerance)
    
    def print_connection_report(self):
        """Print detailed connection validation report."""
        results = self.validate_connections()
        print_connection_report(results)
    
    def get_system_outlet(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the final outlet position and direction of the feed system.
        
        This is where material exits the feed system (deagglomerator outlet)
        to connect to the classifier.
        
        Returns:
            Tuple of (position, direction) as numpy arrays
        """
        port = self.deagglomerator.ports['outlet']
        position = port.get_world_position(self._deagglomerator_position)
        direction = port.direction_array
        return position, direction

    def print_summary(self):
        """Print summary of the feed system."""
        p = self.params

        print("=" * 60)
        print("Feed System Assembly Summary")
        print("=" * 60)

        print("\n1. FEED HOPPER")
        print(f"   Capacity:        {p.hopper_capacity_kg:.0f} kg")
        print(f"   Top diameter:    {self.hopper.params.top_diameter * 1000:.0f} mm")
        print(f"   Discharge dia:   {p.hopper_discharge_diameter * 1000:.0f} mm")
        print(f"   Total height:    {self.hopper.params.total_height * 1000:.0f} mm")

        print("\n2. ROTARY AIRLOCK")
        print(f"   Rotor diameter:  {p.airlock_rotor_diameter * 1000:.0f} mm")
        print(f"   Vanes:           {self.airlock.params.num_vanes}")
        print(f"   Capacity:        {self.airlock.params.capacity_kg_h(p.bulk_density):.0f} kg/h")

        print("\n3. SCREW FEEDER")
        print(f"   Screw diameter:  {p.feeder_screw_diameter * 1000:.0f} mm")
        print(f"   Trough length:   {self.feeder.params.trough_length * 1000:.0f} mm")
        print(f"   Design rate:     {p.feeder_target_rate_kg_h:.0f} kg/h")

        print("\n4. DE-AGGLOMERATOR")
        print(f"   Rotor diameter:  {p.deagg_rotor_diameter * 1000:.0f} mm")
        print(f"   Screen aperture: {p.deagg_screen_aperture * 1000:.1f} mm")
        print(f"   Tip speed:       {self.deagglomerator.get_tip_speed():.1f} m/s")

        print("-" * 60)
        extent = self.get_system_extent()
        print(f"System extent: {extent[0]*1000:.0f} x {extent[1]*1000:.0f} x {extent[2]*1000:.0f} mm")

        if self._mesh_built:
            n_verts = len(self._combined_vertices)
            n_tris = len(self._combined_indices) // 3
            print(f"Total mesh:    {n_verts} vertices, {n_tris} triangles")
        print("=" * 60)

    def print_transition_report(self):
        """
        Print detailed report of all transition connectors.
        
        Shows the flanged transitions that seal each connection point.
        DYNAMIC PROPORTIONAL SIZING is used for all transitions.
        """
        print("\n" + "=" * 60)
        print("TRANSITION CONNECTORS (Flanged, Dust-Tight)")
        print("=" * 60)
        
        if not self._transition_connectors:
            print("No transitions created yet.")
            return
        
        for i, (transition, position, name) in enumerate(self._transition_connectors, 1):
            params = transition.params
            d_in = params.inlet_dimensions[0]
            d_out = params.outlet_dimensions[0]
            d_diff = abs(d_in - d_out)
            
            print(f"\n{i}. {name.upper().replace('_', ' → ')}")
            print(f"   Inlet diameter:  {d_in * 1000:.1f} mm")
            print(f"   Outlet diameter: {d_out * 1000:.1f} mm")
            print(f"   Length:          {params.length * 1000:.1f} mm")
            print(f"   Direction:       ({params.direction[0]:.2f}, {params.direction[1]:.2f}, {params.direction[2]:.2f})")
            print(f"   Flanged:         {params.flanged}")
            
            if d_diff > 0.001:  # Diameters differ
                # Calculate actual half-angle
                half_angle = np.degrees(np.arctan((d_diff / 2) / params.length))
                if d_in > d_out:
                    print(f"   Type:            CONICAL REDUCER ({half_angle:.1f}° half-angle)")
                else:
                    print(f"   Type:            CONICAL EXPANDER ({half_angle:.1f}° half-angle)")
            else:
                print(f"   Type:            CYLINDRICAL (same diameter)")
        
        print("\n" + "-" * 60)
        print("All connections use DYNAMIC PROPORTIONAL SIZING:")
        print("  - Conical transitions for diameter changes")
        print("  - Max 12° half-angle for smooth flow")
        print("  - Flanged ends for dust-tight sealing")
        print("=" * 60)

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


def create_standard_feed_system(device: str = "cpu") -> FeedSystemAssembly:
    """
    Create a standard feed system with default parameters.

    Args:
        device: Warp device

    Returns:
        FeedSystemAssembly instance
    """
    return FeedSystemAssembly(device=device)


def create_feed_system_for_throughput(
    throughput_kg_h: float = 500,
    device: str = "cpu"
) -> FeedSystemAssembly:
    """
    Create a feed system sized for given throughput.

    Args:
        throughput_kg_h: Design throughput [kg/h]
        device: Warp device

    Returns:
        FeedSystemAssembly configured for given throughput
    """
    # Scale parameters based on throughput
    scale = (throughput_kg_h / 500) ** 0.5  # Square root scaling

    params = FeedSystemParams(
        hopper_capacity_kg=throughput_kg_h * 1.0,  # 1 hour buffer
        hopper_discharge_diameter=0.15 * scale,
        airlock_rotor_diameter=0.20 * scale,
        feeder_screw_diameter=0.10 * scale,
        feeder_target_rate_kg_h=throughput_kg_h,
        deagg_rotor_diameter=0.20 * scale,
        deagg_screen_aperture=0.002,  # Fixed screen size
    )

    return FeedSystemAssembly(params, device=device)
