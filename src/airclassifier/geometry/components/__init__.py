"""
Air classifier system components.

This module provides components for complete air classification systems:

Cyclone Components:
- CycloneBody: Main body (cylinder + cone)
- TangentialInlet: Particle/air entry
- VortexFinder: Gas outlet tube
- DustOutlet: Coarse particle collection
- Overflow: Fine particle exit tracking

Phase 1 - Core Classification:
- ZigzagClassifier: Primary particle separation device
- VenturiEducator: Particle entrainment into airstream
- MultiCycloneSystem: Staged cyclone collection
- BagFilter: Fine particle collection

Phase 2 - Feed System:
- FeedHopper: Powder storage and discharge
- RotaryAirlock: Pressure sealing valve
- ScrewFeeder: Controlled powder dosing
- Deagglomerator: Lump breaking

Phase 3 - Air System:
- CentrifugalBlower: Air supply fan
- InletAirFilter: Clean air filtration
- FlowDamper: Flow control valve

Phase 4 - Ductwork:
- RoundDuct: Circular duct sections
- RectangularDuct: Rectangular duct sections
- Transition: Duct transitions (round-to-round, round-to-rect, etc.)
- Elbow: Duct bends and elbows
- DiverterValve: Y-type flow diverters

Phase 5 - Safety & Instrumentation:
- ExplosionVent: Rupture panels, hinged doors, recoil vents
- GroundingPoint: Static grounding/bonding studs
- PressurePort: Pressure transmitter connections
- TemperaturePort: Thermowells for temperature measurement
- SamplePort: Sample extraction ports
- SightGlass: Visual inspection windows
"""

from .cyclone_body import CycloneBody, CycloneBodyParams, cyclone_body_sdf
from .inlet import TangentialInlet, InletParams
from .vortex_finder import VortexFinder, VortexFinderParams, vortex_finder_sdf
from .dust_outlet import DustOutlet, DustOutletParams, is_in_dust_outlet
from .overflow import Overflow, OverflowParams, is_in_overflow, check_overflow_particles

# Phase 1 Components
from .zigzag_classifier import (
    ZigzagClassifier,
    ZigzagClassifierParams,
    create_standard_zigzag_classifier,
)
from .venturi_eductor import (
    VenturiEducator,
    VenturiEducatorParams,
    create_standard_venturi_eductor,
)
from .multi_cyclone import (
    MultiCycloneSystem,
    MultiCycloneParams,
    CycloneStageParams,
    create_protein_separation_cyclones,
    create_two_stage_cyclones,
)
from .bag_filter import (
    BagFilter,
    BagFilterParams,
    create_standard_bag_filter,
)

# Phase 2 Components - Feed System
from .feed_hopper import (
    FeedHopper,
    FeedHopperParams,
    create_standard_feed_hopper,
)
from .rotary_airlock import (
    RotaryAirlock,
    RotaryAirlockParams,
    create_standard_rotary_airlock,
)
from .screw_feeder import (
    ScrewFeeder,
    ScrewFeederParams,
    create_standard_screw_feeder,
)
from .deagglomerator import (
    Deagglomerator,
    DeagglomeratorParams,
    create_standard_deagglomerator,
)

# Phase 3 Components - Air System
from .centrifugal_blower import (
    CentrifugalBlower,
    CentrifugalBlowerParams,
    create_standard_centrifugal_blower,
)
from .air_filter import (
    InletAirFilter,
    InletAirFilterParams,
    create_standard_inlet_filter,
)
from .damper import (
    FlowDamper,
    DamperParams,
    create_standard_damper,
)

# Phase 4 Components - Ductwork
from .ductwork import (
    RoundDuct,
    RoundDuctParams,
    RectangularDuct,
    RectangularDuctParams,
    create_standard_round_duct,
    create_standard_rectangular_duct,
    create_duct_for_flow,
)
from .transitions import (
    Transition,
    TransitionParams,
    create_round_reducer,
    create_round_to_rect_transition,
    create_rect_to_round_transition,
)
from .elbows import (
    Elbow,
    ElbowParams,
    create_90_degree_elbow,
    create_45_degree_elbow,
    create_mitered_elbow,
    create_elbow_with_vanes,
)
from .diverter import (
    DiverterValve,
    DiverterValveParams,
    create_flap_diverter,
    create_rotating_diverter,
    create_plug_diverter,
    create_diverter_for_classifier,
)

# Phase 5 Components - Safety
from .safety import (
    ExplosionVent,
    ExplosionVentParams,
    create_rupture_panel,
    create_hinged_explosion_door,
    create_recoil_vent,
    calculate_vent_area,
    GroundingPoint,
    GroundingPointParams,
    create_weld_stud_ground,
    create_threaded_ground,
    create_grounding_system,
)

# Phase 5 Components - Instrumentation
from .instrumentation import (
    PressurePort,
    PressurePortParams,
    create_flush_pressure_port,
    create_extended_pressure_port,
    create_averaging_pressure_port,
    TemperaturePort,
    TemperaturePortParams,
    create_threaded_thermowell,
    create_flanged_thermowell,
    create_weld_thermowell,
    SamplePort,
    SamplePortParams,
    create_ball_valve_sample_port,
    create_isokinetic_sample_port,
    SightGlass,
    SightGlassParams,
    create_standard_sight_glass,
    create_illuminated_sight_glass,
)

__all__ = [
    # Cyclone Body
    "CycloneBody",
    "CycloneBodyParams",
    "cyclone_body_sdf",
    # Inlet
    "TangentialInlet",
    "InletParams",
    # Vortex Finder
    "VortexFinder",
    "VortexFinderParams",
    "vortex_finder_sdf",
    # Dust Outlet
    "DustOutlet",
    "DustOutletParams",
    "is_in_dust_outlet",
    # Overflow
    "Overflow",
    "OverflowParams",
    "is_in_overflow",
    "check_overflow_particles",
    # Zigzag Classifier
    "ZigzagClassifier",
    "ZigzagClassifierParams",
    "create_standard_zigzag_classifier",
    # Venturi Eductor
    "VenturiEducator",
    "VenturiEducatorParams",
    "create_standard_venturi_eductor",
    # Multi-Cyclone System
    "MultiCycloneSystem",
    "MultiCycloneParams",
    "CycloneStageParams",
    "create_protein_separation_cyclones",
    "create_two_stage_cyclones",
    # Bag Filter
    "BagFilter",
    "BagFilterParams",
    "create_standard_bag_filter",
    # Feed Hopper
    "FeedHopper",
    "FeedHopperParams",
    "create_standard_feed_hopper",
    # Rotary Airlock
    "RotaryAirlock",
    "RotaryAirlockParams",
    "create_standard_rotary_airlock",
    # Screw Feeder
    "ScrewFeeder",
    "ScrewFeederParams",
    "create_standard_screw_feeder",
    # Deagglomerator
    "Deagglomerator",
    "DeagglomeratorParams",
    "create_standard_deagglomerator",
    # Centrifugal Blower
    "CentrifugalBlower",
    "CentrifugalBlowerParams",
    "create_standard_centrifugal_blower",
    # Inlet Air Filter
    "InletAirFilter",
    "InletAirFilterParams",
    "create_standard_inlet_filter",
    # Flow Damper
    "FlowDamper",
    "DamperParams",
    "create_standard_damper",
    # Round Duct
    "RoundDuct",
    "RoundDuctParams",
    "create_standard_round_duct",
    # Rectangular Duct
    "RectangularDuct",
    "RectangularDuctParams",
    "create_standard_rectangular_duct",
    "create_duct_for_flow",
    # Transition
    "Transition",
    "TransitionParams",
    "create_round_reducer",
    "create_round_to_rect_transition",
    "create_rect_to_round_transition",
    # Elbow
    "Elbow",
    "ElbowParams",
    "create_90_degree_elbow",
    "create_45_degree_elbow",
    "create_mitered_elbow",
    "create_elbow_with_vanes",
    # Diverter Valve
    "DiverterValve",
    "DiverterValveParams",
    "create_flap_diverter",
    "create_rotating_diverter",
    "create_plug_diverter",
    "create_diverter_for_classifier",
    # Explosion Vent
    "ExplosionVent",
    "ExplosionVentParams",
    "create_rupture_panel",
    "create_hinged_explosion_door",
    "create_recoil_vent",
    "calculate_vent_area",
    # Grounding Point
    "GroundingPoint",
    "GroundingPointParams",
    "create_weld_stud_ground",
    "create_threaded_ground",
    "create_grounding_system",
    # Pressure Port
    "PressurePort",
    "PressurePortParams",
    "create_flush_pressure_port",
    "create_extended_pressure_port",
    "create_averaging_pressure_port",
    # Temperature Port
    "TemperaturePort",
    "TemperaturePortParams",
    "create_threaded_thermowell",
    "create_flanged_thermowell",
    "create_weld_thermowell",
    # Sample Port
    "SamplePort",
    "SamplePortParams",
    "create_ball_valve_sample_port",
    "create_isokinetic_sample_port",
    # Sight Glass
    "SightGlass",
    "SightGlassParams",
    "create_standard_sight_glass",
    "create_illuminated_sight_glass",
]
