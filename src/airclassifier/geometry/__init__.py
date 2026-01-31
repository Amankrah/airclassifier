"""
Geometry module for cyclone air classifier.

This module provides all geometric primitives, components, and assembly
functionality for building cyclone geometries and complete classification systems.
"""

from .primitives import (
    Cylinder, CylinderParams,
    Cone, ConeParams,
    Tube, TubeParams,
    RectangularDuct, RectangularDuctParams,
)

from .components import (
    # Core cyclone components
    CycloneBody, CycloneBodyParams,
    TangentialInlet, InletParams,
    VortexFinder, VortexFinderParams,
    DustOutlet, DustOutletParams,
    Overflow, OverflowParams,
    # Phase 1 Components
    ZigzagClassifier, ZigzagClassifierParams, create_standard_zigzag_classifier,
    VenturiEducator, VenturiEducatorParams, create_standard_venturi_eductor,
    MultiCycloneSystem, MultiCycloneParams, CycloneStageParams,
    create_protein_separation_cyclones, create_two_stage_cyclones,
    BagFilter, BagFilterParams, create_standard_bag_filter,
    # Phase 2 Components - Feed System
    FeedHopper, FeedHopperParams, create_standard_feed_hopper,
    RotaryAirlock, RotaryAirlockParams, create_standard_rotary_airlock,
    ScrewFeeder, ScrewFeederParams, create_standard_screw_feeder,
    Deagglomerator, DeagglomeratorParams, create_standard_deagglomerator,
    # Phase 3 Components - Air System
    CentrifugalBlower, CentrifugalBlowerParams, create_standard_centrifugal_blower,
    InletAirFilter, InletAirFilterParams, create_standard_inlet_filter,
    FlowDamper, DamperParams, create_standard_damper,
    # Phase 4 Components - Ductwork
    RoundDuct, RoundDuctParams, create_standard_round_duct,
    RectangularDuct as RectDuct, RectangularDuctParams as RectDuctParams, 
    create_standard_rectangular_duct, create_duct_for_flow,
    Transition, TransitionParams, create_round_reducer,
    create_round_to_rect_transition, create_rect_to_round_transition,
    Elbow, ElbowParams, create_90_degree_elbow, create_45_degree_elbow,
    create_mitered_elbow, create_elbow_with_vanes,
    DiverterValve, DiverterValveParams, create_flap_diverter,
    create_rotating_diverter, create_plug_diverter, create_diverter_for_classifier,
    # Phase 5 Components - Safety
    ExplosionVent, ExplosionVentParams, create_rupture_panel,
    create_hinged_explosion_door, create_recoil_vent, calculate_vent_area,
    GroundingPoint, GroundingPointParams, create_weld_stud_ground,
    create_threaded_ground, create_grounding_system,
    # Phase 5 Components - Instrumentation
    PressurePort, PressurePortParams, create_flush_pressure_port,
    create_extended_pressure_port, create_averaging_pressure_port,
    TemperaturePort, TemperaturePortParams, create_threaded_thermowell,
    create_flanged_thermowell, create_weld_thermowell,
    SamplePort, SamplePortParams, create_ball_valve_sample_port, create_isokinetic_sample_port,
    SightGlass, SightGlassParams, create_standard_sight_glass, create_illuminated_sight_glass,
)

from .assembly import (
    # Single cyclone
    CycloneAssembly,
    CycloneGeometryParams,
    create_standard_cyclone,
    # Classification System (Phase 1)
    ClassificationSystemAssembly,
    ClassificationSystemParams,
    create_standard_classification_system,
    create_protein_separation_system,
    # Feed System (Phase 2)
    FeedSystemAssembly,
    FeedSystemParams,
    create_standard_feed_system,
    create_feed_system_for_throughput,
    # Air System (Phase 3)
    AirSystemAssembly,
    AirSystemParams,
    create_standard_air_system,
    create_air_system_for_classifier,
    # Ductwork System (Phase 4)
    DuctworkSystemAssembly,
    DuctworkSystemParams,
    create_standard_ductwork,
    create_ductwork_for_classifier,
    create_simple_duct_run,
    # Safety & Instrumentation System (Phase 5)
    SafetyInstrumentationAssembly,
    SafetyInstrumentationParams,
    create_standard_safety_instrumentation,
    create_minimal_instrumentation,
    create_full_instrumentation,
)

from .sdf import (
    CycloneSDF,
    CycloneSDFParams,
    SDFField,
    create_cyclone_sdf,
    visualize_sdf_slice,
)

from .mesh_generator import (
    GridParams,
    MeshGenerator,
    generate_cyclone_mesh,
    export_mesh_vtk,
    export_mesh_stl,
    create_sampling_points,
)

__all__ = [
    # Primitives
    "Cylinder", "CylinderParams",
    "Cone", "ConeParams",
    "Tube", "TubeParams",
    "RectangularDuct", "RectangularDuctParams",
    # Core Components
    "CycloneBody", "CycloneBodyParams",
    "TangentialInlet", "InletParams",
    "VortexFinder", "VortexFinderParams",
    "DustOutlet", "DustOutletParams",
    "Overflow", "OverflowParams",
    # Phase 1 Components
    "ZigzagClassifier", "ZigzagClassifierParams", "create_standard_zigzag_classifier",
    "VenturiEducator", "VenturiEducatorParams", "create_standard_venturi_eductor",
    "MultiCycloneSystem", "MultiCycloneParams", "CycloneStageParams",
    "create_protein_separation_cyclones", "create_two_stage_cyclones",
    "BagFilter", "BagFilterParams", "create_standard_bag_filter",
    # Phase 2 Components - Feed System
    "FeedHopper", "FeedHopperParams", "create_standard_feed_hopper",
    "RotaryAirlock", "RotaryAirlockParams", "create_standard_rotary_airlock",
    "ScrewFeeder", "ScrewFeederParams", "create_standard_screw_feeder",
    "Deagglomerator", "DeagglomeratorParams", "create_standard_deagglomerator",
    # Phase 3 Components - Air System
    "CentrifugalBlower", "CentrifugalBlowerParams", "create_standard_centrifugal_blower",
    "InletAirFilter", "InletAirFilterParams", "create_standard_inlet_filter",
    "FlowDamper", "DamperParams", "create_standard_damper",
    # Single Cyclone Assembly
    "CycloneAssembly",
    "CycloneGeometryParams",
    "create_standard_cyclone",
    # Classification System Assembly (Phase 1)
    "ClassificationSystemAssembly",
    "ClassificationSystemParams",
    "create_standard_classification_system",
    "create_protein_separation_system",
    # Feed System Assembly (Phase 2)
    "FeedSystemAssembly",
    "FeedSystemParams",
    "create_standard_feed_system",
    "create_feed_system_for_throughput",
    # Air System Assembly (Phase 3)
    "AirSystemAssembly",
    "AirSystemParams",
    "create_standard_air_system",
    "create_air_system_for_classifier",
    # Phase 4 Components - Ductwork
    "RoundDuct", "RoundDuctParams", "create_standard_round_duct",
    "RectDuct", "RectDuctParams", 
    "create_standard_rectangular_duct", "create_duct_for_flow",
    "Transition", "TransitionParams", "create_round_reducer",
    "create_round_to_rect_transition", "create_rect_to_round_transition",
    "Elbow", "ElbowParams", "create_90_degree_elbow", "create_45_degree_elbow",
    "create_mitered_elbow", "create_elbow_with_vanes",
    "DiverterValve", "DiverterValveParams", "create_flap_diverter",
    "create_rotating_diverter", "create_plug_diverter", "create_diverter_for_classifier",
    # Ductwork System Assembly (Phase 4)
    "DuctworkSystemAssembly",
    "DuctworkSystemParams",
    "create_standard_ductwork",
    "create_ductwork_for_classifier",
    "create_simple_duct_run",
    # Phase 5 Components - Safety
    "ExplosionVent", "ExplosionVentParams", "create_rupture_panel",
    "create_hinged_explosion_door", "create_recoil_vent", "calculate_vent_area",
    "GroundingPoint", "GroundingPointParams", "create_weld_stud_ground",
    "create_threaded_ground", "create_grounding_system",
    # Phase 5 Components - Instrumentation
    "PressurePort", "PressurePortParams", "create_flush_pressure_port",
    "create_extended_pressure_port", "create_averaging_pressure_port",
    "TemperaturePort", "TemperaturePortParams", "create_threaded_thermowell",
    "create_flanged_thermowell", "create_weld_thermowell",
    "SamplePort", "SamplePortParams", "create_ball_valve_sample_port", "create_isokinetic_sample_port",
    "SightGlass", "SightGlassParams", "create_standard_sight_glass", "create_illuminated_sight_glass",
    # Safety & Instrumentation System Assembly (Phase 5)
    "SafetyInstrumentationAssembly",
    "SafetyInstrumentationParams",
    "create_standard_safety_instrumentation",
    "create_minimal_instrumentation",
    "create_full_instrumentation",
    # SDF
    "CycloneSDF",
    "CycloneSDFParams",
    "SDFField",
    "create_cyclone_sdf",
    "visualize_sdf_slice",
    # Mesh Generator
    "GridParams",
    "MeshGenerator",
    "generate_cyclone_mesh",
    "export_mesh_vtk",
    "export_mesh_stl",
    "create_sampling_points",
]
