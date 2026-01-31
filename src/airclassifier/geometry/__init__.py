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
