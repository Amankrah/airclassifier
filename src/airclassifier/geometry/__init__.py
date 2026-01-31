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
)

from .assembly import (
    # Single cyclone
    CycloneAssembly,
    CycloneGeometryParams,
    create_standard_cyclone,
    # Complete classification system (Phase 1)
    ClassificationSystemAssembly,
    ClassificationSystemParams,
    create_standard_classification_system,
    create_protein_separation_system,
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
    # Single Cyclone Assembly
    "CycloneAssembly",
    "CycloneGeometryParams",
    "create_standard_cyclone",
    # Complete System Assembly
    "ClassificationSystemAssembly",
    "ClassificationSystemParams",
    "create_standard_classification_system",
    "create_protein_separation_system",
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
