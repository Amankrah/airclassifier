"""
Geometry module for cyclone air classifier.

This module provides all geometric primitives, components, and assembly
functionality for building cyclone geometries.
"""

from .primitives import (
    Cylinder, CylinderParams,
    Cone, ConeParams,
    Tube, TubeParams,
    RectangularDuct, RectangularDuctParams,
)

from .components import (
    CycloneBody, CycloneBodyParams,
    TangentialInlet, InletParams,
    VortexFinder, VortexFinderParams,
    DustOutlet, DustOutletParams,
    Overflow, OverflowParams,
)

from .assembly import (
    CycloneAssembly,
    CycloneGeometryParams,
    create_standard_cyclone,
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
    # Components
    "CycloneBody", "CycloneBodyParams",
    "TangentialInlet", "InletParams",
    "VortexFinder", "VortexFinderParams",
    "DustOutlet", "DustOutletParams",
    "Overflow", "OverflowParams",
    # Assembly
    "CycloneAssembly",
    "CycloneGeometryParams",
    "create_standard_cyclone",
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
