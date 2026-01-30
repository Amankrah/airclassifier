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
]
