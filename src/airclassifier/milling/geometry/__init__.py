"""
Milling Geometry Module
=======================

Geometry generation for the hammer mill machine.
Provides component-by-component mesh generation with animation support.

Main entry points:
    - create_hammer_mill_machine(): Factory for complete assemblies
    - build_hammer_mill_meshes(): Direct mesh generation
    - HammerMillMachineAssembly: Full assembly class

Components:
    - Rotor (animated)
    - Hammers (animated)
    - Screen (static)
    - Housing (static)
    - Feed Chute (static)
    - Drive (static)
"""

from .machine import (
    create_hammer_mill_machine,
    get_default_hammer_mill_meshes,
    HammerMillMachineAssembly,
    COMPONENT_COLORS,
)
from .assembly import build_hammer_mill_meshes, create_hammer_mill_assembly
from .components import (
    RotorParams, RotorGeometry,
    HammerParams, HammerGeometry,
    ScreenParams, ScreenGeometry,
    HousingParams, HousingGeometry,
    FeedChuteParams, FeedChuteGeometry,
    DriveParams, DriveGeometry,
)

__all__ = [
    # Factory functions
    "create_hammer_mill_machine",
    "get_default_hammer_mill_meshes",
    "build_hammer_mill_meshes",
    "create_hammer_mill_assembly",
    # Assembly
    "HammerMillMachineAssembly",
    "COMPONENT_COLORS",
    # Component params
    "RotorParams",
    "HammerParams",
    "ScreenParams",
    "HousingParams",
    "FeedChuteParams",
    "DriveParams",
    # Component geometries
    "RotorGeometry",
    "HammerGeometry",
    "ScreenGeometry",
    "HousingGeometry",
    "FeedChuteGeometry",
    "DriveGeometry",
]
