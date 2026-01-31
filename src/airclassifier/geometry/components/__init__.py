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
]
