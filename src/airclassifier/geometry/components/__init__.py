"""
Cyclone air classifier components.

This module provides the individual components that make up a cyclone:
- CycloneBody: Main body (cylinder + cone)
- TangentialInlet: Particle/air entry
- VortexFinder: Gas outlet tube
- DustOutlet: Coarse particle collection
- Overflow: Fine particle exit tracking
"""

from .cyclone_body import CycloneBody, CycloneBodyParams, cyclone_body_sdf
from .inlet import TangentialInlet, InletParams
from .vortex_finder import VortexFinder, VortexFinderParams, vortex_finder_sdf
from .dust_outlet import DustOutlet, DustOutletParams, is_in_dust_outlet
from .overflow import Overflow, OverflowParams, is_in_overflow, check_overflow_particles

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
]
