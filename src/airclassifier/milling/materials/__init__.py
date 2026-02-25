"""
Milling Materials Module
========================

Material-specific breakage properties for different feedstocks.
"""

from .breakage_properties import (
    MaterialBreakageProperties,
    MATERIAL_LIBRARY,
    get_material_properties,
    compute_particle_mass,
)

__all__ = [
    "MaterialBreakageProperties",
    "MATERIAL_LIBRARY",
    "get_material_properties",
    "compute_particle_mass",
]
