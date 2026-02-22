"""
Milling Materials Module
========================

Material-specific breakage properties for different feedstocks.
"""

from .breakage_properties import (
    MaterialBreakageProperties,
    MATERIAL_LIBRARY,
    get_material_properties,
)

__all__ = [
    "MaterialBreakageProperties",
    "MATERIAL_LIBRARY",
    "get_material_properties",
]
