"""
Particle module for cyclone air classifier.

Provides material definitions, particle systems, size distributions,
drag models, and particle interaction handling.
"""

from .material import (
    ParticleMaterial,
    MaterialProperties,
    SizeDistributionParams,
    SizeDistributionType,
    WarpMaterialProps,
    material_to_warp,
    particle_volume,
    particle_mass,
    particle_projected_area,
)

# Sub-modules
from . import drag_models
from . import interactions

__all__ = [
    "ParticleMaterial",
    "MaterialProperties",
    "SizeDistributionParams",
    "SizeDistributionType",
    "WarpMaterialProps",
    "material_to_warp",
    "particle_volume",
    "particle_mass",
    "particle_projected_area",
    # Sub-modules
    "drag_models",
    "interactions",
]
