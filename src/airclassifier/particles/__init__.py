"""
Particle module for cyclone air classifier.

Provides material definitions, particle systems, size distributions,
drag models, and particle interaction handling.

Supports protein separation from plant-based food powders:
- Yellow peas (Pisum sativum)
- Faba beans (Vicia faba)  
- Oats (Avena sativa)
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

from .particle_system import (
    # Main classes
    WarpParticleSystem,
    ParticleSystemConfig,
    ParticleType,
    IntegrationMethod,
    CollisionState,
    # Factory functions
    create_particle_system,
    create_yellow_pea_simulation,
    create_faba_bean_simulation,
    create_oat_simulation,
)

# Sub-modules
from . import drag_models
from . import interactions

__all__ = [
    # Material definitions
    "ParticleMaterial",
    "MaterialProperties",
    "SizeDistributionParams",
    "SizeDistributionType",
    "WarpMaterialProps",
    "material_to_warp",
    "particle_volume",
    "particle_mass",
    "particle_projected_area",
    # Particle system
    "WarpParticleSystem",
    "ParticleSystemConfig",
    "ParticleType",
    "IntegrationMethod",
    "CollisionState",
    "create_particle_system",
    "create_yellow_pea_simulation",
    "create_faba_bean_simulation",
    "create_oat_simulation",
    # Sub-modules
    "drag_models",
    "interactions",
]
