"""
Particle module for cyclone air classifier.

Provides material definitions, particle systems, size distributions,
drag models, and particle interaction handling.

Supports protein separation from plant-based food powders:
- Yellow peas (Pisum sativum)
- Faba beans (Vicia faba)  
- Oats (Avena sativa)

Reusable across physics simulations:
- Feed flow physics (gravity-driven flow)
- Air flow physics (SPH air flow)
- Classification flow physics (particle separation)

Example usage:
    # Create a particle material
    from airclassifier.particles import ParticleMaterial, FluidConfig
    
    flour = ParticleMaterial.create_food_powder("yellow_pea", "whole")
    air = FluidConfig.air_at_stp()
    
    # Create particle population
    from airclassifier.particles import create_whole_flour_population
    material, diameters, densities, sphericities, types = create_whole_flour_population(
        source="yellow_pea", num_particles=5000
    )
"""

from .material import (
    # Material classes
    ParticleMaterial,
    MaterialProperties,
    SizeDistributionParams,
    SizeDistributionType,
    # Warp structures
    WarpMaterialProps,
    material_to_warp,
    particle_volume,
    particle_mass,
    particle_projected_area,
    # Shared configurations (NEW)
    FluidConfig,
    ParticlePhysicsConfig,
    # Population factory functions (NEW)
    create_particle_population,
    create_bimodal_population,
    create_whole_flour_population,
)

from .particle_system import (
    # Main classes
    WarpParticleSystem,
    ParticleSystemConfig,
    ParticleType,
    IntegrationMethod,
    CollisionState,
    # Warp structures
    WarpParticleData,
    WarpFluidParams,
    WarpDomainBounds,
    # Warp functions (for physics modules)
    compute_particle_reynolds,
    drag_coefficient_schiller_naumann,
    drag_coefficient_haider_levenspiel,
    compute_drag_acceleration,
    compute_centrifugal_acceleration,
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
    # Shared configurations (NEW)
    "FluidConfig",
    "ParticlePhysicsConfig",
    # Population factories (NEW)
    "create_particle_population",
    "create_bimodal_population",
    "create_whole_flour_population",
    # Particle system
    "WarpParticleSystem",
    "ParticleSystemConfig",
    "ParticleType",
    "IntegrationMethod",
    "CollisionState",
    # Warp structures
    "WarpParticleData",
    "WarpFluidParams",
    "WarpDomainBounds",
    # Warp physics functions
    "compute_particle_reynolds",
    "drag_coefficient_schiller_naumann",
    "drag_coefficient_haider_levenspiel",
    "compute_drag_acceleration",
    "compute_centrifugal_acceleration",
    # Factory functions
    "create_particle_system",
    "create_yellow_pea_simulation",
    "create_faba_bean_simulation",
    "create_oat_simulation",
    # Sub-modules
    "drag_models",
    "interactions",
]
