"""
Particle interaction module.

Provides collision handling for:
- Particle-wall interactions (cyclone walls)
- Particle-particle interactions (optional)
"""

from .particle_wall import (
    WallCollisionParams,
    handle_wall_collisions_sdf,
    detect_wall_collisions_mesh,
    reflect_velocity,
    compute_impact_velocity,
    estimate_collision_frequency,
)

from .particle_particle import (
    ParticleCollisionParams,
    ParticleCollisionHandler,
    particle_particle_collision,
    detect_particle_collisions,
    separate_overlapping_particles,
)

__all__ = [
    # Wall collisions
    "WallCollisionParams",
    "handle_wall_collisions_sdf",
    "detect_wall_collisions_mesh",
    "reflect_velocity",
    "compute_impact_velocity",
    "estimate_collision_frequency",
    # Particle-particle collisions
    "ParticleCollisionParams",
    "ParticleCollisionHandler",
    "particle_particle_collision",
    "detect_particle_collisions",
    "separate_overlapping_particles",
]
