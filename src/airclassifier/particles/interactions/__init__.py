"""
Particle interaction module.

Provides collision handling for:
- Particle-wall interactions (cyclone walls, ducts, hoppers)
- Particle-particle interactions (optional)

Reusable across physics simulations:
- Feed flow physics (hopper, airlock, screw housing)
- Classification flow physics (zigzag, cyclones, ductwork)
- Air flow physics (filter, blower, damper)

Example usage:
    from airclassifier.particles.interactions import WallCollisionHandler
    
    handler = WallCollisionHandler(restitution=0.3, friction=0.4)
    handler.add_cylinder(center=(0, 1, 0), axis=1, radius=0.3, length=1.0)
    handler.add_cone(apex=(0, 0, 0), axis=1, half_angle_deg=30, height=0.5)
    handler.process(positions, velocities, diameters, is_active)
"""

from .particle_wall import (
    # Configuration
    WallCollisionParams,
    WallSection,
    # Cyclone-specific SDF collisions
    handle_wall_collisions_sdf,
    detect_wall_collisions_mesh,
    cyclone_sdf_and_normal,
    handle_wall_collision_analytical,
    # Core functions
    reflect_velocity,
    compute_impact_velocity,
    estimate_collision_frequency,
    # Generic wall collision functions (NEW)
    cylindrical_wall_collision,
    conical_wall_collision,
    rectangular_duct_collision,
    handle_generic_wall_collisions,
    # Wall collision manager (NEW)
    WallCollisionHandler,
)

from .particle_particle import (
    ParticleCollisionParams,
    ParticleCollisionHandler,
    particle_particle_collision,
    detect_particle_collisions,
    separate_overlapping_particles,
)

__all__ = [
    # Wall collision configuration
    "WallCollisionParams",
    "WallSection",
    # Cyclone-specific collisions
    "handle_wall_collisions_sdf",
    "detect_wall_collisions_mesh",
    "cyclone_sdf_and_normal",
    "handle_wall_collision_analytical",
    # Core functions
    "reflect_velocity",
    "compute_impact_velocity",
    "estimate_collision_frequency",
    # Generic wall collisions (NEW)
    "cylindrical_wall_collision",
    "conical_wall_collision",
    "rectangular_duct_collision",
    "handle_generic_wall_collisions",
    # Wall collision manager (NEW)
    "WallCollisionHandler",
    # Particle-particle collisions
    "ParticleCollisionParams",
    "ParticleCollisionHandler",
    "particle_particle_collision",
    "detect_particle_collisions",
    "separate_overlapping_particles",
]
