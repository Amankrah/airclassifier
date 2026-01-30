"""
Force calculations for particle dynamics in cyclone air classifier.

This module provides implementations of various forces acting on particles:
- Drag force (fluid resistance)
- Gravitational force (including buoyancy)
- Centrifugal force (from swirling motion)
- Virtual mass force (added mass effects)
"""

from .drag import (
    DragModel,
    particle_reynolds_number,
    drag_coefficient_stokes,
    drag_coefficient_schiller_naumann,
    drag_coefficient_haider_levenspiel,
    drag_force,
    wp_drag_force_schiller_naumann,
    wp_drag_coefficient_schiller_naumann,
    compute_drag_forces,
    compute_drag_accelerations,
)

from .gravity import (
    gravity_force,
    gravity_acceleration,
    terminal_velocity_stokes,
    terminal_velocity_intermediate,
    wp_gravity_force,
    wp_gravity_acceleration,
    compute_gravity_forces,
    add_gravity_acceleration,
)

from .centrifugal import (
    centrifugal_force,
    centrifugal_acceleration,
    separation_number,
    wp_centrifugal_acceleration,
    wp_centrifugal_force,
    compute_centrifugal_accelerations,
    add_centrifugal_acceleration,
)

from .virtual_mass import (
    C_VM_SPHERE,
    virtual_mass_force,
    virtual_mass_coefficient,
    effective_particle_mass,
    is_virtual_mass_significant,
    wp_virtual_mass_acceleration,
    compute_virtual_mass_correction,
)

__all__ = [
    # Drag
    "DragModel",
    "particle_reynolds_number",
    "drag_coefficient_stokes",
    "drag_coefficient_schiller_naumann",
    "drag_coefficient_haider_levenspiel",
    "drag_force",
    "wp_drag_force_schiller_naumann",
    "wp_drag_coefficient_schiller_naumann",
    "compute_drag_forces",
    "compute_drag_accelerations",
    # Gravity
    "gravity_force",
    "gravity_acceleration",
    "terminal_velocity_stokes",
    "terminal_velocity_intermediate",
    "wp_gravity_force",
    "wp_gravity_acceleration",
    "compute_gravity_forces",
    "add_gravity_acceleration",
    # Centrifugal
    "centrifugal_force",
    "centrifugal_acceleration",
    "separation_number",
    "wp_centrifugal_acceleration",
    "wp_centrifugal_force",
    "compute_centrifugal_accelerations",
    "add_centrifugal_acceleration",
    # Virtual Mass
    "C_VM_SPHERE",
    "virtual_mass_force",
    "virtual_mass_coefficient",
    "effective_particle_mass",
    "is_virtual_mass_significant",
    "wp_virtual_mass_acceleration",
    "compute_virtual_mass_correction",
]
