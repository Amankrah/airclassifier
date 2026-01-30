"""
Drag models for particle dynamics.

Provides various drag coefficient correlations for different
Reynolds number regimes and particle shapes.

This module provides Warp-accelerated drag model implementations with
particle-specific utilities (terminal velocity, Stokes number, etc.).

For the DragModel enum and generic force calculation API, see:
    airclassifier.kinetics.forces.drag

Individual drag model modules:
    - stokes: For Re_p < 1 (creeping flow)
    - schiller_naumann: For Re_p < 1000 (transitional regime)
    - haider_levenspiel: For non-spherical particles
"""

# Warp-accelerated implementations with particle utilities
from .stokes import (
    drag_coefficient_stokes,
    drag_force_stokes,
    terminal_velocity_stokes,
    stokes_settling_time,
    stokes_number,
    compute_stokes_drag_kernel,
)

from .schiller_naumann import (
    drag_coefficient_schiller_naumann,
    particle_reynolds_number,
    drag_force_schiller_naumann,
    terminal_velocity_schiller_naumann,
    compute_schiller_naumann_drag_kernel,
)

from .haider_levenspiel import (
    drag_coefficient_haider_levenspiel,
    drag_force_haider_levenspiel,
    terminal_velocity_haider_levenspiel,
    sphericity_from_aspect_ratio,
    get_sphericity,
    SPHERICITY_VALUES,
    compute_haider_levenspiel_drag_kernel,
)

# Re-export DragModel enum and generic drag functions from kinetics
from ...kinetics.forces.drag import (
    DragModel,
    drag_force,
    compute_drag_forces,
    compute_drag_accelerations,
)

__all__ = [
    # Drag model enum
    "DragModel",
    # Generic drag functions
    "drag_force",
    "compute_drag_forces",
    "compute_drag_accelerations",
    # Stokes drag
    "drag_coefficient_stokes",
    "drag_force_stokes",
    "terminal_velocity_stokes",
    "stokes_settling_time",
    "stokes_number",
    "compute_stokes_drag_kernel",
    # Schiller-Naumann drag
    "drag_coefficient_schiller_naumann",
    "particle_reynolds_number",
    "drag_force_schiller_naumann",
    "terminal_velocity_schiller_naumann",
    "compute_schiller_naumann_drag_kernel",
    # Haider-Levenspiel drag
    "drag_coefficient_haider_levenspiel",
    "drag_force_haider_levenspiel",
    "terminal_velocity_haider_levenspiel",
    "sphericity_from_aspect_ratio",
    "get_sphericity",
    "SPHERICITY_VALUES",
    "compute_haider_levenspiel_drag_kernel",
]
