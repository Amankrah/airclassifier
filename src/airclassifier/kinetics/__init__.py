"""
Kinetics module for cyclone air classifier.

Provides force calculations, trajectory integration, and
separation efficiency analysis for particles in cyclone flows.
"""

from .forces import (
    # Drag
    DragModel,
    particle_reynolds_number,
    drag_coefficient_stokes,
    drag_coefficient_schiller_naumann,
    drag_coefficient_haider_levenspiel,
    drag_force,
    # Gravity
    gravity_force,
    gravity_acceleration,
    terminal_velocity_stokes,
    terminal_velocity_intermediate,
    # Centrifugal
    centrifugal_force,
    centrifugal_acceleration,
    separation_number,
    # Virtual Mass
    virtual_mass_force,
    effective_particle_mass,
    is_virtual_mass_significant,
)

from .separation_efficiency import (
    GradeEfficiencyCurve,
    compute_grade_efficiency,
    theoretical_d50_lapple,
    theoretical_d50_barth,
    rosin_rammler_grade_efficiency,
    fit_grade_efficiency_curve,
    plot_grade_efficiency,
)

__all__ = [
    # Drag
    "DragModel",
    "particle_reynolds_number",
    "drag_coefficient_stokes",
    "drag_coefficient_schiller_naumann",
    "drag_coefficient_haider_levenspiel",
    "drag_force",
    # Gravity
    "gravity_force",
    "gravity_acceleration",
    "terminal_velocity_stokes",
    "terminal_velocity_intermediate",
    # Centrifugal
    "centrifugal_force",
    "centrifugal_acceleration",
    "separation_number",
    # Virtual Mass
    "virtual_mass_force",
    "effective_particle_mass",
    "is_virtual_mass_significant",
    # Separation Efficiency
    "GradeEfficiencyCurve",
    "compute_grade_efficiency",
    "theoretical_d50_lapple",
    "theoretical_d50_barth",
    "rosin_rammler_grade_efficiency",
    "fit_grade_efficiency_curve",
    "plot_grade_efficiency",
]
