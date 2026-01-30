"""
Drag force calculations for particles in fluid flow.

Implements various drag correlations for different Reynolds number regimes
and particle shapes.
"""

from enum import Enum
from typing import Tuple
import numpy as np
import warp as wp

from ...utils.constants import PI, NumericalConstants


class DragModel(Enum):
    """Available drag models."""
    STOKES = "stokes"                       # Re < 0.1
    SCHILLER_NAUMANN = "schiller_naumann"   # 0.1 < Re < 1000
    HAIDER_LEVENSPIEL = "haider_levenspiel" # Non-spherical
    MORSI_ALEXANDER = "morsi_alexander"     # Wide range piecewise
    GANSER = "ganser"                       # Non-spherical, wide range


# =============================================================================
# PYTHON FUNCTIONS (for testing and validation)
# =============================================================================

def particle_reynolds_number(
    diameter: float,
    relative_velocity: float,
    fluid_density: float,
    fluid_viscosity: float
) -> float:
    """
    Calculate particle Reynolds number.

    Args:
        diameter: Particle diameter [m]
        relative_velocity: Magnitude of (fluid_vel - particle_vel) [m/s]
        fluid_density: Fluid density [kg/m³]
        fluid_viscosity: Fluid dynamic viscosity [Pa·s]

    Returns:
        Particle Reynolds number [-]
    """
    return fluid_density * relative_velocity * diameter / fluid_viscosity


def drag_coefficient_stokes(Re: float) -> float:
    """Stokes drag coefficient: Cd = 24/Re."""
    if Re < NumericalConstants.EPSILON:
        return 24.0 / NumericalConstants.EPSILON
    return 24.0 / Re


def drag_coefficient_schiller_naumann(Re: float) -> float:
    """
    Schiller-Naumann drag coefficient.

    Valid for Re < 1000 (approximately).
    Cd = 24/Re * (1 + 0.15 * Re^0.687)
    """
    if Re < NumericalConstants.EPSILON:
        return 24.0 / NumericalConstants.EPSILON
    return (24.0 / Re) * (1.0 + 0.15 * Re ** 0.687)


def drag_coefficient_haider_levenspiel(Re: float, sphericity: float) -> float:
    """
    Haider-Levenspiel drag coefficient for non-spherical particles.

    Args:
        Re: Particle Reynolds number
        sphericity: Particle sphericity (0 < phi <= 1)

    Returns:
        Drag coefficient
    """
    if Re < NumericalConstants.EPSILON:
        Re = NumericalConstants.EPSILON

    phi = sphericity

    # Correlation coefficients (functions of sphericity)
    A = np.exp(2.3288 - 6.4581 * phi + 2.4486 * phi ** 2)
    B = 0.0964 + 0.5565 * phi
    C = np.exp(4.905 - 13.8944 * phi + 18.4222 * phi ** 2 - 10.2599 * phi ** 3)
    D = np.exp(1.4681 + 12.2584 * phi - 20.7322 * phi ** 2 + 15.8855 * phi ** 3)

    Cd = 24.0 / Re * (1.0 + A * Re ** B) + C / (1.0 + D / Re)
    return Cd


def drag_force(
    relative_velocity: np.ndarray,
    diameter: float,
    fluid_density: float,
    fluid_viscosity: float,
    drag_model: DragModel = DragModel.SCHILLER_NAUMANN,
    sphericity: float = 1.0
) -> np.ndarray:
    """
    Calculate drag force on a particle.

    F_drag = 0.5 * Cd * rho_f * A_p * |v_rel|^2 * (v_rel / |v_rel|)

    Args:
        relative_velocity: Fluid velocity - particle velocity [m/s]
        diameter: Particle diameter [m]
        fluid_density: Fluid density [kg/m³]
        fluid_viscosity: Fluid dynamic viscosity [Pa·s]
        drag_model: Drag coefficient correlation to use
        sphericity: Particle sphericity (for non-spherical models)

    Returns:
        Drag force vector [N]
    """
    v_rel_mag = np.linalg.norm(relative_velocity)

    if v_rel_mag < NumericalConstants.EPSILON:
        return np.zeros(3)

    # Calculate Reynolds number
    Re = particle_reynolds_number(diameter, v_rel_mag, fluid_density, fluid_viscosity)

    # Get drag coefficient
    if drag_model == DragModel.STOKES:
        Cd = drag_coefficient_stokes(Re)
    elif drag_model == DragModel.SCHILLER_NAUMANN:
        Cd = drag_coefficient_schiller_naumann(Re)
    elif drag_model == DragModel.HAIDER_LEVENSPIEL:
        Cd = drag_coefficient_haider_levenspiel(Re, sphericity)
    else:
        Cd = drag_coefficient_schiller_naumann(Re)  # Default

    # Projected area
    A_p = PI * diameter ** 2 / 4.0

    # Drag force magnitude
    F_mag = 0.5 * Cd * fluid_density * A_p * v_rel_mag ** 2

    # Direction (same as relative velocity)
    F_drag = F_mag * (relative_velocity / v_rel_mag)

    return F_drag


# =============================================================================
# WARP KERNELS FOR GPU COMPUTATION
# =============================================================================

@wp.func
def wp_particle_reynolds(
    diameter: float,
    v_rel_mag: float,
    rho_f: float,
    mu_f: float
) -> float:
    """Calculate particle Reynolds number in Warp kernel."""
    return rho_f * v_rel_mag * diameter / mu_f


@wp.func
def wp_drag_coefficient_stokes(Re: float) -> float:
    """Stokes drag coefficient."""
    eps = 1.0e-10
    if Re < eps:
        return 24.0 / eps
    return 24.0 / Re


@wp.func
def wp_drag_coefficient_schiller_naumann(Re: float) -> float:
    """Schiller-Naumann drag coefficient."""
    eps = 1.0e-10
    if Re < eps:
        return 24.0 / eps
    return (24.0 / Re) * (1.0 + 0.15 * wp.pow(Re, 0.687))


@wp.func
def wp_drag_force_stokes(
    v_rel: wp.vec3,
    diameter: float,
    rho_f: float,
    mu_f: float
) -> wp.vec3:
    """
    Calculate Stokes drag force.

    F = 3 * pi * mu * d * v_rel
    """
    coeff = 3.0 * 3.141592653589793 * mu_f * diameter
    return v_rel * coeff


@wp.func
def wp_drag_force_schiller_naumann(
    v_rel: wp.vec3,
    diameter: float,
    rho_f: float,
    mu_f: float
) -> wp.vec3:
    """
    Calculate drag force using Schiller-Naumann correlation.

    Returns drag force vector [N].
    """
    v_rel_mag = wp.length(v_rel)
    eps = 1.0e-10

    if v_rel_mag < eps:
        return wp.vec3(0.0, 0.0, 0.0)

    # Reynolds number
    Re = wp_particle_reynolds(diameter, v_rel_mag, rho_f, mu_f)

    # Drag coefficient
    Cd = wp_drag_coefficient_schiller_naumann(Re)

    # Projected area
    A_p = (3.141592653589793 / 4.0) * diameter * diameter

    # Force magnitude
    F_mag = 0.5 * Cd * rho_f * A_p * v_rel_mag * v_rel_mag

    # Force vector (in direction of relative velocity)
    return v_rel * (F_mag / v_rel_mag)


@wp.kernel
def compute_drag_forces(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    fluid_velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    rho_f: float,
    mu_f: float
):
    """
    Kernel to compute drag forces for all particles.

    Args:
        positions: Particle positions (not used but common in particle kernels)
        velocities: Particle velocities
        diameters: Particle diameters
        fluid_velocities: Local fluid velocity at each particle position
        forces: Output drag forces
        rho_f: Fluid density
        mu_f: Fluid dynamic viscosity
    """
    tid = wp.tid()

    v_p = velocities[tid]
    v_f = fluid_velocities[tid]
    d = diameters[tid]

    # Relative velocity (fluid - particle)
    v_rel = v_f - v_p

    # Compute drag force
    F_drag = wp_drag_force_schiller_naumann(v_rel, d, rho_f, mu_f)

    forces[tid] = F_drag


@wp.kernel
def compute_drag_accelerations(
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    fluid_velocities: wp.array(dtype=wp.vec3),
    particle_density: float,
    accelerations: wp.array(dtype=wp.vec3),
    rho_f: float,
    mu_f: float
):
    """
    Kernel to compute drag accelerations (F/m) for all particles.

    More efficient than computing force then dividing by mass.
    """
    tid = wp.tid()

    v_p = velocities[tid]
    v_f = fluid_velocities[tid]
    d = diameters[tid]

    # Relative velocity
    v_rel = v_f - v_p
    v_rel_mag = wp.length(v_rel)
    eps = 1.0e-10

    if v_rel_mag < eps:
        accelerations[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    # Reynolds number
    Re = wp_particle_reynolds(d, v_rel_mag, rho_f, mu_f)

    # Drag coefficient
    Cd = wp_drag_coefficient_schiller_naumann(Re)

    # Particle mass
    volume = (3.141592653589793 / 6.0) * d * d * d
    mass = particle_density * volume

    # Projected area
    A_p = (3.141592653589793 / 4.0) * d * d

    # Acceleration magnitude = F/m
    a_mag = 0.5 * Cd * rho_f * A_p * v_rel_mag * v_rel_mag / mass

    # Acceleration vector
    accelerations[tid] = v_rel * (a_mag / v_rel_mag)
