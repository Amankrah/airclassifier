"""
Gravitational force calculations for particles.

Includes both gravitational settling and buoyancy effects.
"""

import numpy as np
import warp as wp

from ...utils.constants import GRAVITY, PI


# =============================================================================
# PYTHON FUNCTIONS
# =============================================================================

def gravity_force(
    diameter: float,
    particle_density: float,
    fluid_density: float = 0.0,
    g: float = GRAVITY,
    direction: np.ndarray = np.array([0.0, -1.0, 0.0])
) -> np.ndarray:
    """
    Calculate gravitational force on a particle including buoyancy.

    F_gravity = (rho_p - rho_f) * V_p * g

    Args:
        diameter: Particle diameter [m]
        particle_density: Particle density [kg/m³]
        fluid_density: Fluid density [kg/m³] (for buoyancy)
        g: Gravitational acceleration magnitude [m/s²]
        direction: Unit vector in direction of gravity

    Returns:
        Gravitational force vector [N]
    """
    # Particle volume
    V_p = (PI / 6.0) * diameter ** 3

    # Net force (gravity - buoyancy)
    F_mag = (particle_density - fluid_density) * V_p * g

    # Normalize direction
    dir_normalized = direction / np.linalg.norm(direction)

    return F_mag * dir_normalized


def gravity_acceleration(
    particle_density: float,
    fluid_density: float = 0.0,
    g: float = GRAVITY,
    direction: np.ndarray = np.array([0.0, -1.0, 0.0])
) -> np.ndarray:
    """
    Calculate gravitational acceleration including buoyancy.

    a_gravity = (1 - rho_f/rho_p) * g

    Note: This is independent of particle size for spheres.

    Args:
        particle_density: Particle density [kg/m³]
        fluid_density: Fluid density [kg/m³]
        g: Gravitational acceleration magnitude [m/s²]
        direction: Unit vector in direction of gravity

    Returns:
        Gravitational acceleration vector [m/s²]
    """
    # Effective acceleration
    a_mag = (1.0 - fluid_density / particle_density) * g

    # Normalize direction
    dir_normalized = direction / np.linalg.norm(direction)

    return a_mag * dir_normalized


def terminal_velocity_stokes(
    diameter: float,
    particle_density: float,
    fluid_density: float,
    fluid_viscosity: float,
    g: float = GRAVITY
) -> float:
    """
    Calculate Stokes terminal settling velocity.

    Valid for Rep < 0.1.

    v_t = (rho_p - rho_f) * g * d^2 / (18 * mu)

    Args:
        diameter: Particle diameter [m]
        particle_density: Particle density [kg/m³]
        fluid_density: Fluid density [kg/m³]
        fluid_viscosity: Fluid dynamic viscosity [Pa·s]
        g: Gravitational acceleration [m/s²]

    Returns:
        Terminal settling velocity [m/s]
    """
    return (particle_density - fluid_density) * g * diameter ** 2 / (18.0 * fluid_viscosity)


def terminal_velocity_intermediate(
    diameter: float,
    particle_density: float,
    fluid_density: float,
    fluid_viscosity: float,
    g: float = GRAVITY,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> float:
    """
    Calculate terminal velocity in intermediate regime using iteration.

    Uses Schiller-Naumann drag correlation.

    Args:
        diameter: Particle diameter [m]
        particle_density: Particle density [kg/m³]
        fluid_density: Fluid density [kg/m³]
        fluid_viscosity: Fluid dynamic viscosity [Pa·s]
        g: Gravitational acceleration [m/s²]
        max_iterations: Maximum iteration count
        tolerance: Convergence tolerance

    Returns:
        Terminal settling velocity [m/s]
    """
    # Initial guess using Stokes
    v_t = terminal_velocity_stokes(
        diameter, particle_density, fluid_density, fluid_viscosity, g
    )

    for _ in range(max_iterations):
        # Reynolds number
        Re = fluid_density * v_t * diameter / fluid_viscosity

        # Schiller-Naumann drag coefficient
        if Re < 1e-10:
            Cd = 24.0 / 1e-10
        else:
            Cd = (24.0 / Re) * (1.0 + 0.15 * Re ** 0.687)

        # New terminal velocity from force balance
        # Drag = Weight - Buoyancy
        # 0.5 * Cd * rho_f * A * v^2 = (rho_p - rho_f) * V * g
        V_p = (PI / 6.0) * diameter ** 3
        A_p = (PI / 4.0) * diameter ** 2

        v_t_new = np.sqrt(
            2.0 * (particle_density - fluid_density) * V_p * g /
            (Cd * fluid_density * A_p)
        )

        # Check convergence
        if abs(v_t_new - v_t) / max(v_t, 1e-10) < tolerance:
            return v_t_new

        v_t = v_t_new

    return v_t


# =============================================================================
# WARP FUNCTIONS AND KERNELS
# =============================================================================

@wp.func
def wp_gravity_acceleration(
    rho_p: float,
    rho_f: float,
    g: float
) -> wp.vec3:
    """
    Calculate gravitational acceleration with buoyancy.

    Returns acceleration in -Y direction (downward).
    """
    a_mag = (1.0 - rho_f / rho_p) * g
    return wp.vec3(0.0, -a_mag, 0.0)


@wp.func
def wp_gravity_force(
    diameter: float,
    rho_p: float,
    rho_f: float,
    g: float
) -> wp.vec3:
    """
    Calculate gravitational force with buoyancy.

    Returns force in -Y direction (downward).
    """
    volume = (3.141592653589793 / 6.0) * diameter * diameter * diameter
    F_mag = (rho_p - rho_f) * volume * g
    return wp.vec3(0.0, -F_mag, 0.0)


@wp.kernel
def compute_gravity_forces(
    diameters: wp.array(dtype=float),
    forces: wp.array(dtype=wp.vec3),
    rho_p: float,
    rho_f: float,
    g: float
):
    """
    Kernel to compute gravitational forces for all particles.

    Args:
        diameters: Particle diameters
        forces: Output force vectors
        rho_p: Particle density
        rho_f: Fluid density
        g: Gravitational acceleration magnitude
    """
    tid = wp.tid()
    d = diameters[tid]
    forces[tid] = wp_gravity_force(d, rho_p, rho_f, g)


@wp.kernel
def add_gravity_acceleration(
    accelerations: wp.array(dtype=wp.vec3),
    rho_p: float,
    rho_f: float,
    g: float
):
    """
    Kernel to add gravitational acceleration to existing accelerations.

    Args:
        accelerations: Acceleration vectors (modified in place)
        rho_p: Particle density
        rho_f: Fluid density
        g: Gravitational acceleration magnitude
    """
    tid = wp.tid()
    a_grav = wp_gravity_acceleration(rho_p, rho_f, g)
    accelerations[tid] = accelerations[tid] + a_grav
