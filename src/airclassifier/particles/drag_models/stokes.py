"""
Stokes drag model for very low Reynolds number flows.

Valid for Re_p < 1 (creeping flow regime).
"""

import warp as wp
import numpy as np


@wp.func
def drag_coefficient_stokes(Re_p: float) -> float:
    """
    Stokes drag coefficient.

    C_D = 24 / Re_p

    Valid for Re_p < 1

    Args:
        Re_p: Particle Reynolds number

    Returns:
        Drag coefficient
    """
    if Re_p < 1.0e-10:
        return 24.0e10  # Very large for numerical stability
    return 24.0 / Re_p


@wp.func
def drag_force_stokes(
    velocity_rel: wp.vec3,
    diameter: float,
    fluid_viscosity: float
) -> wp.vec3:
    """
    Stokes drag force for creeping flow.

    F_D = 3 * pi * mu * d * V_rel

    Args:
        velocity_rel: Relative velocity (fluid - particle)
        diameter: Particle diameter [m]
        fluid_viscosity: Dynamic viscosity [Pa.s]

    Returns:
        Drag force vector [N]
    """
    # Stokes drag: F = 3 * pi * mu * d * V
    coeff = 3.0 * 3.14159265359 * fluid_viscosity * diameter
    return velocity_rel * coeff


@wp.func
def terminal_velocity_stokes(
    diameter: float,
    particle_density: float,
    fluid_density: float,
    fluid_viscosity: float,
    gravity: float
) -> float:
    """
    Terminal settling velocity in Stokes regime.

    V_t = (d^2 * (rho_p - rho_f) * g) / (18 * mu)

    Args:
        diameter: Particle diameter [m]
        particle_density: Particle density [kg/m³]
        fluid_density: Fluid density [kg/m³]
        fluid_viscosity: Dynamic viscosity [Pa.s]
        gravity: Gravitational acceleration [m/s²]

    Returns:
        Terminal velocity [m/s]
    """
    d_sq = diameter * diameter
    delta_rho = particle_density - fluid_density

    if delta_rho <= 0.0:
        return 0.0

    return d_sq * delta_rho * gravity / (18.0 * fluid_viscosity)


@wp.kernel
def compute_stokes_drag_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    fluid_velocities: wp.array(dtype=wp.vec3),
    is_active: wp.array(dtype=wp.int32),
    drag_forces: wp.array(dtype=wp.vec3),
    fluid_viscosity: float,
):
    """Compute Stokes drag force for all particles."""
    tid = wp.tid()

    if is_active[tid] != 1:
        drag_forces[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    vel_rel = fluid_velocities[tid] - velocities[tid]
    d = diameters[tid]

    drag_forces[tid] = drag_force_stokes(vel_rel, d, fluid_viscosity)


def stokes_settling_time(
    diameter: float,
    particle_density: float,
    fluid_viscosity: float
) -> float:
    """
    Characteristic settling time for Stokes flow.

    tau = (rho_p * d^2) / (18 * mu)

    Args:
        diameter: Particle diameter [m]
        particle_density: Particle density [kg/m³]
        fluid_viscosity: Dynamic viscosity [Pa.s]

    Returns:
        Settling time [s]
    """
    return particle_density * diameter**2 / (18.0 * fluid_viscosity)


def stokes_number(
    diameter: float,
    particle_density: float,
    fluid_viscosity: float,
    characteristic_velocity: float,
    characteristic_length: float
) -> float:
    """
    Compute Stokes number.

    Stk = tau * U / L

    where tau is the particle relaxation time.

    Args:
        diameter: Particle diameter [m]
        particle_density: Particle density [kg/m³]
        fluid_viscosity: Dynamic viscosity [Pa.s]
        characteristic_velocity: Characteristic flow velocity [m/s]
        characteristic_length: Characteristic length scale [m]

    Returns:
        Stokes number [-]
    """
    tau = stokes_settling_time(diameter, particle_density, fluid_viscosity)
    return tau * characteristic_velocity / characteristic_length
