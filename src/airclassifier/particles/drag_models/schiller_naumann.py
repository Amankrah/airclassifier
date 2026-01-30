"""
Schiller-Naumann drag model for intermediate Reynolds numbers.

Valid for Re_p < 1000 (transitional regime).
Most commonly used correlation for particle tracking.
"""

import warp as wp
import numpy as np


@wp.func
def drag_coefficient_schiller_naumann(Re_p: float) -> float:
    """
    Schiller-Naumann drag coefficient correlation.

    C_D = (24 / Re_p) * (1 + 0.15 * Re_p^0.687)  for Re_p < 1000
    C_D = 0.44                                     for Re_p >= 1000

    Args:
        Re_p: Particle Reynolds number

    Returns:
        Drag coefficient
    """
    if Re_p < 1.0e-10:
        return 24.0e10

    if Re_p < 1000.0:
        return (24.0 / Re_p) * (1.0 + 0.15 * wp.pow(Re_p, 0.687))
    else:
        return 0.44


@wp.func
def particle_reynolds_number(
    velocity_rel: wp.vec3,
    diameter: float,
    fluid_density: float,
    fluid_viscosity: float
) -> float:
    """
    Compute particle Reynolds number.

    Re_p = rho_f * |V_rel| * d / mu

    Args:
        velocity_rel: Relative velocity (fluid - particle)
        diameter: Particle diameter [m]
        fluid_density: Fluid density [kg/m³]
        fluid_viscosity: Dynamic viscosity [Pa.s]

    Returns:
        Particle Reynolds number
    """
    vel_mag = wp.length(velocity_rel)
    if vel_mag < 1.0e-10:
        return 0.0
    return fluid_density * vel_mag * diameter / fluid_viscosity


@wp.func
def drag_force_schiller_naumann(
    velocity_rel: wp.vec3,
    diameter: float,
    fluid_density: float,
    fluid_viscosity: float
) -> wp.vec3:
    """
    Compute drag force using Schiller-Naumann correlation.

    F_D = 0.5 * C_D * rho_f * A * |V_rel| * V_rel

    where A = pi * d^2 / 4 is the projected area.

    Args:
        velocity_rel: Relative velocity (fluid - particle)
        diameter: Particle diameter [m]
        fluid_density: Fluid density [kg/m³]
        fluid_viscosity: Dynamic viscosity [Pa.s]

    Returns:
        Drag force vector [N]
    """
    vel_mag = wp.length(velocity_rel)
    if vel_mag < 1.0e-10:
        return wp.vec3(0.0, 0.0, 0.0)

    Re_p = fluid_density * vel_mag * diameter / fluid_viscosity
    C_D = drag_coefficient_schiller_naumann(Re_p)

    # Projected area
    area = 0.25 * 3.14159265359 * diameter * diameter

    # Drag force magnitude
    F_mag = 0.5 * C_D * fluid_density * area * vel_mag * vel_mag

    # Return force vector in direction of relative velocity
    return velocity_rel * (F_mag / vel_mag)


@wp.kernel
def compute_schiller_naumann_drag_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    fluid_velocities: wp.array(dtype=wp.vec3),
    is_active: wp.array(dtype=wp.int32),
    drag_forces: wp.array(dtype=wp.vec3),
    fluid_density: float,
    fluid_viscosity: float,
):
    """Compute Schiller-Naumann drag force for all particles."""
    tid = wp.tid()

    if is_active[tid] != 1:
        drag_forces[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    vel_rel = fluid_velocities[tid] - velocities[tid]
    d = diameters[tid]

    drag_forces[tid] = drag_force_schiller_naumann(
        vel_rel, d, fluid_density, fluid_viscosity
    )


def terminal_velocity_schiller_naumann(
    diameter: float,
    particle_density: float,
    fluid_density: float,
    fluid_viscosity: float,
    gravity: float = 9.81,
    max_iterations: int = 100,
    tolerance: float = 1.0e-6
) -> float:
    """
    Compute terminal velocity using Schiller-Naumann drag.

    Uses iterative solution since C_D depends on velocity.

    Args:
        diameter: Particle diameter [m]
        particle_density: Particle density [kg/m³]
        fluid_density: Fluid density [kg/m³]
        fluid_viscosity: Dynamic viscosity [Pa.s]
        gravity: Gravitational acceleration [m/s²]
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Terminal velocity [m/s]
    """
    delta_rho = particle_density - fluid_density
    if delta_rho <= 0:
        return 0.0

    # Initial guess using Stokes
    v_t = (diameter**2 * delta_rho * gravity) / (18.0 * fluid_viscosity)

    for _ in range(max_iterations):
        Re_p = fluid_density * v_t * diameter / fluid_viscosity

        if Re_p < 1000:
            C_D = (24.0 / Re_p) * (1.0 + 0.15 * Re_p**0.687)
        else:
            C_D = 0.44

        # Terminal velocity from force balance
        v_new = np.sqrt((4.0 * diameter * delta_rho * gravity) /
                       (3.0 * C_D * fluid_density))

        if abs(v_new - v_t) / v_t < tolerance:
            return v_new

        v_t = v_new

    return v_t
