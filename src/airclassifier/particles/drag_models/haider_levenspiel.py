"""
Haider-Levenspiel drag model for non-spherical particles.

Accounts for particle shape through sphericity factor.
Valid for a wide range of Reynolds numbers and particle shapes.
"""

import warp as wp
import numpy as np


@wp.func
def drag_coefficient_haider_levenspiel(
    Re_p: float,
    sphericity: float
) -> float:
    """
    Haider-Levenspiel drag coefficient for non-spherical particles.

    C_D = (24/Re)(1 + A*Re^B) + C/(1 + D/Re)

    where A, B, C, D are functions of sphericity.

    Args:
        Re_p: Particle Reynolds number
        sphericity: Particle sphericity (0-1, 1 = perfect sphere)

    Returns:
        Drag coefficient
    """
    if Re_p < 1.0e-10:
        return 24.0e10

    phi = sphericity

    # Correlation coefficients (Haider & Levenspiel, 1989)
    A = wp.exp(2.3288 - 6.4581 * phi + 2.4486 * phi * phi)
    B = 0.0964 + 0.5565 * phi
    C = wp.exp(4.905 - 13.8944 * phi + 18.4222 * phi * phi - 10.2599 * phi * phi * phi)
    D = wp.exp(1.4681 + 12.2584 * phi - 20.7322 * phi * phi + 15.8855 * phi * phi * phi)

    # Drag coefficient
    term1 = (24.0 / Re_p) * (1.0 + A * wp.pow(Re_p, B))
    term2 = C / (1.0 + D / Re_p)

    return term1 + term2


@wp.func
def drag_force_haider_levenspiel(
    velocity_rel: wp.vec3,
    diameter_eq: float,
    sphericity: float,
    fluid_density: float,
    fluid_viscosity: float
) -> wp.vec3:
    """
    Compute drag force using Haider-Levenspiel correlation.

    Args:
        velocity_rel: Relative velocity (fluid - particle)
        diameter_eq: Equivalent sphere diameter [m]
        sphericity: Particle sphericity [-]
        fluid_density: Fluid density [kg/m³]
        fluid_viscosity: Dynamic viscosity [Pa.s]

    Returns:
        Drag force vector [N]
    """
    vel_mag = wp.length(velocity_rel)
    if vel_mag < 1.0e-10:
        return wp.vec3(0.0, 0.0, 0.0)

    Re_p = fluid_density * vel_mag * diameter_eq / fluid_viscosity
    C_D = drag_coefficient_haider_levenspiel(Re_p, sphericity)

    # Projected area (using equivalent diameter)
    area = 0.25 * 3.14159265359 * diameter_eq * diameter_eq

    # Drag force magnitude
    F_mag = 0.5 * C_D * fluid_density * area * vel_mag * vel_mag

    return velocity_rel * (F_mag / vel_mag)


@wp.kernel
def compute_haider_levenspiel_drag_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    sphericities: wp.array(dtype=float),
    fluid_velocities: wp.array(dtype=wp.vec3),
    is_active: wp.array(dtype=wp.int32),
    drag_forces: wp.array(dtype=wp.vec3),
    fluid_density: float,
    fluid_viscosity: float,
):
    """Compute Haider-Levenspiel drag force for all particles."""
    tid = wp.tid()

    if is_active[tid] != 1:
        drag_forces[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    vel_rel = fluid_velocities[tid] - velocities[tid]
    d = diameters[tid]
    phi = sphericities[tid]

    drag_forces[tid] = drag_force_haider_levenspiel(
        vel_rel, d, phi, fluid_density, fluid_viscosity
    )


def sphericity_from_aspect_ratio(aspect_ratio: float) -> float:
    """
    Estimate sphericity from aspect ratio for ellipsoidal particles.

    Args:
        aspect_ratio: Length/diameter ratio

    Returns:
        Estimated sphericity
    """
    if aspect_ratio <= 0:
        return 1.0

    # Approximation for prolate ellipsoids
    e = aspect_ratio
    if e >= 1:
        # Prolate (elongated)
        return (2.0 * e**(2/3)) / (1.0 + (e**2 - 1.0)**0.5 /
               (e * np.arcsin((e**2 - 1.0)**0.5 / e)))
    else:
        # Oblate (flattened)
        return (2.0 * e**(2/3)) / (1.0 + ((1.0 - e**2)**0.5 /
               (e * np.arctanh((1.0 - e**2)**0.5))))


def terminal_velocity_haider_levenspiel(
    diameter: float,
    sphericity: float,
    particle_density: float,
    fluid_density: float,
    fluid_viscosity: float,
    gravity: float = 9.81,
    max_iterations: int = 100,
    tolerance: float = 1.0e-6
) -> float:
    """
    Compute terminal velocity for non-spherical particles.

    Args:
        diameter: Equivalent sphere diameter [m]
        sphericity: Particle sphericity [-]
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

        phi = sphericity
        A = np.exp(2.3288 - 6.4581 * phi + 2.4486 * phi**2)
        B = 0.0964 + 0.5565 * phi
        C = np.exp(4.905 - 13.8944 * phi + 18.4222 * phi**2 - 10.2599 * phi**3)
        D = np.exp(1.4681 + 12.2584 * phi - 20.7322 * phi**2 + 15.8855 * phi**3)

        C_D = (24.0 / Re_p) * (1.0 + A * Re_p**B) + C / (1.0 + D / Re_p)

        v_new = np.sqrt((4.0 * diameter * delta_rho * gravity) /
                       (3.0 * C_D * fluid_density))

        if abs(v_new - v_t) / v_t < tolerance:
            return v_new

        v_t = v_new

    return v_t


# Common sphericity values for different particle shapes
SPHERICITY_VALUES = {
    "sphere": 1.0,
    "cube": 0.806,
    "octahedron": 0.846,
    "tetrahedron": 0.670,
    "cylinder_1_1": 0.874,    # L/D = 1
    "cylinder_2_1": 0.832,    # L/D = 2
    "cylinder_5_1": 0.698,    # L/D = 5
    "disk_1_3": 0.827,        # D/H = 3
    "disk_1_10": 0.594,       # D/H = 10
    "sand_rounded": 0.82,
    "sand_angular": 0.66,
    "coal_crushed": 0.75,
    "glass_crushed": 0.65,
}


def get_sphericity(shape: str) -> float:
    """
    Get sphericity value for a named shape.

    Args:
        shape: Shape name (see SPHERICITY_VALUES)

    Returns:
        Sphericity value
    """
    return SPHERICITY_VALUES.get(shape.lower(), 1.0)
