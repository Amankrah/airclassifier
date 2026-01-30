"""
Centrifugal force calculations for particles in rotating flow.

In a cyclone, particles experience apparent centrifugal force due to
the swirling motion. This is the primary mechanism for particle separation.
"""

import numpy as np
import warp as wp

from ...utils.constants import PI


# =============================================================================
# PYTHON FUNCTIONS
# =============================================================================

def centrifugal_force(
    position: np.ndarray,
    velocity: np.ndarray,
    diameter: float,
    particle_density: float,
    axis_center: np.ndarray = np.array([0.0, 0.0, 0.0]),
    axis_direction: np.ndarray = np.array([0.0, 1.0, 0.0])
) -> np.ndarray:
    """
    Calculate centrifugal force on a particle in swirling flow.

    F_centrifugal = m * v_tangential^2 / r (radially outward)

    Args:
        position: Particle position [m]
        velocity: Particle velocity [m/s]
        diameter: Particle diameter [m]
        particle_density: Particle density [kg/m³]
        axis_center: Point on rotation axis [m]
        axis_direction: Unit vector along rotation axis

    Returns:
        Centrifugal force vector [N]
    """
    # Particle mass
    volume = (PI / 6.0) * diameter ** 3
    mass = particle_density * volume

    # Vector from axis to particle
    axis_dir = axis_direction / np.linalg.norm(axis_direction)
    to_particle = position - axis_center

    # Project out axial component to get radial vector
    axial_component = np.dot(to_particle, axis_dir) * axis_dir
    radial_vector = to_particle - axial_component

    r = np.linalg.norm(radial_vector)
    if r < 1e-10:
        return np.zeros(3)

    radial_unit = radial_vector / r

    # Get tangential velocity component
    # v_tangential = v - v_radial - v_axial
    v_radial = np.dot(velocity, radial_unit) * radial_unit
    v_axial = np.dot(velocity, axis_dir) * axis_dir
    v_tangential = velocity - v_radial - v_axial

    v_tan_mag = np.linalg.norm(v_tangential)

    # Centrifugal force = m * v_tan^2 / r (pointing radially outward)
    F_mag = mass * v_tan_mag ** 2 / r

    return F_mag * radial_unit


def centrifugal_acceleration(
    position: np.ndarray,
    velocity: np.ndarray,
    axis_center: np.ndarray = np.array([0.0, 0.0, 0.0]),
    axis_direction: np.ndarray = np.array([0.0, 1.0, 0.0])
) -> np.ndarray:
    """
    Calculate centrifugal acceleration (F/m) for a particle.

    a_centrifugal = v_tangential^2 / r (radially outward)

    Args:
        position: Particle position [m]
        velocity: Particle velocity [m/s]
        axis_center: Point on rotation axis [m]
        axis_direction: Unit vector along rotation axis

    Returns:
        Centrifugal acceleration vector [m/s²]
    """
    # Vector from axis to particle
    axis_dir = axis_direction / np.linalg.norm(axis_direction)
    to_particle = position - axis_center

    # Radial vector (perpendicular to axis)
    axial_component = np.dot(to_particle, axis_dir) * axis_dir
    radial_vector = to_particle - axial_component

    r = np.linalg.norm(radial_vector)
    if r < 1e-10:
        return np.zeros(3)

    radial_unit = radial_vector / r

    # Tangential velocity
    v_radial = np.dot(velocity, radial_unit) * radial_unit
    v_axial = np.dot(velocity, axis_dir) * axis_dir
    v_tangential = velocity - v_radial - v_axial

    v_tan_mag = np.linalg.norm(v_tangential)

    # Centrifugal acceleration = v_tan^2 / r
    a_mag = v_tan_mag ** 2 / r

    return a_mag * radial_unit


def separation_number(
    tangential_velocity: float,
    radial_position: float,
    diameter: float,
    particle_density: float,
    fluid_density: float,
    fluid_viscosity: float
) -> float:
    """
    Calculate the separation number (ratio of centrifugal to drag force).

    St = (rho_p * d^2 * v_tan) / (18 * mu * r)

    Higher values indicate easier separation.

    Args:
        tangential_velocity: Tangential velocity magnitude [m/s]
        radial_position: Distance from axis [m]
        diameter: Particle diameter [m]
        particle_density: Particle density [kg/m³]
        fluid_density: Fluid density [kg/m³] (not used in Stokes regime)
        fluid_viscosity: Fluid dynamic viscosity [Pa·s]

    Returns:
        Separation number [-]
    """
    return (particle_density * diameter ** 2 * tangential_velocity) / (
        18.0 * fluid_viscosity * radial_position
    )


# =============================================================================
# WARP FUNCTIONS AND KERNELS
# =============================================================================

@wp.func
def wp_centrifugal_acceleration(
    pos: wp.vec3,
    vel: wp.vec3,
    axis_center: wp.vec3
) -> wp.vec3:
    """
    Calculate centrifugal acceleration for Y-axis aligned rotation.

    Args:
        pos: Particle position
        vel: Particle velocity
        axis_center: Center of rotation axis (on Y-axis)

    Returns:
        Centrifugal acceleration vector
    """
    # Radial vector in XZ plane
    rx = pos[0] - axis_center[0]
    rz = pos[2] - axis_center[2]
    r = wp.sqrt(rx * rx + rz * rz)

    eps = 1.0e-10
    if r < eps:
        return wp.vec3(0.0, 0.0, 0.0)

    # Radial unit vector
    radial_x = rx / r
    radial_z = rz / r

    # Tangential velocity (perpendicular to radial in XZ plane)
    # v_tan = v - (v · radial) * radial - v_y * y_hat
    v_radial = vel[0] * radial_x + vel[2] * radial_z
    v_tan_x = vel[0] - v_radial * radial_x
    v_tan_z = vel[2] - v_radial * radial_z
    v_tan_mag_sq = v_tan_x * v_tan_x + v_tan_z * v_tan_z

    # Centrifugal acceleration = v_tan^2 / r (radially outward)
    a_mag = v_tan_mag_sq / r

    return wp.vec3(a_mag * radial_x, 0.0, a_mag * radial_z)


@wp.func
def wp_centrifugal_force(
    pos: wp.vec3,
    vel: wp.vec3,
    diameter: float,
    rho_p: float,
    axis_center: wp.vec3
) -> wp.vec3:
    """
    Calculate centrifugal force for Y-axis aligned rotation.
    """
    # Particle mass
    volume = (3.141592653589793 / 6.0) * diameter * diameter * diameter
    mass = rho_p * volume

    # Get acceleration and multiply by mass
    a_cent = wp_centrifugal_acceleration(pos, vel, axis_center)

    return wp.vec3(mass * a_cent[0], mass * a_cent[1], mass * a_cent[2])


@wp.kernel
def compute_centrifugal_accelerations(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    accelerations: wp.array(dtype=wp.vec3),
    axis_center: wp.vec3
):
    """
    Kernel to compute centrifugal accelerations for all particles.

    Args:
        positions: Particle positions
        velocities: Particle velocities
        accelerations: Output acceleration vectors
        axis_center: Center of rotation axis
    """
    tid = wp.tid()
    accelerations[tid] = wp_centrifugal_acceleration(
        positions[tid], velocities[tid], axis_center
    )


@wp.kernel
def add_centrifugal_acceleration(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    accelerations: wp.array(dtype=wp.vec3),
    axis_center: wp.vec3
):
    """
    Kernel to add centrifugal acceleration to existing accelerations.

    Args:
        positions: Particle positions
        velocities: Particle velocities
        accelerations: Acceleration vectors (modified in place)
        axis_center: Center of rotation axis
    """
    tid = wp.tid()
    a_cent = wp_centrifugal_acceleration(
        positions[tid], velocities[tid], axis_center
    )
    accelerations[tid] = accelerations[tid] + a_cent
