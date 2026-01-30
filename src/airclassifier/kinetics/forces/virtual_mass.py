"""
Virtual (added) mass force calculations.

The virtual mass force accounts for the inertia of fluid that must be
accelerated along with the particle. Important when particle density
is close to fluid density.
"""

import numpy as np
import warp as wp

from ...utils.constants import PI


# =============================================================================
# THEORY
# =============================================================================
# When a particle accelerates through a fluid, it must also accelerate
# some of the surrounding fluid. This requires additional force, equivalent
# to the particle having additional "virtual" mass.
#
# For a sphere: C_vm = 0.5 (virtual mass coefficient)
# F_vm = C_vm * rho_f * V_p * (Dv_f/Dt - dv_p/dt)
#
# where Dv_f/Dt is the material derivative of fluid velocity.


# Virtual mass coefficient for a sphere
C_VM_SPHERE = 0.5


# =============================================================================
# PYTHON FUNCTIONS
# =============================================================================

def virtual_mass_force(
    diameter: float,
    fluid_density: float,
    particle_acceleration: np.ndarray,
    fluid_acceleration: np.ndarray = np.array([0.0, 0.0, 0.0]),
    C_vm: float = C_VM_SPHERE
) -> np.ndarray:
    """
    Calculate virtual (added) mass force on a particle.

    F_vm = C_vm * rho_f * V_p * (a_fluid - a_particle)

    Args:
        diameter: Particle diameter [m]
        fluid_density: Fluid density [kg/m³]
        particle_acceleration: Particle acceleration [m/s²]
        fluid_acceleration: Fluid acceleration (material derivative) [m/s²]
        C_vm: Virtual mass coefficient (0.5 for sphere)

    Returns:
        Virtual mass force vector [N]
    """
    # Particle volume
    V_p = (PI / 6.0) * diameter ** 3

    # Virtual mass
    m_vm = C_vm * fluid_density * V_p

    # Force
    F_vm = m_vm * (fluid_acceleration - particle_acceleration)

    return F_vm


def virtual_mass_coefficient(sphericity: float = 1.0) -> float:
    """
    Get virtual mass coefficient based on particle shape.

    For non-spherical particles, C_vm varies with orientation.
    This returns an approximate value.

    Args:
        sphericity: Particle sphericity (1.0 = sphere)

    Returns:
        Virtual mass coefficient
    """
    # For spheres, C_vm = 0.5
    # For other shapes, it's generally between 0.5 and 1.0
    # This is a simplified approximation
    return 0.5 + 0.3 * (1.0 - sphericity)


def effective_particle_mass(
    diameter: float,
    particle_density: float,
    fluid_density: float,
    C_vm: float = C_VM_SPHERE
) -> float:
    """
    Calculate effective particle mass including virtual mass.

    m_eff = m_p + C_vm * rho_f * V_p

    Args:
        diameter: Particle diameter [m]
        particle_density: Particle density [kg/m³]
        fluid_density: Fluid density [kg/m³]
        C_vm: Virtual mass coefficient

    Returns:
        Effective mass [kg]
    """
    V_p = (PI / 6.0) * diameter ** 3
    m_p = particle_density * V_p
    m_vm = C_vm * fluid_density * V_p

    return m_p + m_vm


def is_virtual_mass_significant(
    particle_density: float,
    fluid_density: float,
    threshold: float = 0.1
) -> bool:
    """
    Determine if virtual mass effects are significant.

    Virtual mass is typically important when rho_f/rho_p > threshold.

    Args:
        particle_density: Particle density [kg/m³]
        fluid_density: Fluid density [kg/m³]
        threshold: Density ratio threshold

    Returns:
        True if virtual mass should be included
    """
    return fluid_density / particle_density > threshold


# =============================================================================
# WARP FUNCTIONS AND KERNELS
# =============================================================================

@wp.func
def wp_virtual_mass_acceleration(
    diameter: float,
    rho_p: float,
    rho_f: float,
    a_particle: wp.vec3,
    a_fluid: wp.vec3,
    C_vm: float
) -> wp.vec3:
    """
    Calculate acceleration contribution from virtual mass.

    The virtual mass effectively modifies the equation of motion:
    m_p * a_p = F_drag + F_grav + ... + F_vm

    where F_vm = C_vm * rho_f * V_p * (a_f - a_p)

    Rearranging:
    (m_p + C_vm * rho_f * V_p) * a_p = F_other + C_vm * rho_f * V_p * a_f

    This function returns the additional acceleration to add.
    """
    # Particle volume
    volume = (3.141592653589793 / 6.0) * diameter * diameter * diameter

    # Masses
    m_p = rho_p * volume
    m_vm = C_vm * rho_f * volume

    # Additional acceleration from virtual mass
    # (This is a simplified form; full treatment requires iterative solution)
    factor = m_vm / (m_p + m_vm)

    return wp.vec3(
        factor * (a_fluid[0] - a_particle[0]),
        factor * (a_fluid[1] - a_particle[1]),
        factor * (a_fluid[2] - a_particle[2])
    )


@wp.func
def wp_effective_mass_ratio(
    rho_p: float,
    rho_f: float,
    C_vm: float
) -> float:
    """
    Calculate ratio of effective mass to particle mass.

    m_eff / m_p = 1 + C_vm * rho_f / rho_p
    """
    return 1.0 + C_vm * rho_f / rho_p


@wp.kernel
def compute_virtual_mass_correction(
    diameters: wp.array(dtype=float),
    particle_accelerations: wp.array(dtype=wp.vec3),
    fluid_accelerations: wp.array(dtype=wp.vec3),
    corrected_accelerations: wp.array(dtype=wp.vec3),
    rho_p: float,
    rho_f: float,
    C_vm: float
):
    """
    Kernel to compute virtual mass corrections for all particles.

    Args:
        diameters: Particle diameters
        particle_accelerations: Current particle accelerations
        fluid_accelerations: Fluid accelerations at particle positions
        corrected_accelerations: Output corrected accelerations
        rho_p: Particle density
        rho_f: Fluid density
        C_vm: Virtual mass coefficient
    """
    tid = wp.tid()

    d = diameters[tid]
    a_p = particle_accelerations[tid]
    a_f = fluid_accelerations[tid]

    # Calculate correction
    a_vm = wp_virtual_mass_acceleration(d, rho_p, rho_f, a_p, a_f, C_vm)

    # Add correction to particle acceleration
    corrected_accelerations[tid] = a_p + a_vm
