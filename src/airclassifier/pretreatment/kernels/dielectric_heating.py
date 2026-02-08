"""
Dielectric Heating Kernels
==========================

Warp kernels for RF power density computation and material property update.

P_v = 2*pi*f*eps_0*eps''*|E|^2   [W/m^3]
"""

# TODO: Import warp when implementing
# import warp as wp

# Precomputed constant: 2*pi * 27.12e6 * 8.854e-12
TWO_PI_F_EPS0 = 1.5098e-3  # for 27.12 MHz


# @wp.kernel
def compute_power_density(
    # e_field_sq: wp.array3d(dtype=float),
    # eps_loss: wp.array3d(dtype=float),
    # power_density: wp.array3d(dtype=float),
    # two_pi_f_eps0: float,
):
    """Compute volumetric RF power density at each grid cell.

    P_v = 2*pi * f * eps_0 * eps'' * |E|^2

    Called every timestep after the E-field and material properties
    have been updated.
    """
    # TODO: Implement as @wp.kernel
    # i, j, k = wp.tid()
    # E2 = e_field_sq[i, j, k]
    # loss = eps_loss[i, j, k]
    # power_density[i, j, k] = two_pi_f_eps0 * loss * E2
    raise NotImplementedError


# @wp.kernel
def update_material_properties():
    """Update all material properties from current T and M fields.

    Computes eps'', eps', rho*c_p, k_eff at each cell.
    Must be called after every thermal and moisture solve step.
    """
    # TODO: Implement as @wp.kernel (see engineering guide section 8.2)
    raise NotImplementedError
