"""
Heat Transfer Kernels
=====================

Warp kernels for the heat equation with variable conductivity,
RF source term, and latent heat sink:

    T_new = T + dt/(rho*c_p) * [div(k*grad(T)) + P_v - L_v*m_evap]

Includes a separate convective boundary condition kernel for the
material bed surface.
"""

# TODO: Import warp when implementing
# import warp as wp


# @wp.kernel
def heat_conduction_step():
    """Advance temperature field by one explicit FDM timestep.

    Uses second-order central differences for the Laplacian with
    variable conductivity. Boundary cells use one-sided differences.

    See engineering guide section 8.3 for full implementation.
    """
    # TODO: Implement as @wp.kernel
    raise NotImplementedError


# @wp.kernel
def apply_convection_bc():
    """Apply convective heat transfer at the material bed surface.

    -k * dT/dy |_surface = h * (T_surface - T_air)

    See engineering guide section 8.6 for full implementation.
    """
    # TODO: Implement as @wp.kernel
    raise NotImplementedError
