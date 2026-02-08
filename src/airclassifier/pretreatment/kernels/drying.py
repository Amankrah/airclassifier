"""
Drying Kernels
==============

Warp kernels for moisture diffusion and evaporation:

    dM/dt = div(D_eff * grad(M)) - m_evap / rho_dry

Evaporation rate: m_evap = rho_dry * k_evap * M * max(0, T - T_threshold)
Diffusivity: D_eff = D0 * exp(-Ea / (R * T_K))
"""

# TODO: Import warp when implementing
# import warp as wp


# @wp.kernel
def moisture_step():
    """Advance moisture field by one timestep.

    Central-difference diffusion with Arrhenius diffusivity,
    plus evaporation sink proportional to temperature above threshold.

    See engineering guide section 8.4 for full implementation.
    """
    # TODO: Implement as @wp.kernel
    raise NotImplementedError
