"""
Heat Transfer Kernels
=====================

Warp GPU kernels for the heat equation with variable conductivity,
RF source term, and latent heat sink:

    T_new = T + dt/(rho*c_p) * [div(k*grad(T)) + P_v - L_v*m_evap]

Includes a separate convective boundary condition kernel for the
material bed surface.
"""

from __future__ import annotations

try:
    import warp as wp

    @wp.kernel
    def heat_conduction_step(
        T: wp.array3d(dtype=float),
        T_new: wp.array3d(dtype=float),
        P_v: wp.array3d(dtype=float),
        evap_rate: wp.array3d(dtype=float),
        rho_cp: wp.array3d(dtype=float),
        k_eff: wp.array3d(dtype=float),
        L_v: float,
        dx: float, dy: float, dz: float, dt: float,
        nx: int, ny: int, nz: int,
    ):
        """Advance temperature field by one explicit FDM timestep.

        T_new = T + dt/(rho*c_p) * [div(k*grad(T)) + P_v - L_v*m_evap]
        """
        i, j, k = wp.tid()
        if i >= nx or j >= ny or k >= nz:
            return

        if i <= 0 or i >= nx - 1 or j <= 0 or j >= ny - 1 or k <= 0 or k >= nz - 1:
            T_new[i, j, k] = T[i, j, k]
            return

        # Central difference Laplacian with variable k
        k_xp = 0.5 * (k_eff[i, j, k] + k_eff[i + 1, j, k])
        k_xm = 0.5 * (k_eff[i, j, k] + k_eff[i - 1, j, k])
        lap_x = (k_xp * (T[i + 1, j, k] - T[i, j, k]) - k_xm * (T[i, j, k] - T[i - 1, j, k])) / (dx * dx)

        k_yp = 0.5 * (k_eff[i, j, k] + k_eff[i, j + 1, k])
        k_ym = 0.5 * (k_eff[i, j, k] + k_eff[i, j - 1, k])
        lap_y = (k_yp * (T[i, j + 1, k] - T[i, j, k]) - k_ym * (T[i, j, k] - T[i, j - 1, k])) / (dy * dy)

        k_zp = 0.5 * (k_eff[i, j, k] + k_eff[i, j, k + 1])
        k_zm = 0.5 * (k_eff[i, j, k] + k_eff[i, j, k - 1])
        lap_z = (k_zp * (T[i, j, k + 1] - T[i, j, k]) - k_zm * (T[i, j, k] - T[i, j, k - 1])) / (dz * dz)

        laplacian = lap_x + lap_y + lap_z
        source = P_v[i, j, k]
        sink = L_v * evap_rate[i, j, k]

        rc = wp.max(rho_cp[i, j, k], 1.0)
        T_new[i, j, k] = T[i, j, k] + dt / rc * (laplacian + source - sink)

    @wp.kernel
    def apply_convection_bc(
        T: wp.array3d(dtype=float),
        j_surface: int,
        h_conv: float,
        T_air: float,
        rho_cp: wp.array3d(dtype=float),
        dy: float, dt: float,
        nx: int, nz: int,
    ):
        """Apply convective heat transfer at the material bed surface.

        -k * dT/dy |_surface = h * (T_surface - T_air)
        """
        i, k = wp.tid()
        if i >= nx or k >= nz:
            return

        j = j_surface
        T_s = T[i, j, k]
        q_conv = h_conv * (T_s - T_air)
        rc = wp.max(rho_cp[i, j, k], 1.0)
        T[i, j, k] = T_s - dt * q_conv / (rc * dy * 0.5)

    _HAS_WARP = True

except ImportError:
    _HAS_WARP = False
