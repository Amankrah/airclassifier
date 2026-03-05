"""
Drying Kernels
==============

Warp GPU kernels for moisture diffusion and evaporation:

    dM/dt = div(D_eff * grad(M)) - m_evap / rho_dry

Evaporation rate: m_evap = rho_dry * k_evap * M * max(0, T - T_threshold)
Diffusivity: D_eff = D0 * exp(-Ea / (R * T_K))
"""

from __future__ import annotations

try:
    import warp as wp

    @wp.kernel
    def moisture_step(
        M: wp.array3d(dtype=float),
        M_new: wp.array3d(dtype=float),
        T: wp.array3d(dtype=float),
        evap_rate: wp.array3d(dtype=float),
        cell_mask: wp.array3d(dtype=int),
        rho_dry: wp.array3d(dtype=float),
        D0: float, Ea: float, R_gas: float,
        k_evap: float, T_threshold: float,
        dx: float, dy: float, dz: float, dt: float,
        nx: int, ny: int, nz: int,
    ):
        """Advance moisture field by one timestep.

        dM/dt = div(D_eff * grad(M)) - m_evap / rho_dry
        """
        i, j, k = wp.tid()
        if i >= nx or j >= ny or k >= nz:
            return

        if cell_mask[i, j, k] != 1:
            M_new[i, j, k] = 0.0
            evap_rate[i, j, k] = 0.0
            return

        if i <= 0 or i >= nx - 1 or j <= 0 or j >= ny - 1 or k <= 0 or k >= nz - 1:
            M_new[i, j, k] = M[i, j, k]
            evap_rate[i, j, k] = 0.0
            return

        temp = T[i, j, k]
        moist = M[i, j, k]

        # Moisture diffusivity: D_eff = D0 * exp(-Ea / (R * T_K))
        T_K = temp + 273.15
        D_eff = D0 * wp.exp(-Ea / (R_gas * T_K))

        # Diffusion (central differences)
        lap_M = D_eff * (
            (M[i + 1, j, k] - 2.0 * moist + M[i - 1, j, k]) / (dx * dx)
            + (M[i, j + 1, k] - 2.0 * moist + M[i, j - 1, k]) / (dy * dy)
            + (M[i, j, k + 1] - 2.0 * moist + M[i, j, k - 1]) / (dz * dz)
        )

        # Evaporation
        rho_d = wp.max(rho_dry[i, j, k], 1.0)
        dT = wp.max(temp - T_threshold, 0.0)
        m_evap = rho_d * k_evap * wp.max(moist, 0.0) * dT
        evap_rate[i, j, k] = m_evap

        M_new[i, j, k] = wp.max(moist + dt * (lap_M - m_evap / rho_d), 0.0)

    _HAS_WARP = True

except Exception:
    # ImportError: warp not installed
    # RuntimeError: warp JIT compilation failed (e.g., PyInstaller bundle)
    _HAS_WARP = False
