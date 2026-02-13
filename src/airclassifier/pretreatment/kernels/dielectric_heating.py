"""
Dielectric Heating Kernels
==========================

RF power density computation and material property update.

P_v = 2*pi*f*eps_0*eps''*|E|^2   [W/m^3]

Provides both NumPy (Phase 1) and Warp GPU (Phase 4) implementations.
The caller selects the backend via the function suffix (_np vs _wp).
"""

from __future__ import annotations

import numpy as np

# Precomputed constant: 2*pi * 27.12e6 * 8.854e-12
TWO_PI_F_EPS0 = 1.5098e-3  # for 27.12 MHz

# ── Warp GPU kernels ─────────────────────────────────────────────────

try:
    import warp as wp

    @wp.kernel
    def _compute_power_density_kernel(
        e_field_sq: wp.array3d(dtype=float),
        eps_loss: wp.array3d(dtype=float),
        power_density: wp.array3d(dtype=float),
        two_pi_f_eps0: float,
        nx: int,
        ny: int,
        nz: int,
    ):
        """P_v = 2*pi*f*eps_0 * eps'' * |E|^2  at each cell."""
        i, j, k = wp.tid()
        if i >= nx or j >= ny or k >= nz:
            return
        power_density[i, j, k] = two_pi_f_eps0 * eps_loss[i, j, k] * e_field_sq[i, j, k]

    @wp.kernel
    def _update_properties_kernel(
        T: wp.array3d(dtype=float),
        M: wp.array3d(dtype=float),
        cell_mask: wp.array3d(dtype=int),
        eps_loss_out: wp.array3d(dtype=float),
        eps_real_out: wp.array3d(dtype=float),
        rho_cp_out: wp.array3d(dtype=float),
        k_eff_out: wp.array3d(dtype=float),
        a1: float, a2: float, a3: float, a4: float, a5: float,
        b1: float, b2: float, b3: float,
        c_p_dry: float, c_p_water: float,
        k_dry: float, k_beta: float,
        rho_solid: float, porosity: float,
        k_dispersion: float,
        nx: int, ny: int, nz: int,
    ):
        """Update all material properties from current T and M."""
        i, j, k = wp.tid()
        if i >= nx or j >= ny or k >= nz:
            return

        k_air = 0.026
        zone = cell_mask[i, j, k]

        if zone == 0:  # Air
            eps_loss_out[i, j, k] = 0.0
            eps_real_out[i, j, k] = 1.0
            rho_cp_out[i, j, k] = 1.2 * 1005.0
            k_eff_out[i, j, k] = k_air
            return
        if zone == 2:  # Belt
            eps_loss_out[i, j, k] = 0.0003
            eps_real_out[i, j, k] = 2.1
            rho_cp_out[i, j, k] = 2200.0 * 1000.0
            k_eff_out[i, j, k] = 0.25
            return

        # Material (zone == 1)
        temp = T[i, j, k]
        moist = M[i, j, k]

        # For batch runs, M≈0 means "empty cell" (belt cleared), not "very dry material".
        # Cells with negligible moisture should not absorb RF power.
        # This enables proper Ia drop to idle after batch clears.
        if moist < 0.005:  # Below 0.5% moisture = effectively empty
            eps_loss_out[i, j, k] = 0.0
            eps_real_out[i, j, k] = 1.0  # Air
        else:
            eps_loss_out[i, j, k] = wp.max(a1 * moist * moist + a2 * moist + a3 * moist * temp + a4 * temp + a5, 0.01)
            eps_real_out[i, j, k] = wp.max(b1 * moist + b2 * temp + b3, 1.5)

        cp = c_p_dry * (1.0 - moist) + c_p_water * moist
        M_db = moist / wp.max(1.0 - moist, 1.0e-6)
        rho = rho_solid * (1.0 - porosity) * (1.0 + M_db)
        rho_cp_out[i, j, k] = rho * cp

        k_solid = k_dry * (1.0 + k_beta * moist)
        k_eff_out[i, j, k] = k_solid * (1.0 - porosity) + k_air * porosity + k_dispersion

    def compute_power_density_wp(
        e_field_sq: wp.array3d,
        eps_loss: wp.array3d,
        power_density: wp.array3d,
        device: str = "cuda",
    ):
        """Launch the Warp P_v kernel."""
        nx, ny, nz = e_field_sq.shape
        wp.launch(
            kernel=_compute_power_density_kernel,
            dim=(nx, ny, nz),
            inputs=[e_field_sq, eps_loss, power_density, TWO_PI_F_EPS0, nx, ny, nz],
            device=device,
        )

    def update_material_properties_wp(
        T, M, cell_mask, eps_loss_out, eps_real_out, rho_cp_out, k_eff_out,
        a1, a2, a3, a4, a5, b1, b2, b3,
        c_p_dry, c_p_water, k_dry, k_beta, rho_solid, porosity,
        k_dispersion=0.0,
        device="cuda",
    ):
        """Launch the Warp property update kernel."""
        nx, ny, nz = T.shape
        wp.launch(
            kernel=_update_properties_kernel,
            dim=(nx, ny, nz),
            inputs=[
                T, M, cell_mask, eps_loss_out, eps_real_out, rho_cp_out, k_eff_out,
                a1, a2, a3, a4, a5, b1, b2, b3,
                c_p_dry, c_p_water, k_dry, k_beta, rho_solid, porosity,
                k_dispersion,
                nx, ny, nz,
            ],
            device=device,
        )

    _HAS_WARP = True

except ImportError:
    _HAS_WARP = False


# ── NumPy implementations (always available) ─────────────────────────

def compute_power_density_np(
    e_field_sq: np.ndarray,
    eps_loss: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """P_v = 2*pi*f*eps_0 * eps'' * |E|^2  [W/m^3]."""
    if out is None:
        out = np.empty_like(e_field_sq)
    np.multiply(eps_loss, e_field_sq, out=out)
    out *= TWO_PI_F_EPS0
    return out


def update_material_properties_np(
    T: np.ndarray,
    M: np.ndarray,
    cell_is_material: np.ndarray,
    *,
    a1: float, a2: float, a3: float, a4: float, a5: float,
    b1: float, b2: float, b3: float,
    c_p_dry: float, c_p_water: float,
    k_dry: float, k_beta: float,
    rho_solid: float, porosity: float,
    k_dispersion: float = 0.0,
    eps_loss_out: np.ndarray,
    eps_real_out: np.ndarray,
    rho_cp_out: np.ndarray,
    k_eff_out: np.ndarray,
) -> None:
    """Update all material properties from current T and M (NumPy)."""
    k_air = 0.026
    rho_air_cp_air = 1.2 * 1005.0

    mat = (cell_is_material == 1)
    belt = (cell_is_material == 2)
    air = (cell_is_material == 0)

    eps_loss_out[air] = 0.0
    eps_real_out[air] = 1.0
    rho_cp_out[air] = rho_air_cp_air
    k_eff_out[air] = k_air

    eps_loss_out[belt] = 0.0003
    eps_real_out[belt] = 2.1
    rho_cp_out[belt] = 2200.0 * 1000.0
    k_eff_out[belt] = 0.25

    if not mat.any():
        return

    Tm = T[mat]
    Mm = M[mat]

    # For batch runs, M≈0 means "empty cell" (belt cleared), not "very dry material".
    # Cells with negligible moisture should not absorb RF power.
    # This enables proper Ia drop to idle after batch clears.
    empty_threshold = 0.005  # Below 0.5% moisture = effectively empty

    # Compute normal dielectric properties
    eps_loss_vals = a1 * Mm * Mm + a2 * Mm + a3 * Mm * Tm + a4 * Tm + a5
    np.clip(eps_loss_vals, 0.01, None, out=eps_loss_vals)
    eps_real_vals = b1 * Mm + b2 * Tm + b3
    np.clip(eps_real_vals, 1.5, None, out=eps_real_vals)

    # Override empty cells (M < threshold) with air properties
    # This treats "empty belt" correctly rather than as "very dry material"
    is_empty = (Mm < empty_threshold)
    eps_loss_vals[is_empty] = 0.0
    eps_real_vals[is_empty] = 1.0

    eps_loss_out[mat] = eps_loss_vals
    eps_real_out[mat] = eps_real_vals

    # Thermal properties
    cp = c_p_dry * (1.0 - Mm) + c_p_water * Mm
    M_db = Mm / np.maximum(1.0 - Mm, 1.0e-6)
    rho = rho_solid * (1.0 - porosity) * (1.0 + M_db)
    rho_cp_vals = rho * cp

    k_solid = k_dry * (1.0 + k_beta * Mm)
    k_eff_vals = k_solid * (1.0 - porosity) + k_air * porosity + k_dispersion

    # Empty cells also get air thermal properties
    rho_cp_vals[is_empty] = rho_air_cp_air
    k_eff_vals[is_empty] = k_air

    rho_cp_out[mat] = rho_cp_vals
    k_eff_out[mat] = k_eff_vals
