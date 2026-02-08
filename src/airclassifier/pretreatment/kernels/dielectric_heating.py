"""
Dielectric Heating Kernels
==========================

RF power density computation and material property update.

P_v = 2*pi*f*eps_0*eps''*|E|^2   [W/m^3]

Phase 1 uses NumPy vectorised implementations.
Phase 2+ will replace these with @wp.kernel equivalents.
"""

from __future__ import annotations

import numpy as np

# Precomputed constant: 2*pi * 27.12e6 * 8.854e-12
TWO_PI_F_EPS0 = 1.5098e-3  # for 27.12 MHz


# ── Phase 1: NumPy implementations ──────────────────────────────────

def compute_power_density_np(
    e_field_sq: np.ndarray,
    eps_loss: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute volumetric RF power density at each grid cell.

    P_v = 2*pi * f * eps_0 * eps'' * |E|^2    [W/m^3]

    Args:
        e_field_sq: |E|^2 field [V^2/m^2].
        eps_loss: eps'' field (dielectric loss factor).
        out: Optional pre-allocated output array.

    Returns:
        P_v array of same shape as *e_field_sq* [W/m^3].
    """
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
    # Loss factor coefficients: eps'' = a1*M^2 + a2*M + a3*M*T + a4*T + a5
    a1: float, a2: float, a3: float, a4: float, a5: float,
    # Dielectric constant: eps' = b1*M + b2*T + b3
    b1: float, b2: float, b3: float,
    # Thermal
    c_p_dry: float, c_p_water: float,
    k_dry: float, k_beta: float,
    rho_solid: float, porosity: float,
    # Outputs (pre-allocated)
    eps_loss_out: np.ndarray,
    eps_real_out: np.ndarray,
    rho_cp_out: np.ndarray,
    k_eff_out: np.ndarray,
) -> None:
    """Update all material properties from current T and M fields.

    Air / belt cells get inert properties (eps'' = 0, eps' = 1 or 2.1).
    Material cells are computed from the parameterised models.
    """
    k_air = 0.026  # W/(m*K) at ~50 °C
    rho_air_cp_air = 1.2 * 1005.0  # air rho*c_p

    mat = (cell_is_material == 1)
    belt = (cell_is_material == 2)
    air = (cell_is_material == 0)

    # ── Air cells ──
    eps_loss_out[air] = 0.0
    eps_real_out[air] = 1.0
    rho_cp_out[air] = rho_air_cp_air
    k_eff_out[air] = k_air

    # ── Belt cells ──
    eps_loss_out[belt] = 0.0003
    eps_real_out[belt] = 2.1
    rho_cp_out[belt] = 2200.0 * 1000.0  # PTFE approx rho*c_p
    k_eff_out[belt] = 0.25              # PTFE conductivity

    # ── Material cells ──
    if not mat.any():
        return

    Tm = T[mat]
    Mm = M[mat]

    # Dielectric loss factor
    eps_loss_out[mat] = a1 * Mm * Mm + a2 * Mm + a3 * Mm * Tm + a4 * Tm + a5

    # Dielectric constant
    eps_real_out[mat] = b1 * Mm + b2 * Tm + b3

    # Specific heat (linear mixing)
    cp = c_p_dry * (1.0 - Mm) + c_p_water * Mm

    # Bulk density
    M_db = Mm / np.maximum(1.0 - Mm, 1.0e-6)
    rho = rho_solid * (1.0 - porosity) * (1.0 + M_db)

    rho_cp_out[mat] = rho * cp

    # Thermal conductivity
    k_solid = k_dry * (1.0 + k_beta * Mm)
    k_eff_out[mat] = k_solid * (1.0 - porosity) + k_air * porosity


# ── Warp kernel stubs (Phase 2+) ────────────────────────────────────

# @wp.kernel
# def compute_power_density(...):  ...
# @wp.kernel
# def update_material_properties(...):  ...
