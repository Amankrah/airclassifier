"""
Material Property Functions
============================

Standalone functions for computing temperature- and moisture-dependent
material properties. These are the Python reference implementations;
the Warp kernel versions are in kernels/dielectric_heating.py.

Functions are used for:
- Pre-simulation analysis and validation
- Post-processing and plotting
- Unit tests against the Warp kernel versions
"""

from __future__ import annotations

import math


def dielectric_loss(T_c: float, M_wb: float, coeffs: tuple) -> float:
    """Dielectric loss factor eps''(T, M) at 27.12 MHz.

    eps'' = a1*M^2 + a2*M + a3*M*T + a4*T + a5
    """
    a1, a2, a3, a4, a5 = coeffs
    return a1 * M_wb**2 + a2 * M_wb + a3 * M_wb * T_c + a4 * T_c + a5


def dielectric_constant(T_c: float, M_wb: float, coeffs: tuple) -> float:
    """Dielectric constant eps'(T, M) at 27.12 MHz.

    eps' = b1*M + b2*T + b3
    """
    b1, b2, b3 = coeffs
    return b1 * M_wb + b2 * T_c + b3


def specific_heat(M_wb: float, c_p_dry: float, c_p_water: float = 4186.0) -> float:
    """Specific heat c_p(M) via linear mixing [J/(kg*K)]."""
    return c_p_dry * (1.0 - M_wb) + c_p_water * M_wb


def thermal_conductivity(
    M_wb: float, k_dry: float, k_beta: float, porosity: float
) -> float:
    """Effective thermal conductivity k(M) [W/(m*K)]."""
    k_solid = k_dry * (1.0 + k_beta * M_wb)
    k_air = 0.026
    return k_solid * (1.0 - porosity) + k_air * porosity


def bulk_density(M_wb: float, rho_solid: float, porosity: float) -> float:
    """Bulk density rho(M) [kg/m^3]."""
    M_db = M_wb / max(1.0 - M_wb, 1e-6)
    return rho_solid * (1.0 - porosity) * (1.0 + M_db)


def moisture_diffusivity(T_c: float, D0: float, Ea: float) -> float:
    """Effective moisture diffusivity D_eff(T) [m^2/s].

    Arrhenius model: D_eff = D0 * exp(-Ea / (R * T_K))
    """
    T_K = T_c + 273.15
    R = 8.314
    return D0 * math.exp(-Ea / (R * T_K))


def rf_power_density(
    e_field_sq: float, eps_loss: float, frequency_hz: float = 27.12e6
) -> float:
    """Volumetric RF power density P_v [W/m^3].

    P_v = 2*pi*f*eps_0*eps''*|E|^2
    """
    eps_0 = 8.854e-12
    return 2.0 * math.pi * frequency_hz * eps_0 * eps_loss * e_field_sq
