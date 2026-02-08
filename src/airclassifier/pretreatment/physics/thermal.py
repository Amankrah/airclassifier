"""
Thermal Solver
==============

Solves the heat equation in the material bed:

    rho*c_p * dT/dt = div(k * grad(T)) + P_v - L_v * m_evap

where P_v is the RF volumetric heating source and L_v * m_evap is
the latent heat sink from moisture evaporation.

Phase 1: Explicit FDM (forward Euler, CFL-constrained dt).
Phase 3: Implicit backward Euler via Warp sparse CG.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class ThermalSolver:
    """Heat equation solver for the material bed."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        cell_sizes: Tuple[float, float, float],
        device: str = "cuda",
    ):
        self._grid_shape = grid_shape
        self._cell_sizes = cell_sizes
        self._device = device

        nx, ny, nz = grid_shape
        self.T = np.zeros(grid_shape, dtype=np.float32)         # Temperature [degC]
        self._T_new = np.zeros(grid_shape, dtype=np.float32)    # Double-buffer

    def initialize(self, T_initial_c: float):
        """Set uniform initial temperature."""
        self.T[:] = T_initial_c

    def step(
        self,
        dt: float,
        P_v: np.ndarray,
        evap_rate: np.ndarray,
        rho_cp: np.ndarray,
        k_eff: np.ndarray,
        cell_is_material: np.ndarray,
        L_v: float = 2.26e6,
    ):
        """Advance temperature field by one explicit timestep.

        Args:
            dt: Timestep [s].
            P_v: RF power density [W/m^3].
            evap_rate: Evaporation rate [kg/(m^3*s)].
            rho_cp: rho*c_p [J/(m^3*K)].
            k_eff: Effective thermal conductivity [W/(m*K)].
            cell_is_material: Material mask (1=material, 0=air).
            L_v: Latent heat of vaporization [J/kg].
        """
        # TODO: Launch Warp kernel heat_conduction_step
        raise NotImplementedError

    def get_cfl_dt(self, k_max: float, rho_cp_min: float) -> float:
        """Compute CFL-constrained maximum timestep.

        dt < 0.5 * min(dx,dy,dz)^2 * rho_cp_min / k_max
        """
        dx, dy, dz = self._cell_sizes
        dmin = min(dx, dy, dz)
        rho_cp_min = max(rho_cp_min, 1.0)
        return 0.5 * dmin * dmin * rho_cp_min / max(k_max, 1e-6)
