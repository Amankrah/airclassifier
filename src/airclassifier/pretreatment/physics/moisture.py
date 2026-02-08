"""
Moisture Solver
===============

Solves the moisture diffusion-evaporation equation:

    dM/dt = div(D_eff * grad(M)) - m_evap / rho_dry

where D_eff(T) follows an Arrhenius model and evaporation rate is
proportional to local temperature above a threshold.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class MoistureSolver:
    """Moisture diffusion and evaporation solver."""

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
        self.M = np.zeros(grid_shape, dtype=np.float32)          # Moisture [wb fraction]
        self._M_new = np.zeros(grid_shape, dtype=np.float32)
        self.evap_rate = np.zeros(grid_shape, dtype=np.float32)  # kg/(m^3*s)

    def initialize(self, M_initial_wb: float):
        """Set uniform initial moisture content."""
        self.M[:] = M_initial_wb

    def step(
        self,
        dt: float,
        T: np.ndarray,
        cell_is_material: np.ndarray,
        rho_dry: np.ndarray,
        D0: float,
        Ea: float,
        k_evap: float,
        T_threshold: float,
    ):
        """Advance moisture field by one timestep.

        Args:
            dt: Timestep [s].
            T: Temperature field [degC].
            cell_is_material: Material mask.
            rho_dry: Dry-basis bulk density [kg/m^3].
            D0: Pre-exponential diffusivity [m^2/s].
            Ea: Activation energy [J/mol].
            k_evap: Evaporation rate constant [1/(degC*s)].
            T_threshold: Temperature threshold for evaporation [degC].
        """
        # TODO: Launch Warp kernel moisture_step
        raise NotImplementedError
