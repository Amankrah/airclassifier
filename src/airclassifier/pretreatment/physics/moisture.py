"""
Moisture Solver
===============

Solves the moisture diffusion-evaporation equation:

    dM/dt = div(D_eff * grad(M)) - m_evap / rho_dry

where D_eff(T) follows an Arrhenius model and evaporation rate is
proportional to local temperature above a threshold.

Phase 1: Explicit FDM (forward Euler, NumPy vectorised).
Phase 2+: Warp GPU kernel.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

R_GAS = 8.314  # J/(mol*K)


class MoistureSolver:
    """Moisture diffusion and evaporation solver.

    Phase 1 uses explicit central differences for the diffusion term
    and a temperature-driven evaporation model from the engineering
    guide (Section 4.3.2):

        m_evap = rho_dry * k_evap * M * max(0, T - T_threshold)
    """

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

    # ------------------------------------------------------------------
    # Phase 1 — explicit FDM step (NumPy)
    # ------------------------------------------------------------------

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
        M_inlet_wb: float = 0.10,
    ):
        """Advance moisture field by one timestep.

        Implements::

            dM/dt = div(D_eff * grad(M))  -  m_evap / rho_dry

        where ``D_eff = D0 * exp(-Ea / (R * T_K))`` and
        ``m_evap = rho_dry * k_evap * M * max(0, T - T_threshold)``.

        Args:
            dt: Timestep [s].
            T: Temperature field [degC].
            cell_is_material: Material mask (1=material).
            rho_dry: Dry-basis bulk density [kg/m^3].
            D0: Pre-exponential diffusivity [m^2/s].
            Ea: Activation energy [J/mol].
            k_evap: Evaporation rate constant [1/(degC*s)].
            T_threshold: Temperature threshold for evaporation [degC].
            M_inlet_wb: Moisture content of fresh material at infeed.
        """
        dx, dy, dz = self._cell_sizes
        M = self.M
        Mn = self._M_new

        mat = (cell_is_material == 1)

        # Moisture diffusivity at each cell: D_eff = D0 * exp(-Ea / (R*T_K))
        T_K = T + 273.15
        D_eff = np.where(mat, D0 * np.exp(-Ea / (R_GAS * T_K)), 0.0).astype(np.float32)

        # --- Diffusion Laplacian (constant D per cell, central differences) ---
        # Interior slice [1:-1, 1:-1, 1:-1]
        lap_M = D_eff[1:-1, 1:-1, 1:-1] * (
            (M[2:, 1:-1, 1:-1] - 2.0 * M[1:-1, 1:-1, 1:-1] + M[:-2, 1:-1, 1:-1]) / (dx * dx)
            + (M[1:-1, 2:, 1:-1] - 2.0 * M[1:-1, 1:-1, 1:-1] + M[1:-1, :-2, 1:-1]) / (dy * dy)
            + (M[1:-1, 1:-1, 2:] - 2.0 * M[1:-1, 1:-1, 1:-1] + M[1:-1, 1:-1, :-2]) / (dz * dz)
        )

        # --- Evaporation ---
        rho_d = np.maximum(rho_dry[1:-1, 1:-1, 1:-1], 1.0)
        dT = np.maximum(T[1:-1, 1:-1, 1:-1] - T_threshold, 0.0)
        M_int = np.maximum(M[1:-1, 1:-1, 1:-1], 0.0)
        m_evap = rho_d * k_evap * M_int * dT
        self.evap_rate[1:-1, 1:-1, 1:-1] = m_evap

        # Forward Euler update (interior)
        Mn[1:-1, 1:-1, 1:-1] = np.maximum(
            M[1:-1, 1:-1, 1:-1] + dt * (lap_M - m_evap / rho_d),
            0.0,
        )

        # Non-material cells hold zero moisture
        Mn[cell_is_material != 1] = 0.0

        # --- Boundary conditions ---
        # Infeed (x=0): fresh material
        Mn[0, :, :] = np.where(cell_is_material[0, :, :] == 1, M_inlet_wb, 0.0)

        # Outfeed (x=-1): zero gradient
        Mn[-1, :, :] = Mn[-2, :, :]

        # Z edges: adiabatic
        Mn[:, :, 0] = Mn[:, :, 1]
        Mn[:, :, -1] = Mn[:, :, -2]

        # Y edges: zero flux
        Mn[:, 0, :] = Mn[:, 1, :]
        Mn[:, -1, :] = Mn[:, -2, :]

        # Re-enforce non-material = 0 after BCs
        Mn[cell_is_material != 1] = 0.0

        # Swap buffers
        self.M, self._M_new = Mn, M
