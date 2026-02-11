"""
Thermal Solver
==============

Solves the heat equation in the material bed:

    rho*c_p * dT/dt = div(k * grad(T)) + P_v - L_v * m_evap

where P_v is the RF volumetric heating source and L_v * m_evap is
the latent heat sink from moisture evaporation.

Phase 1: Explicit FDM (forward Euler, CFL-constrained dt) — NumPy.
Phase 2+: Warp GPU kernel (@wp.kernel).
Phase 3: Implicit backward Euler via Warp sparse CG.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class ThermalSolver:
    """Heat equation solver for the material bed.

    Phase 1 uses an explicit finite-difference method with forward Euler
    time integration and second-order central differences for the
    Laplacian.  Variable conductivity is handled via half-cell harmonic
    averaging of neighbouring ``k_eff`` values.

    Boundary conditions are applied *after* the interior update:

    * **Infeed (i=0):**   Dirichlet — T = T_inlet
    * **Outfeed (i=nx-1):** Neumann — dT/dx = 0 (copy from i=nx-2)
    * **Belt edges (k=0, k=nz-1):** Adiabatic dT/dz = 0
    * **Bottom (j=0):** Contact through belt to electrode (isothermal sink)
    * **Top of bed:** Convective BC applied separately via
      :meth:`apply_convection_bc`
    """

    # Belt thermal properties for contact BC at lower electrode
    _K_BELT = 0.25       # PTFE thermal conductivity [W/(m*K)]
    _D_BELT = 0.0035     # Belt stack thickness [m]

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        cell_sizes: Tuple[float, float, float],
        device: str = "cuda",
        belt_stack_m: float = 0.0035,
    ):
        self._grid_shape = grid_shape
        self._cell_sizes = cell_sizes
        self._device = device
        self._D_BELT = belt_stack_m

        nx, ny, nz = grid_shape
        self.T = np.zeros(grid_shape, dtype=np.float32)         # Temperature [degC]
        self._T_new = np.zeros(grid_shape, dtype=np.float32)    # Double-buffer

    def initialize(self, T_initial_c: float):
        """Set uniform initial temperature."""
        self.T[:] = T_initial_c

    # ------------------------------------------------------------------
    # Phase 1  —  explicit FDM step (NumPy vectorised)
    # ------------------------------------------------------------------

    def step(
        self,
        dt: float,
        P_v: np.ndarray,
        evap_rate: np.ndarray,
        rho_cp: np.ndarray,
        k_eff: np.ndarray,
        cell_is_material: np.ndarray,
        L_v: float = 2.26e6,
        T_inlet_c: float = 22.0,
        T_electrode_c: float | None = None,
    ):
        """Advance temperature field by one explicit timestep.

        Implements::

            T_new = T + dt/(rho*c_p) * [div(k grad T) + P_v - L_v * evap]

        with second-order central differences for the Laplacian.
        Boundary cells are set after the interior update.

        Args:
            dt: Timestep [s].
            P_v: RF power density [W/m^3].
            evap_rate: Evaporation rate [kg/(m^3*s)].
            rho_cp: rho*c_p [J/(m^3*K)].
            k_eff: Effective thermal conductivity [W/(m*K)].
            cell_is_material: Material mask (1=material, 2=belt, 0=air).
            L_v: Latent heat of vaporization [J/kg].
            T_inlet_c: Inlet temperature for Dirichlet BC [degC].
            T_electrode_c: Lower electrode temperature [degC] for the
                Robin BC at j=0.  If None, defaults to T_inlet_c
                (legacy: cold electrode at ambient).
        """
        dx, dy, dz = self._cell_sizes
        T = self.T
        Tn = self._T_new

        # Guard against division by zero
        rc = np.maximum(rho_cp, 1.0)

        # --- Interior Laplacian with variable k (central differences) ---
        # k at half-cell faces (arithmetic average of neighbours)
        # X direction
        k_xp = 0.5 * (k_eff[:-2, 1:-1, 1:-1] + k_eff[2:, 1:-1, 1:-1])
        k_xm = 0.5 * (k_eff[1:-1, 1:-1, 1:-1] + k_eff[:-2, 1:-1, 1:-1])
        # Note: k_xm uses left neighbour. Let me be more careful:
        # face i+1/2:  avg(k[i], k[i+1])   →  (k_eff[1:-1] + k_eff[2:])/2
        # face i-1/2:  avg(k[i-1], k[i])   →  (k_eff[:-2] + k_eff[1:-1])/2
        k_xp2 = 0.5 * (k_eff[1:-1, 1:-1, 1:-1] + k_eff[2:, 1:-1, 1:-1])
        k_xm2 = 0.5 * (k_eff[:-2, 1:-1, 1:-1] + k_eff[1:-1, 1:-1, 1:-1])
        lap_x = (
            k_xp2 * (T[2:, 1:-1, 1:-1] - T[1:-1, 1:-1, 1:-1])
            - k_xm2 * (T[1:-1, 1:-1, 1:-1] - T[:-2, 1:-1, 1:-1])
        ) / (dx * dx)

        # Y direction
        k_yp = 0.5 * (k_eff[1:-1, 1:-1, 1:-1] + k_eff[1:-1, 2:, 1:-1])
        k_ym = 0.5 * (k_eff[1:-1, :-2, 1:-1] + k_eff[1:-1, 1:-1, 1:-1])
        lap_y = (
            k_yp * (T[1:-1, 2:, 1:-1] - T[1:-1, 1:-1, 1:-1])
            - k_ym * (T[1:-1, 1:-1, 1:-1] - T[1:-1, :-2, 1:-1])
        ) / (dy * dy)

        # Z direction
        k_zp = 0.5 * (k_eff[1:-1, 1:-1, 1:-1] + k_eff[1:-1, 1:-1, 2:])
        k_zm = 0.5 * (k_eff[1:-1, 1:-1, :-2] + k_eff[1:-1, 1:-1, 1:-1])
        lap_z = (
            k_zp * (T[1:-1, 1:-1, 2:] - T[1:-1, 1:-1, 1:-1])
            - k_zm * (T[1:-1, 1:-1, 1:-1] - T[1:-1, 1:-1, :-2])
        ) / (dz * dz)

        laplacian = lap_x + lap_y + lap_z

        # Source and sink (interior only)
        source = P_v[1:-1, 1:-1, 1:-1]
        sink = L_v * evap_rate[1:-1, 1:-1, 1:-1]

        # Forward Euler update
        Tn[1:-1, 1:-1, 1:-1] = (
            T[1:-1, 1:-1, 1:-1]
            + dt / rc[1:-1, 1:-1, 1:-1] * (laplacian + source - sink)
        )

        # Guard against NaN / Inf (numerical safety net)
        np.nan_to_num(Tn, copy=False, nan=T_inlet_c, posinf=200.0, neginf=T_inlet_c)
        # Physical clamp: temperature cannot go below 0 °C or above 200 °C
        # (GP-15 processes below 100 °C; 200 °C is a generous safety margin
        # that catches runaway heating without masking real physics)
        np.clip(Tn, 0.0, 200.0, out=Tn)

        # --- Boundary conditions ---
        # Infeed (x=0): Dirichlet — fresh material at T_inlet
        Tn[0, :, :] = T_inlet_c

        # Outfeed (x=nx-1): Neumann dT/dx = 0 (copy from interior)
        Tn[-1, :, :] = Tn[-2, :, :]

        # Belt edges (z=0, z=nz-1): adiabatic dT/dz = 0
        Tn[:, :, 0] = Tn[:, :, 1]
        Tn[:, :, -1] = Tn[:, :, -2]

        # Bottom (j=0): Robin BC for material cells — models heat
        # transfer through the PTFE belt stack to the lower
        # electrode / aluminum trays.
        #
        # Robin BC:  q = h_contact * (T - T_electrode)
        # where h_contact = k_belt / d_belt ≈ 71 W/(m²·K)
        #
        # T_electrode tracks the lumped tray temperature (updated
        # in coupling.py).  Falls back to T_inlet_c if not provided.
        T_elec = T_electrode_c if T_electrode_c is not None else T_inlet_c
        mat_j0 = (cell_is_material[:, 0, :] == 1)
        if np.any(mat_j0):
            h_contact = self._K_BELT / max(self._D_BELT, 1e-6)
            q_contact = h_contact * (Tn[:, 0, :] - T_elec)
            rc_0 = np.maximum(rho_cp[:, 0, :], 1.0)
            correction = dt * q_contact / (rc_0 * dy * 0.5)
            Tn[:, 0, :] = np.where(mat_j0,
                                    Tn[:, 0, :] - correction,
                                    T_elec)
        else:
            Tn[:, 0, :] = T_elec

        # Top (j=ny-1): copy for now — convection BC applied separately
        Tn[:, -1, :] = Tn[:, -2, :]

        # --- Swap buffers ---
        self.T, self._T_new = Tn, T

    # ------------------------------------------------------------------
    # Convective BC at bed surface
    # ------------------------------------------------------------------

    def apply_convection_bc(
        self,
        j_surface: int,
        h_conv: float,
        T_air_c: float,
        rho_cp: np.ndarray,
        k_eff: np.ndarray,
        dt: float,
    ):
        """Apply convective heat transfer at the material bed surface.

        ``-k dT/dy |_surface = h_conv * (T_surface - T_air)``

        Modifies ``self.T`` in-place at row *j_surface*.

        Args:
            j_surface: Y-index of the bed surface.
            h_conv: Convective heat transfer coefficient [W/(m^2*K)].
            T_air_c: Oven air temperature [degC].
            rho_cp: rho*c_p field.
            k_eff: Effective conductivity field.
            dt: Timestep [s].
        """
        _, dy, _ = self._cell_sizes[0], self._cell_sizes[1], self._cell_sizes[2]
        dy = self._cell_sizes[1]

        T_s = self.T[:, j_surface, :]
        q_conv = h_conv * (T_s - T_air_c)
        rc = np.maximum(rho_cp[:, j_surface, :], 1.0)

        # Surface flux applied over half-cell thickness
        self.T[:, j_surface, :] = T_s - dt * q_conv / (rc * dy * 0.5)

    # ------------------------------------------------------------------
    # CFL stability limit
    # ------------------------------------------------------------------

    def get_cfl_dt(self, k_max: float, rho_cp_min: float) -> float:
        """Compute CFL-constrained maximum timestep.

        For the 3D explicit scheme the stability condition is::

            dt < (1/6) * min(dx,dy,dz)^2 * rho_cp_min / k_max

        We use a conservative factor of 0.4 instead of 1/6 ≈ 0.167.
        """
        dx, dy, dz = self._cell_sizes
        dmin = min(dx, dy, dz)
        rho_cp_min = max(rho_cp_min, 1.0)
        return 0.4 * dmin * dmin * rho_cp_min / max(k_max, 1e-6)
