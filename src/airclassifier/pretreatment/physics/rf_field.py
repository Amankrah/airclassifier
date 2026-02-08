"""
RF Electric Field Solver
========================

Solves the Laplace equation for the electrostatic potential in the
parallel-plate capacitor (oven applicator):

    div(eps' * grad(phi)) = 0

with Dirichlet BCs on the electrodes (phi = V_upper, phi = 0).
Computes |E|^2 = |grad(phi)|^2 for dielectric heating.

Phase 1: Uniform parallel-plate approximation with series-capacitor
         voltage division across the layered dielectric stack.
Phase 2: FDM with spatially varying eps'(T, M).
Phase 3: Warp FEM (warp.fem.Grid3D + warp.sparse.cg).
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

from ..config import MachineConfig, MaterialProperties


class RFFieldSolver:
    """RF electric field solver for the GP-15 applicator.

    Computes the electric field intensity |E|^2 in the oven domain
    given the electrode voltage, gap, and dielectric properties.

    Phase 1 uses a uniform parallel-plate model with series-capacitor
    voltage division across air / material / belt layers:

        E_layer_i = V_total / (eps'_i * sum(d_j / eps'_j))

    The layered stack (bottom to top):
        belt (PTFE, eps'≈2.1) | material (eps' from T,M) | air (eps'=1)
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        cell_sizes: Tuple[float, float, float],
        machine: MachineConfig,
        device: str = "cuda",
    ):
        self._grid_shape = grid_shape
        self._cell_sizes = cell_sizes
        self._machine = machine
        self._device = device

        # Pre-allocate field arrays
        nx, ny, nz = grid_shape
        self.potential = np.zeros(grid_shape, dtype=np.float32)      # phi [V]
        self.e_field_sq = np.zeros(grid_shape, dtype=np.float32)     # |E|^2 [V^2/m^2]

        # Cache for per-layer field values from the last solve
        self._E_air = 0.0
        self._E_bed = 0.0
        self._E_belt = 0.0

    # ------------------------------------------------------------------
    # Phase 1 — uniform parallel-plate with series-capacitor model
    # ------------------------------------------------------------------

    def solve(
        self,
        electrode_gap_m: float,
        voltage_kv: float,
        eps_real: Optional[np.ndarray] = None,
        cell_is_material: Optional[np.ndarray] = None,
        bed_depth_m: float = 0.05,
        belt_stack_m: float = 0.0035,
        eps_bed_avg: Optional[float] = None,
    ) -> np.ndarray:
        """Solve for |E|^2 given current electrode gap and dielectric field.

        Phase 1 computes a uniform field within each layer using the
        series-capacitor voltage division model from the engineering
        guide (Section 4.1.3).

        Args:
            electrode_gap_m: Current gap between electrodes [m].
            voltage_kv: RF voltage across the electrodes [kV].
            eps_real: Optional eps'(T,M) 3D field. If provided *and*
                ``cell_is_material`` is also given, the average eps'
                over material cells is used for voltage division.
                Otherwise ``eps_bed_avg`` is used.
            cell_is_material: Material mask (1 = material, 0 = air/belt).
            bed_depth_m: Material bed depth [m].
            belt_stack_m: Belt + wear strip + top sheet thickness [m].
            eps_bed_avg: Scalar average dielectric constant of the bed.
                Defaults to 5.0 if not provided and cannot be computed.

        Returns:
            |E|^2 array of shape *grid_shape* [V^2/m^2].
        """
        V_total = voltage_kv * 1000.0  # Convert kV → V

        # --- Layer thicknesses ---
        d_belt = belt_stack_m
        d_bed = bed_depth_m
        d_air = max(electrode_gap_m - d_belt - d_bed, 0.0)

        # --- Permittivities ---
        eps_belt = self._machine.belt_permittivity_real   # ~2.1  (PTFE)
        eps_air = 1.0

        # Determine average eps' for the bed layer
        if eps_real is not None and cell_is_material is not None:
            mat_mask = (cell_is_material == 1)
            if mat_mask.any():
                eps_bed = float(np.mean(eps_real[mat_mask]))
            else:
                eps_bed = eps_bed_avg if eps_bed_avg is not None else 5.0
        else:
            eps_bed = eps_bed_avg if eps_bed_avg is not None else 5.0
        eps_bed = max(eps_bed, 0.1)  # guard

        # --- Series-capacitor voltage division ---
        # V_total = E_air*d_air + E_bed*d_bed + E_belt*d_belt
        # D (displacement) is continuous → eps_i * E_i = const
        # so  E_i = D / eps_i  and  V = D * sum(d_j / eps_j)
        cap_sum = d_air / eps_air + d_bed / eps_bed + d_belt / eps_belt
        if cap_sum < 1e-12:
            self.e_field_sq[:] = 0.0
            return self.e_field_sq

        D = V_total / cap_sum  # displacement field [V/m] (unnormalised)

        self._E_air = D / eps_air
        self._E_bed = D / eps_bed
        self._E_belt = D / eps_belt

        # --- Fill the 3D field ---
        # Potential linearly interpolated from 0 (ground) to V_total.
        # |E|^2 is constant within each layer.
        nx, ny, nz = self._grid_shape
        dy = electrode_gap_m / ny

        for j in range(ny):
            y_centre = (j + 0.5) * dy
            if y_centre < d_belt:
                E = self._E_belt
            elif y_centre < d_belt + d_bed:
                E = self._E_bed
            else:
                E = self._E_air
            self.e_field_sq[:, j, :] = E * E

        # Build approximate potential (useful for diagnostics)
        # phi(y) integrated from y = 0 (ground, phi = 0)
        phi = 0.0
        for j in range(ny):
            y_centre = (j + 0.5) * dy
            if y_centre < d_belt:
                E = self._E_belt
            elif y_centre < d_belt + d_bed:
                E = self._E_bed
            else:
                E = self._E_air
            phi += E * dy
            self.potential[:, j, :] = phi

        return self.e_field_sq

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get_bed_field_strength(self) -> float:
        """Return E [V/m] in the material bed from the last solve."""
        return self._E_bed

    def compute_total_rf_power(
        self,
        eps_loss: np.ndarray,
        cell_is_material: np.ndarray,
        cell_volume_m3: float,
    ) -> float:
        """Integrate P_v over all material cells to get total RF power [W].

        P_v = 2*pi*f*eps_0 * eps'' * |E|^2

        Args:
            eps_loss: eps'' field.
            cell_is_material: Material mask (1 = material).
            cell_volume_m3: Volume of a single grid cell [m^3].

        Returns:
            Total RF power dissipated in the material [W].
        """
        TWO_PI_F_EPS0 = 1.5098e-3  # 2*pi * 27.12e6 * 8.854e-12
        mat_mask = (cell_is_material == 1)
        Pv = TWO_PI_F_EPS0 * eps_loss[mat_mask] * self.e_field_sq[mat_mask]
        return float(np.sum(Pv) * cell_volume_m3)
