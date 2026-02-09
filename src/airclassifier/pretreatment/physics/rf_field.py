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
Phase 2: FDM Jacobi / SOR solve with spatially varying eps'(T, M).
         Power-constrained voltage iteration (Approach A).
Phase 3: Warp FEM (warp.fem.Grid3D + warp.sparse.cg).
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

from ..config import MachineConfig, MaterialProperties
from ..kernels.field_solve import (
    solve_laplace_jacobi,
    compute_gradient_sq_np,
)
from ..kernels.dielectric_heating import TWO_PI_F_EPS0


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
    # Phase 2 — FDM Laplace solve with per-cell eps'
    # ------------------------------------------------------------------

    def solve_fdm(
        self,
        electrode_gap_m: float,
        voltage_kv: float,
        eps_real: np.ndarray,
        max_iter: int = 500,
        tol: float = 1e-5,
    ) -> np.ndarray:
        """Solve div(eps' * grad(phi)) = 0 via Jacobi / SOR.

        This is the Phase 2 solver that respects the full spatial
        distribution of eps'(T, M).  It captures cell-by-cell field
        variations that the Phase 1 layer-average model cannot.

        Boundary conditions (from the engineering guide §4.1.2):
            phi = V_rf  on the upper electrode (j = ny-1)
            phi = 0     on the lower electrode (j = 0)
            dφ/dn = 0   on lateral faces (Neumann)

        Args:
            electrode_gap_m: Current electrode gap [m].
            voltage_kv: RF voltage across the electrodes [kV].
            eps_real: eps'(T, M) field of shape *grid_shape*.
            max_iter: Max SOR iterations.
            tol: Convergence tolerance.

        Returns:
            |E|^2 array of shape *grid_shape* [V^2/m^2].
        """
        V_total = voltage_kv * 1000.0
        dx, dy, dz = self._cell_sizes

        # Solve Laplace equation
        self.potential = solve_laplace_jacobi(
            eps=eps_real,
            V_upper=V_total,
            dx=dx, dy=dy, dz=dz,
            max_iter=max_iter,
            tol=tol,
            phi_init=self.potential,
        )

        # Compute |E|^2 = |grad(phi)|^2
        self.e_field_sq = compute_gradient_sq_np(self.potential, dx, dy, dz)

        # Update layer-average caches (for diagnostics)
        ny = self._grid_shape[1]
        belt_top_j = max(1, int(0.0035 / (electrode_gap_m / ny)))
        bed_top_j = min(ny - 1, belt_top_j + max(1, int(0.05 / (electrode_gap_m / ny))))
        E_bed_cells = self.e_field_sq[:, belt_top_j:bed_top_j, :]
        self._E_bed = float(np.sqrt(np.mean(E_bed_cells))) if E_bed_cells.size else 0.0
        self._E_air = float(np.sqrt(np.mean(
            self.e_field_sq[:, bed_top_j:, :]
        ))) if bed_top_j < ny else 0.0
        self._E_belt = float(np.sqrt(np.mean(
            self.e_field_sq[:, :belt_top_j, :]
        ))) if belt_top_j > 0 else 0.0

        return self.e_field_sq

    # ------------------------------------------------------------------
    # Power-constrained solve (Approach A, §4.1.3)
    # ------------------------------------------------------------------

    def solve_power_constrained(
        self,
        electrode_gap_m: float,
        target_power_kw: float,
        eps_real: np.ndarray,
        eps_loss: np.ndarray,
        cell_is_material: np.ndarray,
        cell_volume_m3: float,
        bed_depth_m: float = 0.05,
        belt_stack_m: float = 0.0035,
        use_fdm: bool = False,
        V_guess_kv: float | None = None,
        max_bisect: int = 30,
    ) -> float:
        """Find V_rf such that total dissipated power = target.

        Uses bisection on V_rf.  At each trial voltage the field is
        solved (Phase 1 or Phase 2) and the total power integrated.

        This implements **Approach A** from the engineering guide §4.1.3:
        given P_total from the recipe / anode current, find the
        corresponding electrode voltage.

        Args:
            electrode_gap_m: Current electrode gap [m].
            target_power_kw: Desired total RF power [kW].
            eps_real: eps'(T, M) field.
            eps_loss: eps''(T, M) field.
            cell_is_material: Material mask.
            cell_volume_m3: Cell volume [m^3].
            bed_depth_m: Material bed depth [m].
            belt_stack_m: Belt stack thickness [m].
            use_fdm: If True, use Phase 2 FDM solver. Else Phase 1.
            V_guess_kv: Initial guess for voltage [kV].
            max_bisect: Maximum bisection iterations.

        Returns:
            The converged V_rf [kV].
        """
        P_target_w = target_power_kw * 1000.0
        if P_target_w <= 0:
            self.e_field_sq[:] = 0.0
            return 0.0

        # Bisection bounds
        V_lo = 0.01  # kV
        V_hi = V_guess_kv * 2.0 if V_guess_kv else 20.0  # kV

        def _power_at_voltage(V_kv: float) -> float:
            if use_fdm:
                self.solve_fdm(electrode_gap_m, V_kv, eps_real)
            else:
                self.solve(
                    electrode_gap_m, V_kv,
                    eps_real=eps_real,
                    cell_is_material=cell_is_material,
                    bed_depth_m=bed_depth_m,
                    belt_stack_m=belt_stack_m,
                )
            return self.compute_total_rf_power(
                eps_loss, cell_is_material, cell_volume_m3,
            )

        # Ensure V_hi is high enough
        P_hi = _power_at_voltage(V_hi)
        while P_hi < P_target_w and V_hi < 100.0:
            V_hi *= 2.0
            P_hi = _power_at_voltage(V_hi)

        # Bisection
        for _ in range(max_bisect):
            V_mid = 0.5 * (V_lo + V_hi)
            P_mid = _power_at_voltage(V_mid)

            if abs(P_mid - P_target_w) / max(P_target_w, 1.0) < 0.01:
                return V_mid  # converged to 1%

            if P_mid < P_target_w:
                V_lo = V_mid
            else:
                V_hi = V_mid

        return 0.5 * (V_lo + V_hi)

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
        mat_mask = (cell_is_material == 1)
        Pv = TWO_PI_F_EPS0 * eps_loss[mat_mask] * self.e_field_sq[mat_mask]
        return float(np.sum(Pv) * cell_volume_m3)
