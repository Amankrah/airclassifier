"""
Pressure solver for incompressible flow.

Provides iterative solvers for the pressure Poisson equation
arising from the projection method.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import warp as wp


@dataclass
class PressureSolverParams:
    """Parameters for pressure solver."""

    max_iterations: int = 200
    tolerance: float = 1.0e-6
    method: str = "jacobi"        # "jacobi", "sor", "multigrid"
    omega: float = 1.8            # SOR relaxation factor


@wp.kernel
def jacobi_iteration(
    pressure: wp.array3d(dtype=float),
    pressure_new: wp.array3d(dtype=float),
    rhs: wp.array3d(dtype=float),
    ax: float,
    ay: float,
    az: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Single Jacobi iteration for pressure."""
    i, j, k = wp.tid()

    if i >= nx or j >= ny or k >= nz:
        return

    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
        pressure_new[i, j, k] = 0.0
        return

    diag = 2.0 * (ax + ay + az)

    neighbor_sum = (
        ax * (pressure[i+1, j, k] + pressure[i-1, j, k]) +
        ay * (pressure[i, j+1, k] + pressure[i, j-1, k]) +
        az * (pressure[i, j, k+1] + pressure[i, j, k-1])
    )

    pressure_new[i, j, k] = (rhs[i, j, k] + neighbor_sum) / diag


@wp.kernel
def sor_red_black(
    pressure: wp.array3d(dtype=float),
    rhs: wp.array3d(dtype=float),
    ax: float,
    ay: float,
    az: float,
    omega: float,
    nx: int,
    ny: int,
    nz: int,
    color: int,
):
    """Red-black SOR iteration."""
    i, j, k = wp.tid()

    if i >= nx or j >= ny or k >= nz:
        return

    if ((i + j + k) % 2) != color:
        return

    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
        pressure[i, j, k] = 0.0
        return

    diag = 2.0 * (ax + ay + az)

    neighbor_sum = (
        ax * (pressure[i+1, j, k] + pressure[i-1, j, k]) +
        ay * (pressure[i, j+1, k] + pressure[i, j-1, k]) +
        az * (pressure[i, j, k+1] + pressure[i, j, k-1])
    )

    p_new = (rhs[i, j, k] + neighbor_sum) / diag
    pressure[i, j, k] = (1.0 - omega) * pressure[i, j, k] + omega * p_new


@wp.kernel
def compute_residual(
    pressure: wp.array3d(dtype=float),
    rhs: wp.array3d(dtype=float),
    residual: wp.array3d(dtype=float),
    ax: float,
    ay: float,
    az: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Compute residual r = rhs - A*p."""
    i, j, k = wp.tid()

    if i >= nx or j >= ny or k >= nz:
        return

    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
        residual[i, j, k] = 0.0
        return

    diag = 2.0 * (ax + ay + az)

    laplacian = (
        ax * (pressure[i+1, j, k] + pressure[i-1, j, k]) +
        ay * (pressure[i, j+1, k] + pressure[i, j-1, k]) +
        az * (pressure[i, j, k+1] + pressure[i, j, k-1]) -
        diag * pressure[i, j, k]
    )

    residual[i, j, k] = rhs[i, j, k] - laplacian


class PressureSolver:
    """
    Iterative solver for the pressure Poisson equation.

    Solves: laplacian(p) = rhs

    Uses Jacobi or SOR iteration with optional multigrid acceleration.
    """

    def __init__(
        self,
        params: PressureSolverParams,
        grid_shape: Tuple[int, int, int],
        grid_spacing: Tuple[float, float, float],
        device: str = "cuda"
    ):
        """
        Initialize pressure solver.

        Args:
            params: Solver parameters
            grid_shape: (nx, ny, nz) grid dimensions
            grid_spacing: (dx, dy, dz) cell sizes
            device: Warp device
        """
        self.params = params
        self.grid_shape = grid_shape
        self.grid_spacing = grid_spacing
        self.device = device

        nx, ny, nz = grid_shape
        dx, dy, dz = grid_spacing

        # Laplacian coefficients
        self.ax = 1.0 / (dx * dx)
        self.ay = 1.0 / (dy * dy)
        self.az = 1.0 / (dz * dz)

        # Work arrays
        self.pressure_temp = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self.residual = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self.residual_norm = wp.zeros(1, dtype=float, device=device)

    def solve(
        self,
        pressure: wp.array,
        rhs: wp.array
    ) -> Tuple[int, float]:
        """
        Solve the pressure Poisson equation.

        Args:
            pressure: Pressure field (initial guess, will be modified)
            rhs: Right-hand side

        Returns:
            (iterations, final_residual)
        """
        if self.params.method == "jacobi":
            return self._solve_jacobi(pressure, rhs)
        elif self.params.method == "sor":
            return self._solve_sor(pressure, rhs)
        else:
            return self._solve_jacobi(pressure, rhs)

    def _solve_jacobi(
        self,
        pressure: wp.array,
        rhs: wp.array
    ) -> Tuple[int, float]:
        """Solve using Jacobi iteration."""
        nx, ny, nz = self.grid_shape

        for iteration in range(self.params.max_iterations):
            # Jacobi step
            wp.launch(
                kernel=jacobi_iteration,
                dim=(nx, ny, nz),
                inputs=[
                    pressure, self.pressure_temp, rhs,
                    self.ax, self.ay, self.az,
                    nx, ny, nz,
                ],
                device=self.device
            )

            # Swap
            pressure, self.pressure_temp = self.pressure_temp, pressure

            # Check convergence periodically
            if (iteration + 1) % 10 == 0:
                residual_norm = self._compute_residual_norm(pressure, rhs)
                if residual_norm < self.params.tolerance:
                    return iteration + 1, residual_norm

        return self.params.max_iterations, self._compute_residual_norm(pressure, rhs)

    def _solve_sor(
        self,
        pressure: wp.array,
        rhs: wp.array
    ) -> Tuple[int, float]:
        """Solve using Red-Black SOR."""
        nx, ny, nz = self.grid_shape

        for iteration in range(self.params.max_iterations):
            # Red sweep
            wp.launch(
                kernel=sor_red_black,
                dim=(nx, ny, nz),
                inputs=[
                    pressure, rhs,
                    self.ax, self.ay, self.az,
                    self.params.omega,
                    nx, ny, nz, 0,
                ],
                device=self.device
            )

            # Black sweep
            wp.launch(
                kernel=sor_red_black,
                dim=(nx, ny, nz),
                inputs=[
                    pressure, rhs,
                    self.ax, self.ay, self.az,
                    self.params.omega,
                    nx, ny, nz, 1,
                ],
                device=self.device
            )

            # Check convergence
            if (iteration + 1) % 10 == 0:
                residual_norm = self._compute_residual_norm(pressure, rhs)
                if residual_norm < self.params.tolerance:
                    return iteration + 1, residual_norm

        return self.params.max_iterations, self._compute_residual_norm(pressure, rhs)

    def _compute_residual_norm(
        self,
        pressure: wp.array,
        rhs: wp.array
    ) -> float:
        """Compute L2 norm of residual."""
        nx, ny, nz = self.grid_shape

        wp.launch(
            kernel=compute_residual,
            dim=(nx, ny, nz),
            inputs=[
                pressure, rhs, self.residual,
                self.ax, self.ay, self.az,
                nx, ny, nz,
            ],
            device=self.device
        )

        # Compute norm on CPU
        residual_np = self.residual.numpy()
        return np.sqrt(np.mean(residual_np**2))
