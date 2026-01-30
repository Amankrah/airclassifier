"""
Wall functions for turbulence models.

Provides wall treatment for RANS turbulence models,
including standard wall functions and enhanced wall treatment.
"""

from dataclasses import dataclass
import numpy as np
import warp as wp


@dataclass
class WallFunctionParams:
    """Parameters for wall functions."""

    kappa: float = 0.41          # von Karman constant
    E: float = 9.793             # Wall roughness parameter (smooth wall)
    y_plus_transition: float = 11.63  # Transition between viscous and log layer


@wp.func
def compute_y_plus(
    y: float,
    u_tau: float,
    nu: float
) -> float:
    """
    Compute dimensionless wall distance y+.

    Args:
        y: Distance from wall [m]
        u_tau: Friction velocity [m/s]
        nu: Kinematic viscosity [m²/s]

    Returns:
        y+ value
    """
    return y * u_tau / nu


@wp.func
def compute_u_plus_viscous(y_plus: float) -> float:
    """
    Compute u+ in viscous sublayer (y+ < 5).

    In the viscous sublayer: u+ = y+
    """
    return y_plus


@wp.func
def compute_u_plus_log(
    y_plus: float,
    kappa: float,
    E: float
) -> float:
    """
    Compute u+ in log-law region (y+ > 30).

    Log law: u+ = (1/kappa) * ln(E * y+)
    """
    return (1.0 / kappa) * wp.log(E * y_plus)


@wp.func
def friction_velocity_from_velocity(
    U: float,
    y: float,
    nu: float,
    kappa: float,
    E: float
) -> float:
    """
    Estimate friction velocity from near-wall velocity.

    Uses Newton iteration to solve implicitly.
    """
    # Initial guess
    u_tau = wp.sqrt(nu * U / y)

    # Newton iterations
    for _ in range(10):
        y_plus = y * u_tau / nu

        if y_plus < 11.63:
            u_plus = y_plus
            du_plus = y / nu
        else:
            u_plus = (1.0 / kappa) * wp.log(E * y_plus)
            du_plus = 1.0 / (kappa * u_tau)

        f = u_tau - U / u_plus
        df = 1.0 + U * du_plus / (u_plus * u_plus)

        delta = f / df
        u_tau = u_tau - delta

        if wp.abs(delta) < 1.0e-8:
            break

    return wp.max(u_tau, 1.0e-10)


@wp.kernel
def apply_wall_functions_k_epsilon(
    k: wp.array3d(dtype=float),
    epsilon: wp.array3d(dtype=float),
    wall_distance: wp.array3d(dtype=float),
    is_wall_cell: wp.array3d(dtype=wp.int32),
    velocity_mag: wp.array3d(dtype=float),
    nu: float,
    kappa: float,
    C_mu: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Apply wall functions for k-epsilon model."""
    i, j, k_idx = wp.tid()

    if i >= nx or j >= ny or k_idx >= nz:
        return

    if is_wall_cell[i, j, k_idx] != 1:
        return

    y = wall_distance[i, j, k_idx]
    U = velocity_mag[i, j, k_idx]

    if y < 1.0e-10 or U < 1.0e-10:
        return

    u_tau = friction_velocity_from_velocity(U, y, nu, kappa, 9.793)
    k[i, j, k_idx] = u_tau * u_tau / wp.sqrt(C_mu)
    epsilon[i, j, k_idx] = u_tau * u_tau * u_tau / (kappa * y)


class WallFunctionManager:
    """Manages wall function calculations for turbulence models."""

    def __init__(
        self,
        params: WallFunctionParams,
        grid_shape: tuple,
        device: str = "cuda"
    ):
        self.params = params
        self.grid_shape = grid_shape
        self.device = device

        nx, ny, nz = grid_shape
        self.wall_distance = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self.is_wall_cell = wp.zeros((nx, ny, nz), dtype=wp.int32, device=device)
        self.velocity_mag = wp.zeros((nx, ny, nz), dtype=float, device=device)

    def set_wall_cells(
        self,
        wall_distance_np: np.ndarray,
        is_wall_cell_np: np.ndarray
    ):
        """Set wall distance and wall cell flags."""
        wp.copy(
            self.wall_distance,
            wp.array(wall_distance_np, dtype=float, device=self.device)
        )
        wp.copy(
            self.is_wall_cell,
            wp.array(is_wall_cell_np, dtype=wp.int32, device=self.device)
        )

    def apply_k_epsilon_wall_functions(
        self,
        k: wp.array,
        epsilon: wp.array,
        vel_x: wp.array,
        vel_y: wp.array,
        vel_z: wp.array,
        nu: float,
        C_mu: float = 0.09
    ):
        """Apply wall functions to k-epsilon turbulence fields."""
        nx, ny, nz = self.grid_shape

        vel_x_np = vel_x.numpy()
        vel_y_np = vel_y.numpy()
        vel_z_np = vel_z.numpy()
        vel_mag_np = np.sqrt(vel_x_np**2 + vel_y_np**2 + vel_z_np**2)
        wp.copy(
            self.velocity_mag,
            wp.array(vel_mag_np, dtype=float, device=self.device)
        )

        wp.launch(
            kernel=apply_wall_functions_k_epsilon,
            dim=(nx, ny, nz),
            inputs=[
                k, epsilon, self.wall_distance, self.is_wall_cell,
                self.velocity_mag, nu, self.params.kappa, C_mu,
                nx, ny, nz,
            ],
            device=self.device
        )
