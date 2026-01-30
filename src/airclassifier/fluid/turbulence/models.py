"""
Turbulence models for cyclone flow simulation.

Provides RANS-based turbulence models for computing eddy viscosity.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import warp as wp


@dataclass
class KEpsilonParams:
    """Parameters for k-epsilon turbulence model."""

    # Model constants
    C_mu: float = 0.09
    C_1: float = 1.44
    C_2: float = 1.92
    sigma_k: float = 1.0
    sigma_epsilon: float = 1.3

    # Wall function parameters
    kappa: float = 0.41        # von Karman constant
    E: float = 9.793           # Wall roughness parameter


@dataclass
class SmagorinskyParams:
    """Parameters for Smagorinsky LES model."""

    C_s: float = 0.1           # Smagorinsky constant
    filter_width_factor: float = 1.0  # Multiplier for grid spacing


@wp.kernel
def compute_strain_rate_magnitude(
    vel_x: wp.array3d(dtype=float),
    vel_y: wp.array3d(dtype=float),
    vel_z: wp.array3d(dtype=float),
    strain_mag: wp.array3d(dtype=float),
    dx: float,
    dy: float,
    dz: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Compute magnitude of strain rate tensor."""
    i, j, k = wp.tid()

    if i <= 0 or i >= nx - 1 or j <= 0 or j >= ny - 1 or k <= 0 or k >= nz - 1:
        if i < nx and j < ny and k < nz:
            strain_mag[i, j, k] = 0.0
        return

    # Velocity gradients (central differences)
    dudx = (vel_x[i+1, j, k] - vel_x[i-1, j, k]) / (2.0 * dx)
    dudy = (vel_x[i, j+1, k] - vel_x[i, j-1, k]) / (2.0 * dy)
    dudz = (vel_x[i, j, k+1] - vel_x[i, j, k-1]) / (2.0 * dz)

    dvdx = (vel_y[i+1, j, k] - vel_y[i-1, j, k]) / (2.0 * dx)
    dvdy = (vel_y[i, j+1, k] - vel_y[i, j-1, k]) / (2.0 * dy)
    dvdz = (vel_y[i, j, k+1] - vel_y[i, j, k-1]) / (2.0 * dz)

    dwdx = (vel_z[i+1, j, k] - vel_z[i-1, j, k]) / (2.0 * dx)
    dwdy = (vel_z[i, j+1, k] - vel_z[i, j-1, k]) / (2.0 * dy)
    dwdz = (vel_z[i, j, k+1] - vel_z[i, j, k-1]) / (2.0 * dz)

    # Strain rate tensor components S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    S11 = dudx
    S22 = dvdy
    S33 = dwdz
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (dudz + dwdx)
    S23 = 0.5 * (dvdz + dwdy)

    # |S| = sqrt(2 * S_ij * S_ij)
    S_sq = S11*S11 + S22*S22 + S33*S33 + 2.0*(S12*S12 + S13*S13 + S23*S23)
    strain_mag[i, j, k] = wp.sqrt(2.0 * S_sq)


@wp.kernel
def compute_smagorinsky_viscosity(
    strain_mag: wp.array3d(dtype=float),
    nu_t: wp.array3d(dtype=float),
    C_s: float,
    delta: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Compute Smagorinsky eddy viscosity."""
    i, j, k = wp.tid()

    if i >= nx or j >= ny or k >= nz:
        return

    # nu_t = (C_s * delta)^2 * |S|
    l_smag = C_s * delta
    nu_t[i, j, k] = l_smag * l_smag * strain_mag[i, j, k]


@wp.kernel
def compute_k_epsilon_viscosity(
    k: wp.array3d(dtype=float),
    epsilon: wp.array3d(dtype=float),
    nu_t: wp.array3d(dtype=float),
    C_mu: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Compute k-epsilon eddy viscosity."""
    i, j, k_idx = wp.tid()

    if i >= nx or j >= ny or k_idx >= nz:
        return

    k_val = k[i, j, k_idx]
    eps_val = wp.max(epsilon[i, j, k_idx], 1.0e-10)

    # nu_t = C_mu * k^2 / epsilon
    nu_t[i, j, k_idx] = C_mu * k_val * k_val / eps_val


@wp.kernel
def transport_k_equation(
    k: wp.array3d(dtype=float),
    k_new: wp.array3d(dtype=float),
    epsilon: wp.array3d(dtype=float),
    nu_t: wp.array3d(dtype=float),
    strain_mag: wp.array3d(dtype=float),
    nu: float,
    sigma_k: float,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Transport equation for turbulent kinetic energy k."""
    i, j, k_idx = wp.tid()

    if i <= 0 or i >= nx - 1 or j <= 0 or j >= ny - 1 or k_idx <= 0 or k_idx >= nz - 1:
        if i < nx and j < ny and k_idx < nz:
            k_new[i, j, k_idx] = k[i, j, k_idx]
        return

    k_val = k[i, j, k_idx]
    eps_val = epsilon[i, j, k_idx]
    nu_t_val = nu_t[i, j, k_idx]
    S = strain_mag[i, j, k_idx]

    # Effective diffusivity
    nu_eff = nu + nu_t_val / sigma_k

    # Diffusion (Laplacian)
    d2kdx2 = (k[i+1, j, k_idx] - 2.0*k_val + k[i-1, j, k_idx]) / (dx*dx)
    d2kdy2 = (k[i, j+1, k_idx] - 2.0*k_val + k[i, j-1, k_idx]) / (dy*dy)
    d2kdz2 = (k[i, j, k_idx+1] - 2.0*k_val + k[i, j, k_idx-1]) / (dz*dz)

    diffusion = nu_eff * (d2kdx2 + d2kdy2 + d2kdz2)

    # Production P_k = nu_t * |S|^2
    production = nu_t_val * S * S

    # Dissipation
    dissipation = eps_val

    # Update k
    dk = (production - dissipation + diffusion) * dt
    k_new[i, j, k_idx] = wp.max(k_val + dk, 1.0e-10)


@wp.kernel
def transport_epsilon_equation(
    k: wp.array3d(dtype=float),
    epsilon: wp.array3d(dtype=float),
    epsilon_new: wp.array3d(dtype=float),
    nu_t: wp.array3d(dtype=float),
    strain_mag: wp.array3d(dtype=float),
    nu: float,
    C_1: float,
    C_2: float,
    sigma_epsilon: float,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Transport equation for turbulent dissipation rate epsilon."""
    i, j, k_idx = wp.tid()

    if i <= 0 or i >= nx - 1 or j <= 0 or j >= ny - 1 or k_idx <= 0 or k_idx >= nz - 1:
        if i < nx and j < ny and k_idx < nz:
            epsilon_new[i, j, k_idx] = epsilon[i, j, k_idx]
        return

    k_val = wp.max(k[i, j, k_idx], 1.0e-10)
    eps_val = epsilon[i, j, k_idx]
    nu_t_val = nu_t[i, j, k_idx]
    S = strain_mag[i, j, k_idx]

    # Effective diffusivity
    nu_eff = nu + nu_t_val / sigma_epsilon

    # Diffusion
    d2edx2 = (epsilon[i+1, j, k_idx] - 2.0*eps_val + epsilon[i-1, j, k_idx]) / (dx*dx)
    d2edy2 = (epsilon[i, j+1, k_idx] - 2.0*eps_val + epsilon[i, j-1, k_idx]) / (dy*dy)
    d2edz2 = (epsilon[i, j, k_idx+1] - 2.0*eps_val + epsilon[i, j, k_idx-1]) / (dz*dz)

    diffusion = nu_eff * (d2edx2 + d2edy2 + d2edz2)

    # Production
    production = C_1 * eps_val / k_val * nu_t_val * S * S

    # Destruction
    destruction = C_2 * eps_val * eps_val / k_val

    # Update epsilon
    deps = (production - destruction + diffusion) * dt
    epsilon_new[i, j, k_idx] = wp.max(eps_val + deps, 1.0e-10)


class SmagorinskyModel:
    """
    Smagorinsky LES turbulence model.

    Computes subgrid-scale eddy viscosity as:
    nu_t = (C_s * delta)^2 * |S|

    where |S| is the strain rate magnitude.
    """

    def __init__(
        self,
        params: SmagorinskyParams,
        grid_shape: tuple,
        grid_spacing: tuple,
        device: str = "cuda"
    ):
        """
        Initialize Smagorinsky model.

        Args:
            params: Model parameters
            grid_shape: (nx, ny, nz) grid dimensions
            grid_spacing: (dx, dy, dz) grid spacing
            device: Warp device
        """
        self.params = params
        self.grid_shape = grid_shape
        self.grid_spacing = grid_spacing
        self.device = device

        nx, ny, nz = grid_shape
        self.strain_mag = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self.nu_t = wp.zeros((nx, ny, nz), dtype=float, device=device)

        # Filter width (average grid spacing)
        self.delta = params.filter_width_factor * (
            grid_spacing[0] * grid_spacing[1] * grid_spacing[2]
        ) ** (1/3)

    def compute_eddy_viscosity(
        self,
        vel_x: wp.array,
        vel_y: wp.array,
        vel_z: wp.array,
    ) -> wp.array:
        """
        Compute eddy viscosity from velocity field.

        Args:
            vel_x, vel_y, vel_z: Velocity components

        Returns:
            Eddy viscosity field
        """
        nx, ny, nz = self.grid_shape
        dx, dy, dz = self.grid_spacing

        # Compute strain rate magnitude
        wp.launch(
            kernel=compute_strain_rate_magnitude,
            dim=(nx, ny, nz),
            inputs=[
                vel_x, vel_y, vel_z,
                self.strain_mag,
                dx, dy, dz,
                nx, ny, nz,
            ],
            device=self.device
        )

        # Compute eddy viscosity
        wp.launch(
            kernel=compute_smagorinsky_viscosity,
            dim=(nx, ny, nz),
            inputs=[
                self.strain_mag,
                self.nu_t,
                self.params.C_s,
                self.delta,
                nx, ny, nz,
            ],
            device=self.device
        )

        return self.nu_t


class KEpsilonModel:
    """
    Standard k-epsilon RANS turbulence model.

    Solves transport equations for turbulent kinetic energy (k)
    and dissipation rate (epsilon) to compute eddy viscosity.
    """

    def __init__(
        self,
        params: KEpsilonParams,
        grid_shape: tuple,
        grid_spacing: tuple,
        molecular_viscosity: float,
        device: str = "cuda"
    ):
        """
        Initialize k-epsilon model.

        Args:
            params: Model parameters
            grid_shape: Grid dimensions
            grid_spacing: Grid spacing
            molecular_viscosity: Kinematic viscosity [m²/s]
            device: Warp device
        """
        self.params = params
        self.grid_shape = grid_shape
        self.grid_spacing = grid_spacing
        self.nu = molecular_viscosity
        self.device = device

        nx, ny, nz = grid_shape

        # Turbulence fields
        self.k = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self.epsilon = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self.k_new = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self.epsilon_new = wp.zeros((nx, ny, nz), dtype=float, device=device)

        self.nu_t = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self.strain_mag = wp.zeros((nx, ny, nz), dtype=float, device=device)

    def initialize(self, k_init: float = 0.1, epsilon_init: float = 0.01):
        """
        Initialize turbulence fields.

        Args:
            k_init: Initial turbulent kinetic energy [m²/s²]
            epsilon_init: Initial dissipation rate [m²/s³]
        """
        nx, ny, nz = self.grid_shape
        k_np = np.full((nx, ny, nz), k_init)
        eps_np = np.full((nx, ny, nz), epsilon_init)

        wp.copy(self.k, wp.array(k_np, dtype=float, device=self.device))
        wp.copy(self.epsilon, wp.array(eps_np, dtype=float, device=self.device))

    def step(
        self,
        vel_x: wp.array,
        vel_y: wp.array,
        vel_z: wp.array,
        dt: float
    ):
        """
        Advance turbulence model by one time step.

        Args:
            vel_x, vel_y, vel_z: Velocity components
            dt: Time step
        """
        nx, ny, nz = self.grid_shape
        dx, dy, dz = self.grid_spacing

        # Compute strain rate magnitude
        wp.launch(
            kernel=compute_strain_rate_magnitude,
            dim=(nx, ny, nz),
            inputs=[
                vel_x, vel_y, vel_z,
                self.strain_mag,
                dx, dy, dz,
                nx, ny, nz,
            ],
            device=self.device
        )

        # Compute eddy viscosity
        wp.launch(
            kernel=compute_k_epsilon_viscosity,
            dim=(nx, ny, nz),
            inputs=[
                self.k, self.epsilon, self.nu_t,
                self.params.C_mu,
                nx, ny, nz,
            ],
            device=self.device
        )

        # Transport k
        wp.launch(
            kernel=transport_k_equation,
            dim=(nx, ny, nz),
            inputs=[
                self.k, self.k_new, self.epsilon, self.nu_t, self.strain_mag,
                self.nu, self.params.sigma_k, dt,
                dx, dy, dz,
                nx, ny, nz,
            ],
            device=self.device
        )

        # Transport epsilon
        wp.launch(
            kernel=transport_epsilon_equation,
            dim=(nx, ny, nz),
            inputs=[
                self.k, self.epsilon, self.epsilon_new, self.nu_t, self.strain_mag,
                self.nu, self.params.C_1, self.params.C_2, self.params.sigma_epsilon,
                dt, dx, dy, dz,
                nx, ny, nz,
            ],
            device=self.device
        )

        # Swap buffers
        self.k, self.k_new = self.k_new, self.k
        self.epsilon, self.epsilon_new = self.epsilon_new, self.epsilon

    def get_eddy_viscosity(self) -> wp.array:
        """Return current eddy viscosity field."""
        return self.nu_t
