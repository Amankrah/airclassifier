"""
Navier-Stokes solver for cyclone flow simulation.

Implements a GPU-accelerated incompressible Navier-Stokes solver
using the projection method on a structured grid.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
import warp as wp


@dataclass
class GridParams:
    """Parameters for the computational grid."""

    # Domain size [m]
    domain_size: Tuple[float, float, float] = (0.4, 1.5, 0.4)

    # Grid resolution
    resolution: Tuple[int, int, int] = (64, 128, 64)

    # Grid spacing (computed)
    dx: float = field(init=False)
    dy: float = field(init=False)
    dz: float = field(init=False)

    def __post_init__(self):
        self.dx = self.domain_size[0] / self.resolution[0]
        self.dy = self.domain_size[1] / self.resolution[1]
        self.dz = self.domain_size[2] / self.resolution[2]


@dataclass
class FluidProperties:
    """Physical properties of the fluid."""

    density: float = 1.2          # kg/m³ (air)
    kinematic_viscosity: float = 1.5e-5  # m²/s (air at 20°C)

    @property
    def dynamic_viscosity(self) -> float:
        return self.density * self.kinematic_viscosity


@dataclass
class SolverParams:
    """Solver parameters."""

    dt: float = 1.0e-4           # Time step [s]
    max_iterations: int = 100     # Max pressure iterations
    tolerance: float = 1.0e-6     # Pressure convergence tolerance
    relaxation: float = 1.8       # SOR relaxation factor

    # Numerical schemes
    advection_scheme: str = "semi_lagrangian"  # or "upwind"
    use_turbulence_model: bool = True


# Warp kernels for Navier-Stokes operations

@wp.kernel
def advect_semi_lagrangian(
    vel_x: wp.array3d(dtype=float),
    vel_y: wp.array3d(dtype=float),
    vel_z: wp.array3d(dtype=float),
    vel_x_new: wp.array3d(dtype=float),
    vel_y_new: wp.array3d(dtype=float),
    vel_z_new: wp.array3d(dtype=float),
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Semi-Lagrangian advection for velocity field."""
    i, j, k = wp.tid()

    if i >= nx or j >= ny or k >= nz:
        return

    # Current velocity at this cell
    vx = vel_x[i, j, k]
    vy = vel_y[i, j, k]
    vz = vel_z[i, j, k]

    # Backtrace position
    px = float(i) * dx - vx * dt
    py = float(j) * dy - vy * dt
    pz = float(k) * dz - vz * dt

    # Convert back to grid coordinates
    gx = px / dx
    gy = py / dy
    gz = pz / dz

    # Clamp to grid bounds
    gx = wp.clamp(gx, 0.5, float(nx) - 1.5)
    gy = wp.clamp(gy, 0.5, float(ny) - 1.5)
    gz = wp.clamp(gz, 0.5, float(nz) - 1.5)

    # Trilinear interpolation indices
    i0 = int(wp.floor(gx))
    j0 = int(wp.floor(gy))
    k0 = int(wp.floor(gz))

    i1 = wp.min(i0 + 1, nx - 1)
    j1 = wp.min(j0 + 1, ny - 1)
    k1 = wp.min(k0 + 1, nz - 1)

    # Interpolation weights
    sx = gx - float(i0)
    sy = gy - float(j0)
    sz = gz - float(k0)

    # Trilinear interpolation for each velocity component
    vel_x_new[i, j, k] = (
        vel_x[i0, j0, k0] * (1.0 - sx) * (1.0 - sy) * (1.0 - sz) +
        vel_x[i1, j0, k0] * sx * (1.0 - sy) * (1.0 - sz) +
        vel_x[i0, j1, k0] * (1.0 - sx) * sy * (1.0 - sz) +
        vel_x[i0, j0, k1] * (1.0 - sx) * (1.0 - sy) * sz +
        vel_x[i1, j1, k0] * sx * sy * (1.0 - sz) +
        vel_x[i1, j0, k1] * sx * (1.0 - sy) * sz +
        vel_x[i0, j1, k1] * (1.0 - sx) * sy * sz +
        vel_x[i1, j1, k1] * sx * sy * sz
    )

    vel_y_new[i, j, k] = (
        vel_y[i0, j0, k0] * (1.0 - sx) * (1.0 - sy) * (1.0 - sz) +
        vel_y[i1, j0, k0] * sx * (1.0 - sy) * (1.0 - sz) +
        vel_y[i0, j1, k0] * (1.0 - sx) * sy * (1.0 - sz) +
        vel_y[i0, j0, k1] * (1.0 - sx) * (1.0 - sy) * sz +
        vel_y[i1, j1, k0] * sx * sy * (1.0 - sz) +
        vel_y[i1, j0, k1] * sx * (1.0 - sy) * sz +
        vel_y[i0, j1, k1] * (1.0 - sx) * sy * sz +
        vel_y[i1, j1, k1] * sx * sy * sz
    )

    vel_z_new[i, j, k] = (
        vel_z[i0, j0, k0] * (1.0 - sx) * (1.0 - sy) * (1.0 - sz) +
        vel_z[i1, j0, k0] * sx * (1.0 - sy) * (1.0 - sz) +
        vel_z[i0, j1, k0] * (1.0 - sx) * sy * (1.0 - sz) +
        vel_z[i0, j0, k1] * (1.0 - sx) * (1.0 - sy) * sz +
        vel_z[i1, j1, k0] * sx * sy * (1.0 - sz) +
        vel_z[i1, j0, k1] * sx * (1.0 - sy) * sz +
        vel_z[i0, j1, k1] * (1.0 - sx) * sy * sz +
        vel_z[i1, j1, k1] * sx * sy * sz
    )


@wp.kernel
def apply_diffusion(
    vel_x: wp.array3d(dtype=float),
    vel_y: wp.array3d(dtype=float),
    vel_z: wp.array3d(dtype=float),
    vel_x_new: wp.array3d(dtype=float),
    vel_y_new: wp.array3d(dtype=float),
    vel_z_new: wp.array3d(dtype=float),
    nu: float,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Explicit diffusion step."""
    i, j, k = wp.tid()

    if i <= 0 or i >= nx - 1 or j <= 0 or j >= ny - 1 or k <= 0 or k >= nz - 1:
        # Boundary cells keep their values
        if i < nx and j < ny and k < nz:
            vel_x_new[i, j, k] = vel_x[i, j, k]
            vel_y_new[i, j, k] = vel_y[i, j, k]
            vel_z_new[i, j, k] = vel_z[i, j, k]
        return

    # Laplacian coefficients
    cx = nu * dt / (dx * dx)
    cy = nu * dt / (dy * dy)
    cz = nu * dt / (dz * dz)

    # Compute Laplacian for each component
    laplacian_x = (
        cx * (vel_x[i+1, j, k] - 2.0 * vel_x[i, j, k] + vel_x[i-1, j, k]) +
        cy * (vel_x[i, j+1, k] - 2.0 * vel_x[i, j, k] + vel_x[i, j-1, k]) +
        cz * (vel_x[i, j, k+1] - 2.0 * vel_x[i, j, k] + vel_x[i, j, k-1])
    )

    laplacian_y = (
        cx * (vel_y[i+1, j, k] - 2.0 * vel_y[i, j, k] + vel_y[i-1, j, k]) +
        cy * (vel_y[i, j+1, k] - 2.0 * vel_y[i, j, k] + vel_y[i, j-1, k]) +
        cz * (vel_y[i, j, k+1] - 2.0 * vel_y[i, j, k] + vel_y[i, j, k-1])
    )

    laplacian_z = (
        cx * (vel_z[i+1, j, k] - 2.0 * vel_z[i, j, k] + vel_z[i-1, j, k]) +
        cy * (vel_z[i, j+1, k] - 2.0 * vel_z[i, j, k] + vel_z[i, j-1, k]) +
        cz * (vel_z[i, j, k+1] - 2.0 * vel_z[i, j, k] + vel_z[i, j, k-1])
    )

    vel_x_new[i, j, k] = vel_x[i, j, k] + laplacian_x
    vel_y_new[i, j, k] = vel_y[i, j, k] + laplacian_y
    vel_z_new[i, j, k] = vel_z[i, j, k] + laplacian_z


@wp.kernel
def apply_diffusion_variable_viscosity(
    vel_x: wp.array3d(dtype=float),
    vel_y: wp.array3d(dtype=float),
    vel_z: wp.array3d(dtype=float),
    vel_x_new: wp.array3d(dtype=float),
    vel_y_new: wp.array3d(dtype=float),
    vel_z_new: wp.array3d(dtype=float),
    nu_molecular: float,
    nu_t: wp.array3d(dtype=float),
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Diffusion step with variable turbulent viscosity."""
    i, j, k = wp.tid()

    if i <= 0 or i >= nx - 1 or j <= 0 or j >= ny - 1 or k <= 0 or k >= nz - 1:
        if i < nx and j < ny and k < nz:
            vel_x_new[i, j, k] = vel_x[i, j, k]
            vel_y_new[i, j, k] = vel_y[i, j, k]
            vel_z_new[i, j, k] = vel_z[i, j, k]
        return

    # Effective viscosity: molecular + turbulent
    nu_eff = nu_molecular + nu_t[i, j, k]

    # Laplacian coefficients with effective viscosity
    cx = nu_eff * dt / (dx * dx)
    cy = nu_eff * dt / (dy * dy)
    cz = nu_eff * dt / (dz * dz)

    # Compute Laplacian for each component
    laplacian_x = (
        cx * (vel_x[i+1, j, k] - 2.0 * vel_x[i, j, k] + vel_x[i-1, j, k]) +
        cy * (vel_x[i, j+1, k] - 2.0 * vel_x[i, j, k] + vel_x[i, j-1, k]) +
        cz * (vel_x[i, j, k+1] - 2.0 * vel_x[i, j, k] + vel_x[i, j, k-1])
    )

    laplacian_y = (
        cx * (vel_y[i+1, j, k] - 2.0 * vel_y[i, j, k] + vel_y[i-1, j, k]) +
        cy * (vel_y[i, j+1, k] - 2.0 * vel_y[i, j, k] + vel_y[i, j-1, k]) +
        cz * (vel_y[i, j, k+1] - 2.0 * vel_y[i, j, k] + vel_y[i, j, k-1])
    )

    laplacian_z = (
        cx * (vel_z[i+1, j, k] - 2.0 * vel_z[i, j, k] + vel_z[i-1, j, k]) +
        cy * (vel_z[i, j+1, k] - 2.0 * vel_z[i, j, k] + vel_z[i, j-1, k]) +
        cz * (vel_z[i, j, k+1] - 2.0 * vel_z[i, j, k] + vel_z[i, j, k-1])
    )

    vel_x_new[i, j, k] = vel_x[i, j, k] + laplacian_x
    vel_y_new[i, j, k] = vel_y[i, j, k] + laplacian_y
    vel_z_new[i, j, k] = vel_z[i, j, k] + laplacian_z


@wp.kernel
def compute_divergence(
    vel_x: wp.array3d(dtype=float),
    vel_y: wp.array3d(dtype=float),
    vel_z: wp.array3d(dtype=float),
    divergence: wp.array3d(dtype=float),
    dx: float,
    dy: float,
    dz: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Compute velocity divergence."""
    i, j, k = wp.tid()

    if i >= nx or j >= ny or k >= nz:
        return

    # Handle boundaries
    im = wp.max(i - 1, 0)
    ip = wp.min(i + 1, nx - 1)
    jm = wp.max(j - 1, 0)
    jp = wp.min(j + 1, ny - 1)
    km = wp.max(k - 1, 0)
    kp = wp.min(k + 1, nz - 1)

    # Central differences
    dudx = (vel_x[ip, j, k] - vel_x[im, j, k]) / (2.0 * dx)
    dvdy = (vel_y[i, jp, k] - vel_y[i, jm, k]) / (2.0 * dy)
    dwdz = (vel_z[i, j, kp] - vel_z[i, j, km]) / (2.0 * dz)

    divergence[i, j, k] = dudx + dvdy + dwdz


@wp.kernel
def pressure_jacobi_iteration(
    pressure: wp.array3d(dtype=float),
    pressure_new: wp.array3d(dtype=float),
    divergence: wp.array3d(dtype=float),
    dx: float,
    dy: float,
    dz: float,
    density: float,
    dt: float,
    nx: int,
    ny: int,
    nz: int,
):
    """One Jacobi iteration for pressure solve."""
    i, j, k = wp.tid()

    if i >= nx or j >= ny or k >= nz:
        return

    # Neumann boundary condition (zero gradient)
    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
        pressure_new[i, j, k] = 0.0
        return

    # Coefficients
    ax = 1.0 / (dx * dx)
    ay = 1.0 / (dy * dy)
    az = 1.0 / (dz * dz)
    diag = 2.0 * (ax + ay + az)

    # RHS
    rhs = -density * divergence[i, j, k] / dt

    # Neighbor contributions
    neighbor_sum = (
        ax * (pressure[i+1, j, k] + pressure[i-1, j, k]) +
        ay * (pressure[i, j+1, k] + pressure[i, j-1, k]) +
        az * (pressure[i, j, k+1] + pressure[i, j, k-1])
    )

    pressure_new[i, j, k] = (rhs + neighbor_sum) / diag


@wp.kernel
def project_velocity(
    vel_x: wp.array3d(dtype=float),
    vel_y: wp.array3d(dtype=float),
    vel_z: wp.array3d(dtype=float),
    pressure: wp.array3d(dtype=float),
    density: float,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Project velocity to be divergence-free."""
    i, j, k = wp.tid()

    if i >= nx or j >= ny or k >= nz:
        return

    # Compute pressure gradient
    im = wp.max(i - 1, 0)
    ip = wp.min(i + 1, nx - 1)
    jm = wp.max(j - 1, 0)
    jp = wp.min(j + 1, ny - 1)
    km = wp.max(k - 1, 0)
    kp = wp.min(k + 1, nz - 1)

    dpdx = (pressure[ip, j, k] - pressure[im, j, k]) / (2.0 * dx)
    dpdy = (pressure[i, jp, k] - pressure[i, jm, k]) / (2.0 * dy)
    dpdz = (pressure[i, j, kp] - pressure[i, j, km]) / (2.0 * dz)

    # Update velocity
    vel_x[i, j, k] = vel_x[i, j, k] - dt * dpdx / density
    vel_y[i, j, k] = vel_y[i, j, k] - dt * dpdy / density
    vel_z[i, j, k] = vel_z[i, j, k] - dt * dpdz / density


class NavierStokesSolver:
    """
    GPU-accelerated incompressible Navier-Stokes solver.

    Uses the projection method:
    1. Advect velocity (semi-Lagrangian or upwind)
    2. Apply diffusion
    3. Solve pressure Poisson equation
    4. Project to divergence-free field
    """

    def __init__(
        self,
        grid_params: GridParams,
        fluid_props: FluidProperties,
        solver_params: SolverParams,
        device: str = "cuda"
    ):
        """
        Initialize the Navier-Stokes solver.

        Args:
            grid_params: Grid configuration
            fluid_props: Fluid physical properties
            solver_params: Solver parameters
            device: Warp device
        """
        self.grid = grid_params
        self.fluid = fluid_props
        self.params = solver_params
        self.device = device

        nx, ny, nz = self.grid.resolution

        # Allocate velocity arrays (staggered MAC grid would be better but
        # using collocated for simplicity)
        self.vel_x = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self.vel_y = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self.vel_z = wp.zeros((nx, ny, nz), dtype=float, device=device)

        # Temporary velocity arrays
        self.vel_x_temp = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self.vel_y_temp = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self.vel_z_temp = wp.zeros((nx, ny, nz), dtype=float, device=device)

        # Pressure and divergence
        self.pressure = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self.pressure_temp = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self.divergence = wp.zeros((nx, ny, nz), dtype=float, device=device)

        # Turbulent viscosity (if using turbulence model)
        self.nu_t = wp.zeros((nx, ny, nz), dtype=float, device=device)
        self._use_turbulent_viscosity = False

        self._time = 0.0

    def initialize_cyclone_flow(
        self,
        inlet_velocity: float,
        cylinder_radius: float,
        vortex_finder_radius: float,
    ):
        """
        Initialize velocity field for cyclone flow.

        Sets up an initial swirling flow pattern.

        Args:
            inlet_velocity: Inlet velocity magnitude [m/s]
            cylinder_radius: Cyclone body radius [m]
            vortex_finder_radius: Vortex finder radius [m]
        """
        nx, ny, nz = self.grid.resolution
        dx, dy, dz = self.grid.dx, self.grid.dy, self.grid.dz

        # Initialize on CPU then copy
        vel_x_np = np.zeros((nx, ny, nz))
        vel_y_np = np.zeros((nx, ny, nz))
        vel_z_np = np.zeros((nx, ny, nz))

        # Center of domain
        cx = self.grid.domain_size[0] / 2
        cz = self.grid.domain_size[2] / 2

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x = i * dx - cx
                    z = k * dz - cz
                    r = np.sqrt(x*x + z*z) + 1e-10

                    if r < cylinder_radius:
                        # Tangential velocity (Rankine vortex approximation)
                        if r < vortex_finder_radius:
                            # Forced vortex region
                            v_tan = inlet_velocity * r / vortex_finder_radius
                        else:
                            # Free vortex region
                            v_tan = inlet_velocity * vortex_finder_radius / r

                        # Convert to Cartesian
                        vel_x_np[i, j, k] = -v_tan * z / r
                        vel_z_np[i, j, k] = v_tan * x / r

                        # Small downward flow
                        vel_y_np[i, j, k] = -0.1 * inlet_velocity

        # Copy to GPU
        wp.copy(self.vel_x, wp.array(vel_x_np, dtype=float, device=self.device))
        wp.copy(self.vel_y, wp.array(vel_y_np, dtype=float, device=self.device))
        wp.copy(self.vel_z, wp.array(vel_z_np, dtype=float, device=self.device))

    def set_eddy_viscosity(self, nu_t: wp.array):
        """
        Set the turbulent eddy viscosity field.

        Args:
            nu_t: 3D array of eddy viscosity values [m²/s]
        """
        wp.copy(self.nu_t, nu_t)
        self._use_turbulent_viscosity = True

    def step(self):
        """Advance the simulation by one time step."""
        nx, ny, nz = self.grid.resolution
        dt = self.params.dt

        # 1. Advection
        wp.launch(
            kernel=advect_semi_lagrangian,
            dim=(nx, ny, nz),
            inputs=[
                self.vel_x, self.vel_y, self.vel_z,
                self.vel_x_temp, self.vel_y_temp, self.vel_z_temp,
                dt,
                self.grid.dx, self.grid.dy, self.grid.dz,
                nx, ny, nz,
            ],
            device=self.device
        )

        # Swap buffers
        self.vel_x, self.vel_x_temp = self.vel_x_temp, self.vel_x
        self.vel_y, self.vel_y_temp = self.vel_y_temp, self.vel_y
        self.vel_z, self.vel_z_temp = self.vel_z_temp, self.vel_z

        # 2. Diffusion (with or without turbulent viscosity)
        if self._use_turbulent_viscosity:
            wp.launch(
                kernel=apply_diffusion_variable_viscosity,
                dim=(nx, ny, nz),
                inputs=[
                    self.vel_x, self.vel_y, self.vel_z,
                    self.vel_x_temp, self.vel_y_temp, self.vel_z_temp,
                    self.fluid.kinematic_viscosity, self.nu_t,
                    dt,
                    self.grid.dx, self.grid.dy, self.grid.dz,
                    nx, ny, nz,
                ],
                device=self.device
            )
        else:
            wp.launch(
                kernel=apply_diffusion,
                dim=(nx, ny, nz),
                inputs=[
                    self.vel_x, self.vel_y, self.vel_z,
                    self.vel_x_temp, self.vel_y_temp, self.vel_z_temp,
                    self.fluid.kinematic_viscosity, dt,
                    self.grid.dx, self.grid.dy, self.grid.dz,
                    nx, ny, nz,
                ],
                device=self.device
            )

        self.vel_x, self.vel_x_temp = self.vel_x_temp, self.vel_x
        self.vel_y, self.vel_y_temp = self.vel_y_temp, self.vel_y
        self.vel_z, self.vel_z_temp = self.vel_z_temp, self.vel_z

        # 3. Compute divergence
        wp.launch(
            kernel=compute_divergence,
            dim=(nx, ny, nz),
            inputs=[
                self.vel_x, self.vel_y, self.vel_z,
                self.divergence,
                self.grid.dx, self.grid.dy, self.grid.dz,
                nx, ny, nz,
            ],
            device=self.device
        )

        # 4. Solve pressure Poisson equation (Jacobi iterations)
        for _ in range(self.params.max_iterations):
            wp.launch(
                kernel=pressure_jacobi_iteration,
                dim=(nx, ny, nz),
                inputs=[
                    self.pressure, self.pressure_temp,
                    self.divergence,
                    self.grid.dx, self.grid.dy, self.grid.dz,
                    self.fluid.density, dt,
                    nx, ny, nz,
                ],
                device=self.device
            )
            self.pressure, self.pressure_temp = self.pressure_temp, self.pressure

        # 5. Project velocity to divergence-free
        wp.launch(
            kernel=project_velocity,
            dim=(nx, ny, nz),
            inputs=[
                self.vel_x, self.vel_y, self.vel_z,
                self.pressure,
                self.fluid.density, dt,
                self.grid.dx, self.grid.dy, self.grid.dz,
                nx, ny, nz,
            ],
            device=self.device
        )

        self._time += dt

    def get_velocity_at(self, position: np.ndarray) -> np.ndarray:
        """
        Get interpolated velocity at a point.

        Args:
            position: 3D position [m]

        Returns:
            Velocity vector [m/s]
        """
        # Convert position to grid coordinates
        gx = position[0] / self.grid.dx
        gy = position[1] / self.grid.dy
        gz = position[2] / self.grid.dz

        nx, ny, nz = self.grid.resolution

        # Clamp to grid
        gx = np.clip(gx, 0, nx - 1)
        gy = np.clip(gy, 0, ny - 1)
        gz = np.clip(gz, 0, nz - 1)

        # Get integer indices
        i0 = int(np.floor(gx))
        j0 = int(np.floor(gy))
        k0 = int(np.floor(gz))

        i1 = min(i0 + 1, nx - 1)
        j1 = min(j0 + 1, ny - 1)
        k1 = min(k0 + 1, nz - 1)

        # Interpolation weights
        sx = gx - i0
        sy = gy - j0
        sz = gz - k0

        # Copy small region to CPU for interpolation
        vel_x_np = self.vel_x.numpy()
        vel_y_np = self.vel_y.numpy()
        vel_z_np = self.vel_z.numpy()

        # Trilinear interpolation
        def interp(arr):
            return (
                arr[i0, j0, k0] * (1-sx) * (1-sy) * (1-sz) +
                arr[i1, j0, k0] * sx * (1-sy) * (1-sz) +
                arr[i0, j1, k0] * (1-sx) * sy * (1-sz) +
                arr[i0, j0, k1] * (1-sx) * (1-sy) * sz +
                arr[i1, j1, k0] * sx * sy * (1-sz) +
                arr[i1, j0, k1] * sx * (1-sy) * sz +
                arr[i0, j1, k1] * (1-sx) * sy * sz +
                arr[i1, j1, k1] * sx * sy * sz
            )

        return np.array([
            interp(vel_x_np),
            interp(vel_y_np),
            interp(vel_z_np),
        ])

    @property
    def time(self) -> float:
        """Current simulation time."""
        return self._time
