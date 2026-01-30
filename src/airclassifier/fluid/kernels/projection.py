"""
Projection kernels for CFD solver.

Provides pressure projection to enforce incompressibility
(divergence-free velocity field).
"""

import warp as wp


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
    """
    Compute velocity divergence using central differences.

    div(V) = du/dx + dv/dy + dw/dz

    Args:
        vel_x, vel_y, vel_z: Velocity components
        divergence: Output divergence field
        dx, dy, dz: Grid spacing
        nx, ny, nz: Grid dimensions
    """
    i, j, k = wp.tid()

    if i >= nx or j >= ny or k >= nz:
        return

    # Handle boundaries with one-sided differences
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
def pressure_poisson_jacobi(
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
    """
    Jacobi iteration for pressure Poisson equation.

    Solves: laplacian(p) = (rho/dt) * div(V)

    Args:
        pressure: Current pressure iteration
        pressure_new: Next pressure iteration
        divergence: Velocity divergence
        dx, dy, dz: Grid spacing
        density: Fluid density
        dt: Time step
        nx, ny, nz: Grid dimensions
    """
    i, j, k = wp.tid()

    if i >= nx or j >= ny or k >= nz:
        return

    # Neumann BC at boundaries
    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
        pressure_new[i, j, k] = 0.0
        return

    # Coefficients for non-uniform grids
    ax = 1.0 / (dx * dx)
    ay = 1.0 / (dy * dy)
    az = 1.0 / (dz * dz)
    diag = 2.0 * (ax + ay + az)

    # RHS
    rhs = -density * divergence[i, j, k] / dt

    # Neighbor sum
    neighbor_sum = (
        ax * (pressure[i+1, j, k] + pressure[i-1, j, k]) +
        ay * (pressure[i, j+1, k] + pressure[i, j-1, k]) +
        az * (pressure[i, j, k+1] + pressure[i, j, k-1])
    )

    pressure_new[i, j, k] = (rhs + neighbor_sum) / diag


@wp.kernel
def pressure_poisson_sor(
    pressure: wp.array3d(dtype=float),
    divergence: wp.array3d(dtype=float),
    dx: float,
    dy: float,
    dz: float,
    density: float,
    dt: float,
    omega: float,
    nx: int,
    ny: int,
    nz: int,
    red_black: int,
):
    """
    Red-Black SOR iteration for pressure Poisson equation.

    Args:
        pressure: Pressure field (updated in-place)
        divergence: Velocity divergence
        dx, dy, dz: Grid spacing
        density: Fluid density
        dt: Time step
        omega: SOR relaxation factor (1.0-2.0, typically ~1.8)
        nx, ny, nz: Grid dimensions
        red_black: 0 for red cells, 1 for black cells
    """
    i, j, k = wp.tid()

    if i >= nx or j >= ny or k >= nz:
        return

    # Red-black ordering
    if ((i + j + k) % 2) != red_black:
        return

    # Boundary cells
    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
        pressure[i, j, k] = 0.0
        return

    ax = 1.0 / (dx * dx)
    ay = 1.0 / (dy * dy)
    az = 1.0 / (dz * dz)
    diag = 2.0 * (ax + ay + az)

    rhs = -density * divergence[i, j, k] / dt

    neighbor_sum = (
        ax * (pressure[i+1, j, k] + pressure[i-1, j, k]) +
        ay * (pressure[i, j+1, k] + pressure[i, j-1, k]) +
        az * (pressure[i, j, k+1] + pressure[i, j, k-1])
    )

    p_new = (rhs + neighbor_sum) / diag
    pressure[i, j, k] = (1.0 - omega) * pressure[i, j, k] + omega * p_new


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
    """
    Project velocity field to be divergence-free.

    V_new = V - (dt/rho) * grad(p)

    Args:
        vel_x, vel_y, vel_z: Velocity components (modified in-place)
        pressure: Pressure field
        density: Fluid density
        dt: Time step
        dx, dy, dz: Grid spacing
        nx, ny, nz: Grid dimensions
    """
    i, j, k = wp.tid()

    if i >= nx or j >= ny or k >= nz:
        return

    # Pressure gradient (central differences with boundary handling)
    im = wp.max(i - 1, 0)
    ip = wp.min(i + 1, nx - 1)
    jm = wp.max(j - 1, 0)
    jp = wp.min(j + 1, ny - 1)
    km = wp.max(k - 1, 0)
    kp = wp.min(k + 1, nz - 1)

    dpdx = (pressure[ip, j, k] - pressure[im, j, k]) / (2.0 * dx)
    dpdy = (pressure[i, jp, k] - pressure[i, jm, k]) / (2.0 * dy)
    dpdz = (pressure[i, j, kp] - pressure[i, j, km]) / (2.0 * dz)

    # Project
    coeff = dt / density
    vel_x[i, j, k] = vel_x[i, j, k] - coeff * dpdx
    vel_y[i, j, k] = vel_y[i, j, k] - coeff * dpdy
    vel_z[i, j, k] = vel_z[i, j, k] - coeff * dpdz


@wp.kernel
def compute_max_divergence(
    divergence: wp.array3d(dtype=float),
    max_div: wp.array(dtype=float),
    nx: int,
    ny: int,
    nz: int,
):
    """Compute maximum absolute divergence (for convergence check)."""
    i, j, k = wp.tid()

    if i >= nx or j >= ny or k >= nz:
        return

    div_abs = wp.abs(divergence[i, j, k])
    wp.atomic_max(max_div, 0, div_abs)
