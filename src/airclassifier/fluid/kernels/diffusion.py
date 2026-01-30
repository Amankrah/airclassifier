"""
Diffusion kernels for CFD solver.

Provides explicit and implicit diffusion operators
for viscous transport.
"""

import warp as wp


@wp.kernel
def diffuse_explicit(
    field: wp.array3d(dtype=float),
    field_new: wp.array3d(dtype=float),
    diffusivity: float,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    nx: int,
    ny: int,
    nz: int,
):
    """
    Explicit diffusion step using central differences.

    Updates field_new = field + dt * D * laplacian(field)

    Args:
        field: Input scalar field
        field_new: Output field
        diffusivity: Diffusion coefficient (e.g., kinematic viscosity)
        dt: Time step
        dx, dy, dz: Grid spacing
        nx, ny, nz: Grid dimensions
    """
    i, j, k = wp.tid()

    if i <= 0 or i >= nx - 1 or j <= 0 or j >= ny - 1 or k <= 0 or k >= nz - 1:
        if i < nx and j < ny and k < nz:
            field_new[i, j, k] = field[i, j, k]
        return

    # Second derivatives (Laplacian)
    d2fdx2 = (field[i+1, j, k] - 2.0*field[i, j, k] + field[i-1, j, k]) / (dx*dx)
    d2fdy2 = (field[i, j+1, k] - 2.0*field[i, j, k] + field[i, j-1, k]) / (dy*dy)
    d2fdz2 = (field[i, j, k+1] - 2.0*field[i, j, k] + field[i, j, k-1]) / (dz*dz)

    laplacian = d2fdx2 + d2fdy2 + d2fdz2

    field_new[i, j, k] = field[i, j, k] + dt * diffusivity * laplacian


@wp.kernel
def diffuse_jacobi_iteration(
    field: wp.array3d(dtype=float),
    field_new: wp.array3d(dtype=float),
    field_rhs: wp.array3d(dtype=float),
    alpha: float,
    beta: float,
    nx: int,
    ny: int,
    nz: int,
):
    """
    Jacobi iteration for implicit diffusion.

    Solves: (1 - alpha*laplacian) * field_new = field_rhs

    Args:
        field: Current iteration
        field_new: Next iteration
        field_rhs: Right-hand side (initial field / dt)
        alpha: dt * diffusivity / dx^2
        beta: 1 / (1 + 6*alpha) for uniform grid
        nx, ny, nz: Grid dimensions
    """
    i, j, k = wp.tid()

    if i <= 0 or i >= nx - 1 or j <= 0 or j >= ny - 1 or k <= 0 or k >= nz - 1:
        if i < nx and j < ny and k < nz:
            field_new[i, j, k] = field[i, j, k]
        return

    neighbor_sum = (
        field[i+1, j, k] + field[i-1, j, k] +
        field[i, j+1, k] + field[i, j-1, k] +
        field[i, j, k+1] + field[i, j, k-1]
    )

    field_new[i, j, k] = beta * (field_rhs[i, j, k] + alpha * neighbor_sum)


@wp.kernel
def diffuse_with_variable_diffusivity(
    field: wp.array3d(dtype=float),
    field_new: wp.array3d(dtype=float),
    diffusivity: wp.array3d(dtype=float),
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    nx: int,
    ny: int,
    nz: int,
):
    """
    Diffusion with spatially varying diffusivity (e.g., turbulent viscosity).

    Args:
        field: Input field
        field_new: Output field
        diffusivity: Spatially varying diffusion coefficient
        dt: Time step
        dx, dy, dz: Grid spacing
        nx, ny, nz: Grid dimensions
    """
    i, j, k = wp.tid()

    if i <= 0 or i >= nx - 1 or j <= 0 or j >= ny - 1 or k <= 0 or k >= nz - 1:
        if i < nx and j < ny and k < nz:
            field_new[i, j, k] = field[i, j, k]
        return

    # Face-averaged diffusivities
    D_xp = 0.5 * (diffusivity[i, j, k] + diffusivity[i+1, j, k])
    D_xm = 0.5 * (diffusivity[i, j, k] + diffusivity[i-1, j, k])
    D_yp = 0.5 * (diffusivity[i, j, k] + diffusivity[i, j+1, k])
    D_ym = 0.5 * (diffusivity[i, j, k] + diffusivity[i, j-1, k])
    D_zp = 0.5 * (diffusivity[i, j, k] + diffusivity[i, j, k+1])
    D_zm = 0.5 * (diffusivity[i, j, k] + diffusivity[i, j, k-1])

    # Flux form of diffusion
    flux_x = (D_xp * (field[i+1, j, k] - field[i, j, k]) -
              D_xm * (field[i, j, k] - field[i-1, j, k])) / (dx * dx)
    flux_y = (D_yp * (field[i, j+1, k] - field[i, j, k]) -
              D_ym * (field[i, j, k] - field[i, j-1, k])) / (dy * dy)
    flux_z = (D_zp * (field[i, j, k+1] - field[i, j, k]) -
              D_zm * (field[i, j, k] - field[i, j, k-1])) / (dz * dz)

    field_new[i, j, k] = field[i, j, k] + dt * (flux_x + flux_y + flux_z)


def compute_diffusion_stability_limit(
    diffusivity: float,
    dx: float,
    dy: float,
    dz: float
) -> float:
    """
    Compute maximum stable time step for explicit diffusion.

    dt_max = dx^2 / (2 * D * n_dims) for n_dims dimensions.

    Args:
        diffusivity: Diffusion coefficient
        dx, dy, dz: Grid spacing

    Returns:
        Maximum stable time step
    """
    min_dx = min(dx, dy, dz)
    return min_dx**2 / (6.0 * diffusivity)
