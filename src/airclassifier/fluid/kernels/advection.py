"""
Advection kernels for CFD solver.

Provides semi-Lagrangian and upwind advection schemes
for velocity field transport.
"""

import warp as wp


@wp.func
def trilinear_interp(
    field: wp.array3d(dtype=float),
    gx: float,
    gy: float,
    gz: float,
    nx: int,
    ny: int,
    nz: int
) -> float:
    """
    Trilinear interpolation from field at position (gx, gy, gz).

    Args:
        field: 3D field to interpolate
        gx, gy, gz: Grid coordinates (floating point)
        nx, ny, nz: Grid dimensions

    Returns:
        Interpolated value
    """
    # Clamp to valid range
    gx = wp.clamp(gx, 0.5, float(nx) - 1.5)
    gy = wp.clamp(gy, 0.5, float(ny) - 1.5)
    gz = wp.clamp(gz, 0.5, float(nz) - 1.5)

    # Integer indices
    i0 = int(wp.floor(gx))
    j0 = int(wp.floor(gy))
    k0 = int(wp.floor(gz))

    i1 = wp.min(i0 + 1, nx - 1)
    j1 = wp.min(j0 + 1, ny - 1)
    k1 = wp.min(k0 + 1, nz - 1)

    # Weights
    sx = gx - float(i0)
    sy = gy - float(j0)
    sz = gz - float(k0)

    # Interpolate
    return (
        field[i0, j0, k0] * (1.0 - sx) * (1.0 - sy) * (1.0 - sz) +
        field[i1, j0, k0] * sx * (1.0 - sy) * (1.0 - sz) +
        field[i0, j1, k0] * (1.0 - sx) * sy * (1.0 - sz) +
        field[i0, j0, k1] * (1.0 - sx) * (1.0 - sy) * sz +
        field[i1, j1, k0] * sx * sy * (1.0 - sz) +
        field[i1, j0, k1] * sx * (1.0 - sy) * sz +
        field[i0, j1, k1] * (1.0 - sx) * sy * sz +
        field[i1, j1, k1] * sx * sy * sz
    )


@wp.kernel
def advect_semi_lagrangian_scalar(
    field: wp.array3d(dtype=float),
    field_new: wp.array3d(dtype=float),
    vel_x: wp.array3d(dtype=float),
    vel_y: wp.array3d(dtype=float),
    vel_z: wp.array3d(dtype=float),
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Semi-Lagrangian advection for a scalar field."""
    i, j, k = wp.tid()

    if i >= nx or j >= ny or k >= nz:
        return

    # Velocity at this cell
    vx = vel_x[i, j, k]
    vy = vel_y[i, j, k]
    vz = vel_z[i, j, k]

    # Backtrace position
    px = float(i) * dx - vx * dt
    py = float(j) * dy - vy * dt
    pz = float(k) * dz - vz * dt

    # Convert to grid coordinates
    gx = px / dx
    gy = py / dy
    gz = pz / dz

    # Interpolate
    field_new[i, j, k] = trilinear_interp(field, gx, gy, gz, nx, ny, nz)


@wp.kernel
def advect_upwind_scalar(
    field: wp.array3d(dtype=float),
    field_new: wp.array3d(dtype=float),
    vel_x: wp.array3d(dtype=float),
    vel_y: wp.array3d(dtype=float),
    vel_z: wp.array3d(dtype=float),
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    nx: int,
    ny: int,
    nz: int,
):
    """First-order upwind advection for a scalar field."""
    i, j, k = wp.tid()

    if i <= 0 or i >= nx - 1 or j <= 0 or j >= ny - 1 or k <= 0 or k >= nz - 1:
        if i < nx and j < ny and k < nz:
            field_new[i, j, k] = field[i, j, k]
        return

    # Velocities
    vx = vel_x[i, j, k]
    vy = vel_y[i, j, k]
    vz = vel_z[i, j, k]

    # Upwind differences
    if vx > 0.0:
        dfdx = (field[i, j, k] - field[i-1, j, k]) / dx
    else:
        dfdx = (field[i+1, j, k] - field[i, j, k]) / dx

    if vy > 0.0:
        dfdy = (field[i, j, k] - field[i, j-1, k]) / dy
    else:
        dfdy = (field[i, j+1, k] - field[i, j, k]) / dy

    if vz > 0.0:
        dfdz = (field[i, j, k] - field[i, j, k-1]) / dz
    else:
        dfdz = (field[i, j, k+1] - field[i, j, k]) / dz

    # Update
    advection = vx * dfdx + vy * dfdy + vz * dfdz
    field_new[i, j, k] = field[i, j, k] - dt * advection


@wp.kernel
def advect_maccormack_predict(
    field: wp.array3d(dtype=float),
    field_pred: wp.array3d(dtype=float),
    vel_x: wp.array3d(dtype=float),
    vel_y: wp.array3d(dtype=float),
    vel_z: wp.array3d(dtype=float),
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Predictor step for MacCormack advection (forward differences)."""
    i, j, k = wp.tid()

    if i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        if i < nx and j < ny and k < nz:
            field_pred[i, j, k] = field[i, j, k]
        return

    vx = vel_x[i, j, k]
    vy = vel_y[i, j, k]
    vz = vel_z[i, j, k]

    # Forward differences
    dfdx = (field[i+1, j, k] - field[i, j, k]) / dx
    dfdy = (field[i, j+1, k] - field[i, j, k]) / dy
    dfdz = (field[i, j, k+1] - field[i, j, k]) / dz

    advection = vx * dfdx + vy * dfdy + vz * dfdz
    field_pred[i, j, k] = field[i, j, k] - dt * advection


@wp.kernel
def advect_maccormack_correct(
    field: wp.array3d(dtype=float),
    field_pred: wp.array3d(dtype=float),
    field_new: wp.array3d(dtype=float),
    vel_x: wp.array3d(dtype=float),
    vel_y: wp.array3d(dtype=float),
    vel_z: wp.array3d(dtype=float),
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Corrector step for MacCormack advection (backward differences)."""
    i, j, k = wp.tid()

    if i <= 0 or j <= 0 or k <= 0 or i >= nx or j >= ny or k >= nz:
        if i < nx and j < ny and k < nz:
            field_new[i, j, k] = field_pred[i, j, k]
        return

    vx = vel_x[i, j, k]
    vy = vel_y[i, j, k]
    vz = vel_z[i, j, k]

    # Backward differences on predicted field
    dfdx = (field_pred[i, j, k] - field_pred[i-1, j, k]) / dx
    dfdy = (field_pred[i, j, k] - field_pred[i, j-1, k]) / dy
    dfdz = (field_pred[i, j, k] - field_pred[i, j, k-1]) / dz

    advection = vx * dfdx + vy * dfdy + vz * dfdz
    field_corr = field_pred[i, j, k] - dt * advection

    # Average predictor and corrector
    field_new[i, j, k] = 0.5 * (field[i, j, k] + field_corr)
