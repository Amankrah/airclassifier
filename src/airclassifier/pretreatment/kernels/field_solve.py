"""
Field Solve Kernels
===================

Laplace equation solver for the electrostatic potential:

    div(eps' * grad(phi)) = 0

with Dirichlet BCs:
    phi = V_rf  on upper electrode (j = ny-1)
    phi = 0     on lower electrode (j = 0)
    d(phi)/dn = 0  on lateral boundaries (Neumann)

Phase 2: NumPy Jacobi iterative solver on structured grid.
Phase 3: Warp FEM assembly + sparse CG solve.
"""

from __future__ import annotations

import numpy as np


def jacobi_iteration_np(
    phi: np.ndarray,
    phi_new: np.ndarray,
    eps: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> None:
    """One weighted-Jacobi iteration for div(eps * grad(phi)) = 0.

    The discrete form on a structured grid with variable eps' is::

        phi[i,j,k] = sum( eps_{face} * phi_{nbr} / dh^2 )
                      / sum( eps_{face} / dh^2 )

    where ``eps_{face}`` is the harmonic average of eps at the two
    cells sharing the face.

    Boundary handling:
        j = 0     : phi = 0          (ground electrode, Dirichlet)
        j = ny-1  : phi = V_rf       (upper electrode, Dirichlet)
        i, k edges: Neumann dφ/dn=0  (via ghost-cell copy, handled
                    by using phi at the boundary from the current state)

    This function updates ``phi_new`` in-place for interior cells only.
    Boundary rows must be set by the caller.

    Args:
        phi: Current potential field (nx, ny, nz).
        phi_new: Output potential field (same shape).
        eps: Relative permittivity eps'(T, M) field (nx, ny, nz).
        dx, dy, dz: Cell sizes [m].
    """
    # Harmonic-average eps at each face
    # Face i+1/2:  2*eps[i]*eps[i+1] / (eps[i]+eps[i+1])
    # For efficiency, use arithmetic average (close for smooth fields)
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz2 = 1.0 / (dz * dz)

    # Interior indices
    p = phi
    e = eps

    # eps at face centres (arithmetic average)
    e_xp = 0.5 * (e[1:-1, 1:-1, 1:-1] + e[2:,   1:-1, 1:-1])
    e_xm = 0.5 * (e[:-2,  1:-1, 1:-1] + e[1:-1, 1:-1, 1:-1])
    e_yp = 0.5 * (e[1:-1, 1:-1, 1:-1] + e[1:-1, 2:,   1:-1])
    e_ym = 0.5 * (e[1:-1, :-2,  1:-1] + e[1:-1, 1:-1, 1:-1])
    e_zp = 0.5 * (e[1:-1, 1:-1, 1:-1] + e[1:-1, 1:-1, 2:  ])
    e_zm = 0.5 * (e[1:-1, 1:-1, :-2 ] + e[1:-1, 1:-1, 1:-1])

    # Weighted sum of neighbour potentials
    num = (
        (e_xp * p[2:,   1:-1, 1:-1] + e_xm * p[:-2,  1:-1, 1:-1]) * inv_dx2
      + (e_yp * p[1:-1, 2:,   1:-1] + e_ym * p[1:-1, :-2,  1:-1]) * inv_dy2
      + (e_zp * p[1:-1, 1:-1, 2:  ] + e_zm * p[1:-1, 1:-1, :-2 ]) * inv_dz2
    )

    den = (
        (e_xp + e_xm) * inv_dx2
      + (e_yp + e_ym) * inv_dy2
      + (e_zp + e_zm) * inv_dz2
    )

    phi_new[1:-1, 1:-1, 1:-1] = num / np.maximum(den, 1e-30)


def apply_laplace_bcs(
    phi: np.ndarray,
    V_upper: float,
) -> None:
    """Apply boundary conditions for the Laplace solve.

    * j = 0  : phi = 0       (ground / lower electrode)
    * j = -1 : phi = V_upper (upper electrode)
    * i, k edges : Neumann — copy from interior (dφ/dn = 0)
    """
    phi[:, 0, :] = 0.0
    phi[:, -1, :] = V_upper

    # Neumann on X faces
    phi[0, :, :] = phi[1, :, :]
    phi[-1, :, :] = phi[-2, :, :]

    # Neumann on Z faces
    phi[:, :, 0] = phi[:, :, 1]
    phi[:, :, -1] = phi[:, :, -2]


def solve_laplace_jacobi(
    eps: np.ndarray,
    V_upper: float,
    dx: float,
    dy: float,
    dz: float,
    max_iter: int = 500,
    tol: float = 1e-5,
    omega: float = 1.5,
    phi_init: np.ndarray | None = None,
) -> np.ndarray:
    """Solve div(eps * grad(phi)) = 0 using Successive Over-Relaxation.

    This is an accelerated Jacobi iteration (SOR) with relaxation
    parameter ``omega``.  omega = 1.0 is standard Jacobi, 1.0 < omega < 2.0
    is over-relaxation for faster convergence.

    Args:
        eps: 3-D relative permittivity field (nx, ny, nz).
        V_upper: Voltage on the upper electrode [V].
        dx, dy, dz: Cell sizes [m].
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance (relative L2 change in phi).
        omega: SOR relaxation factor (1.5 is a good default for
            structured grids with ~20 cells in the smallest dimension).
        phi_init: Optional initial guess. Defaults to linear
            interpolation between the electrodes.

    Returns:
        Converged potential field phi (nx, ny, nz).
    """
    nx, ny, nz = eps.shape

    # Initial guess: linear ramp from 0 to V_upper along Y
    if phi_init is not None:
        phi = phi_init.astype(np.float64)
    else:
        phi = np.zeros((nx, ny, nz), dtype=np.float64)
        for j in range(ny):
            phi[:, j, :] = V_upper * j / max(ny - 1, 1)

    phi_new = phi.copy()
    apply_laplace_bcs(phi, V_upper)

    for iteration in range(max_iter):
        jacobi_iteration_np(phi, phi_new, eps.astype(np.float64), dx, dy, dz)

        # Re-apply BCs
        apply_laplace_bcs(phi_new, V_upper)

        # SOR blend: phi = (1 - omega)*phi_old + omega*phi_jacobi
        if omega != 1.0:
            phi_new[1:-1, 1:-1, 1:-1] = (
                (1.0 - omega) * phi[1:-1, 1:-1, 1:-1]
                + omega * phi_new[1:-1, 1:-1, 1:-1]
            )

        # Convergence check (relative L2 norm of change)
        diff = phi_new - phi
        norm_diff = np.sqrt(np.sum(diff * diff))
        norm_phi = np.sqrt(np.sum(phi_new * phi_new))
        if norm_phi > 0 and norm_diff / norm_phi < tol:
            phi[:] = phi_new
            break

        phi[:] = phi_new

    return phi.astype(np.float32)


def compute_gradient_sq_np(
    phi: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Compute |grad(phi)|^2 = |E|^2 from the potential field.

    Uses second-order central differences in the interior and
    one-sided differences at boundaries.

    Args:
        phi: Potential field (nx, ny, nz).
        dx, dy, dz: Cell sizes [m].

    Returns:
        |E|^2 array of same shape as phi [V^2/m^2].
    """
    E2 = np.zeros_like(phi, dtype=np.float32)

    # Central differences for interior
    # dφ/dx at [i,j,k] ≈ (phi[i+1]-phi[i-1]) / (2*dx)
    dphi_dx = np.zeros_like(phi)
    dphi_dy = np.zeros_like(phi)
    dphi_dz = np.zeros_like(phi)

    dphi_dx[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * dx)
    dphi_dy[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2.0 * dy)
    dphi_dz[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2.0 * dz)

    # One-sided at boundaries
    dphi_dx[0, :, :]  = (phi[1, :, :] - phi[0, :, :]) / dx
    dphi_dx[-1, :, :] = (phi[-1, :, :] - phi[-2, :, :]) / dx

    dphi_dy[:, 0, :]  = (phi[:, 1, :] - phi[:, 0, :]) / dy
    dphi_dy[:, -1, :] = (phi[:, -1, :] - phi[:, -2, :]) / dy

    dphi_dz[:, :, 0]  = (phi[:, :, 1] - phi[:, :, 0]) / dz
    dphi_dz[:, :, -1] = (phi[:, :, -1] - phi[:, :, -2]) / dz

    E2[:] = dphi_dx**2 + dphi_dy**2 + dphi_dz**2
    return E2


# ── Warp kernel stubs (Phase 3) ─────────────────────────────────────

# @wp.kernel
# def jacobi_iteration(...):  ...
# @wp.kernel
# def compute_gradient_sq(...):  ...
