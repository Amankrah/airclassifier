"""
Field Solve Kernels
===================

Laplace equation solver for the electrostatic potential:

    div(eps' * grad(phi)) = 0

with Dirichlet BCs:
    phi = V_rf  on upper electrode (j = j_upper .. ny-1)
    phi = 0     on lower electrode (j = 0)
    d(phi)/dn = 0  on lateral boundaries (Neumann)

Phase 2: NumPy Jacobi iterative solver on structured grid.
Phase 3: Warp FEM assembly + sparse CG solve.
"""

from __future__ import annotations

import numpy as np


def _compute_weighted_avg(
    phi: np.ndarray,
    eps: np.ndarray,
    inv_dx2: float,
    inv_dy2: float,
    inv_dz2: float,
) -> tuple:
    """Compute the weighted-average numerator and denominator for all interior cells.

    Returns (num, den) arrays of shape (nx-2, ny-2, nz-2).
    """
    e = eps
    p = phi
    e_xp = 0.5 * (e[1:-1, 1:-1, 1:-1] + e[2:,   1:-1, 1:-1])
    e_xm = 0.5 * (e[:-2,  1:-1, 1:-1] + e[1:-1, 1:-1, 1:-1])
    e_yp = 0.5 * (e[1:-1, 1:-1, 1:-1] + e[1:-1, 2:,   1:-1])
    e_ym = 0.5 * (e[1:-1, :-2,  1:-1] + e[1:-1, 1:-1, 1:-1])
    e_zp = 0.5 * (e[1:-1, 1:-1, 1:-1] + e[1:-1, 1:-1, 2:  ])
    e_zm = 0.5 * (e[1:-1, 1:-1, :-2 ] + e[1:-1, 1:-1, 1:-1])

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
    return num, den


def jacobi_iteration_np(
    phi: np.ndarray,
    phi_new: np.ndarray,
    eps: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> None:
    """One weighted-Jacobi iteration for div(eps * grad(phi)) = 0.

    Updates ``phi_new`` in-place for interior cells only.
    Boundary rows must be set by the caller.
    """
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz2 = 1.0 / (dz * dz)
    num, den = _compute_weighted_avg(phi, eps, inv_dx2, inv_dy2, inv_dz2)
    phi_new[1:-1, 1:-1, 1:-1] = num / np.maximum(den, 1e-30)


def _redblack_gs_sor_sweep(
    phi: np.ndarray,
    eps: np.ndarray,
    inv_dx2: float,
    inv_dy2: float,
    inv_dz2: float,
    omega: float,
    j_upper: int,
    color: int,
) -> None:
    """One half-sweep of Red-Black Gauss-Seidel SOR (in-place).

    Gauss-Seidel uses the LATEST values of phi (including cells already
    updated in this sweep).  Red-Black ordering allows vectorised NumPy
    updates: first update all "red" cells (i+j+k even), then all "black"
    cells using the freshly-updated red values.

    This is unconditionally stable for omega in (0, 2) and converges
    much faster than Jacobi+SOR, which diverges for omega > ~1.003
    on anisotropic grids (dx != dy).

    Args:
        phi: Potential field, updated IN-PLACE for cells of the given color.
        color: 0 for red cells (i+j+k even), 1 for black cells.
        j_upper: Only update cells with j < j_upper (electrode boundary).
    """
    nx, ny, nz = phi.shape
    # Recompute weighted average using CURRENT phi (includes cells
    # already updated in the red sweep when processing black).
    num, den = _compute_weighted_avg(phi, eps, inv_dx2, inv_dy2, inv_dz2)
    phi_star = num / np.maximum(den, 1e-30)

    # Build color mask for interior cells (shape: nx-2, ny-2, nz-2)
    # Red: (i+j+k) % 2 == 0,  Black: (i+j+k) % 2 == 1
    # Interior indices: i in [1, nx-2], j in [1, ny-2], k in [1, nz-2]
    i_idx = np.arange(1, nx - 1).reshape(-1, 1, 1)
    j_idx = np.arange(1, ny - 1).reshape(1, -1, 1)
    k_idx = np.arange(1, nz - 1).reshape(1, 1, -1)
    mask = ((i_idx + j_idx + k_idx) % 2 == color)

    # Only update cells within the active gap (j < j_upper)
    j_limit = min(j_upper, ny - 1)
    j_local_limit = j_limit - 1  # in interior-index space (j=1 -> index 0)
    if j_local_limit <= 0:
        return
    mask[:, j_local_limit:, :] = False

    # SOR update in-place
    interior = phi[1:-1, 1:-1, 1:-1]
    interior[mask] = (1.0 - omega) * interior[mask] + omega * phi_star[mask]


def apply_laplace_bcs(
    phi: np.ndarray,
    V_upper: float,
    j_upper: int | None = None,
) -> None:
    """Apply boundary conditions for the Laplace solve.

    * j = 0         : phi = 0       (ground / lower electrode)
    * j >= j_upper  : phi = V_upper (upper electrode and above)
    * i, k edges    : Neumann — copy from interior (dφ/dn = 0)

    Args:
        phi: Potential field (nx, ny, nz).
        V_upper: Voltage on the upper electrode [V].
        j_upper: Y-index of the upper electrode. If None, defaults
            to ny-1 (legacy: electrode at grid top).
    """
    ny = phi.shape[1]
    if j_upper is None:
        j_upper = ny - 1

    phi[:, 0, :] = 0.0
    phi[:, j_upper:, :] = V_upper

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
    j_upper: int | None = None,
) -> np.ndarray:
    """Solve div(eps * grad(phi)) = 0 using Red-Black Gauss-Seidel SOR.

    Red-Black Gauss-Seidel updates cells in a checkerboard pattern so
    that all neighbors of each updated cell hold *current-iteration*
    values.  This is unconditionally stable for omega in (0, 2) and
    converges much faster than Jacobi + SOR (which diverges when
    omega > ~1.003 on anisotropic grids with dx != dy).

    Args:
        eps: 3-D relative permittivity field (nx, ny, nz).
        V_upper: Voltage on the upper electrode [V].
        dx, dy, dz: Cell sizes [m].
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance (relative L2 change in phi).
        omega: SOR relaxation factor.  1.5 is a good default for
            structured grids with ~20 cells per dimension.
        phi_init: Optional initial guess. Defaults to linear
            interpolation between the electrodes.
        j_upper: Y-index of the upper electrode. If None, defaults
            to ny-1 (legacy: electrode at grid top). The potential is
            pinned to V_upper at j >= j_upper.

    Returns:
        Converged potential field phi (nx, ny, nz).
    """
    nx, ny, nz = eps.shape
    if j_upper is None:
        j_upper = ny - 1

    # Ensure eps >= 1.0 everywhere (air permittivity minimum).
    eps_safe = np.maximum(eps.astype(np.float64), 1.0)

    # Initial guess: linear ramp from 0 to V_upper within the gap,
    # constant V_upper above the upper electrode.
    use_ramp = True
    if phi_init is not None:
        phi = phi_init.astype(np.float64)
        if np.isfinite(phi).all():
            use_ramp = False
    if use_ramp:
        phi = np.zeros((nx, ny, nz), dtype=np.float64)
        for j in range(ny):
            if j <= j_upper:
                phi[:, j, :] = V_upper * j / max(j_upper, 1)
            else:
                phi[:, j, :] = V_upper

    apply_laplace_bcs(phi, V_upper, j_upper)
    phi_prev = phi.copy()

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz2 = 1.0 / (dz * dz)

    for iteration in range(max_iter):
        # Red-Black Gauss-Seidel SOR: two half-sweeps per iteration.
        # Red sweep: update cells where (i+j+k) % 2 == 0
        _redblack_gs_sor_sweep(
            phi, eps_safe, inv_dx2, inv_dy2, inv_dz2,
            omega, j_upper, color=0,
        )
        apply_laplace_bcs(phi, V_upper, j_upper)

        # Black sweep: update cells where (i+j+k) % 2 == 1
        _redblack_gs_sor_sweep(
            phi, eps_safe, inv_dx2, inv_dy2, inv_dz2,
            omega, j_upper, color=1,
        )
        apply_laplace_bcs(phi, V_upper, j_upper)

        # Clamp to physical range [0, V_upper] (maximum principle)
        np.clip(phi, 0.0, V_upper, out=phi)
        apply_laplace_bcs(phi, V_upper, j_upper)

        # Convergence check (relative L2 norm of change)
        diff = phi - phi_prev
        norm_diff = np.sqrt(np.sum(diff * diff))
        norm_phi = np.sqrt(np.sum(phi * phi))
        if norm_phi > 0 and norm_diff / norm_phi < tol:
            break
        phi_prev[:] = phi

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


# ── Warp GPU kernels ─────────────────────────────────────────────────

try:
    import warp as wp

    @wp.kernel
    def jacobi_iteration_wp(
        phi: wp.array3d(dtype=float),
        phi_new: wp.array3d(dtype=float),
        eps: wp.array3d(dtype=float),
        inv_dx2: float, inv_dy2: float, inv_dz2: float,
        nx: int, ny: int, nz: int,
    ):
        """One Jacobi iteration for div(eps * grad(phi)) = 0."""
        i, j, k = wp.tid()
        if i <= 0 or i >= nx - 1 or j <= 0 or j >= ny - 1 or k <= 0 or k >= nz - 1:
            return

        e_xp = 0.5 * (eps[i, j, k] + eps[i + 1, j, k])
        e_xm = 0.5 * (eps[i - 1, j, k] + eps[i, j, k])
        e_yp = 0.5 * (eps[i, j, k] + eps[i, j + 1, k])
        e_ym = 0.5 * (eps[i, j - 1, k] + eps[i, j, k])
        e_zp = 0.5 * (eps[i, j, k] + eps[i, j, k + 1])
        e_zm = 0.5 * (eps[i, j, k - 1] + eps[i, j, k])

        num = (
            (e_xp * phi[i + 1, j, k] + e_xm * phi[i - 1, j, k]) * inv_dx2
            + (e_yp * phi[i, j + 1, k] + e_ym * phi[i, j - 1, k]) * inv_dy2
            + (e_zp * phi[i, j, k + 1] + e_zm * phi[i, j, k - 1]) * inv_dz2
        )
        den = (
            (e_xp + e_xm) * inv_dx2
            + (e_yp + e_ym) * inv_dy2
            + (e_zp + e_zm) * inv_dz2
        )

        phi_new[i, j, k] = num / wp.max(den, 1.0e-30)

    @wp.kernel
    def compute_gradient_sq_wp(
        phi: wp.array3d(dtype=float),
        E2: wp.array3d(dtype=float),
        dx: float, dy: float, dz: float,
        nx: int, ny: int, nz: int,
    ):
        """Compute |grad(phi)|^2 = |E|^2 from the potential field."""
        i, j, k = wp.tid()
        if i >= nx or j >= ny or k >= nz:
            return

        im = wp.max(i - 1, 0)
        ip = wp.min(i + 1, nx - 1)
        jm = wp.max(j - 1, 0)
        jp = wp.min(j + 1, ny - 1)
        km = wp.max(k - 1, 0)
        kp = wp.min(k + 1, nz - 1)

        # Use safe difference denominators
        dx_eff = dx * float(ip - im)
        dy_eff = dy * float(jp - jm)
        dz_eff = dz * float(kp - km)

        dphi_dx = (phi[ip, j, k] - phi[im, j, k]) / wp.max(dx_eff, 1.0e-30)
        dphi_dy = (phi[i, jp, k] - phi[i, jm, k]) / wp.max(dy_eff, 1.0e-30)
        dphi_dz = (phi[i, j, kp] - phi[i, j, km]) / wp.max(dz_eff, 1.0e-30)

        E2[i, j, k] = dphi_dx * dphi_dx + dphi_dy * dphi_dy + dphi_dz * dphi_dz

    @wp.kernel
    def _apply_laplace_bcs_wp(
        phi: wp.array3d(dtype=float),
        V_upper: float,
        j_upper: int,
        nx: int, ny: int, nz: int,
    ):
        """Apply Dirichlet + Neumann BCs on the GPU."""
        i, j, k = wp.tid()
        if i >= nx or j >= ny or k >= nz:
            return

        # Ground electrode (j=0)
        if j == 0:
            phi[i, j, k] = 0.0
        # Upper electrode and above
        if j >= j_upper:
            phi[i, j, k] = V_upper
        # Neumann on X faces
        if i == 0:
            phi[i, j, k] = phi[1, j, k]
        if i == nx - 1:
            phi[i, j, k] = phi[nx - 2, j, k]
        # Neumann on Z faces
        if k == 0:
            phi[i, j, k] = phi[i, j, 1]
        if k == nz - 1:
            phi[i, j, k] = phi[i, j, nz - 2]

    def solve_laplace_jacobi_gpu(
        eps: np.ndarray,
        V_upper: float,
        dx: float,
        dy: float,
        dz: float,
        max_iter: int = 2000,
        tol: float = 1e-5,
        omega: float = 1.5,
        phi_init: np.ndarray | None = None,
        j_upper: int | None = None,
        device: str = "cuda:0",
    ) -> np.ndarray:
        """GPU-accelerated Laplace solver using Warp kernels.

        Uses pure Jacobi iterations (no SOR blend) on the GPU.  Jacobi
        is embarrassingly parallel and maps well to GPU hardware.  SOR
        is NOT applied because the Jacobi update with omega > 1 diverges
        on anisotropic grids (dx != dy); the stability limit for Jacobi
        SOR is omega < 2/(1+rho_J) which is near 1.0 for typical grids.

        Convergence is checked every 50 iterations on the CPU.  The
        default max_iter of 2000 provides sufficient headroom for
        grids with anisotropic cell sizes.

        Returns:
            Converged potential field phi (nx, ny, nz) as NumPy float32.
        """
        nx, ny, nz = eps.shape
        if j_upper is None:
            j_upper = ny - 1

        eps_safe = np.maximum(eps.astype(np.float64), 1.0).astype(np.float32)

        # Initial guess: linear ramp within gap
        use_ramp = True
        if phi_init is not None:
            phi_np = phi_init.astype(np.float32)
            if np.isfinite(phi_np).all():
                use_ramp = False
        if use_ramp:
            phi_np = np.zeros((nx, ny, nz), dtype=np.float32)
            for j in range(ny):
                if j <= j_upper:
                    phi_np[:, j, :] = V_upper * j / max(j_upper, 1)
                else:
                    phi_np[:, j, :] = V_upper

        # Apply BCs on CPU for initial state
        apply_laplace_bcs(phi_np, V_upper, j_upper)

        # Upload to GPU
        phi_gpu = wp.array(phi_np, dtype=wp.float32, device=device)
        phi_new_gpu = wp.array(phi_np.copy(), dtype=wp.float32, device=device)
        eps_gpu = wp.array(eps_safe, dtype=wp.float32, device=device)

        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        inv_dz2 = 1.0 / (dz * dz)

        check_interval = 50  # convergence check every N iterations

        for iteration in range(max_iter):
            # Jacobi iteration on GPU (pure, no SOR blend)
            wp.launch(jacobi_iteration_wp, dim=(nx, ny, nz),
                      inputs=[phi_gpu, phi_new_gpu, eps_gpu,
                              inv_dx2, inv_dy2, inv_dz2,
                              nx, ny, nz],
                      device=device)

            # Apply BCs on GPU
            wp.launch(_apply_laplace_bcs_wp, dim=(nx, ny, nz),
                      inputs=[phi_new_gpu, V_upper,
                              j_upper, nx, ny, nz],
                      device=device)

            # Convergence check every N iterations (sync to CPU)
            if (iteration + 1) % check_interval == 0 or iteration == max_iter - 1:
                wp.synchronize()
                phi_np = phi_gpu.numpy()
                phi_new_np = phi_new_gpu.numpy()

                # Clamp to physical range
                np.clip(phi_new_np, 0.0, V_upper, out=phi_new_np)
                apply_laplace_bcs(phi_new_np, V_upper, j_upper)

                # Convergence check
                diff = phi_new_np - phi_np
                norm_diff = np.sqrt(np.sum(diff * diff))
                norm_phi = np.sqrt(np.sum(phi_new_np * phi_new_np))
                if norm_phi > 0 and norm_diff / norm_phi < tol:
                    return phi_new_np

                # Re-upload clamped result for next batch
                wp.copy(phi_gpu, wp.array(phi_new_np, dtype=wp.float32, device="cpu"))
                wp.copy(phi_new_gpu, wp.array(phi_new_np, dtype=wp.float32, device="cpu"))
            else:
                # Swap buffers on GPU (no CPU sync)
                phi_gpu, phi_new_gpu = phi_new_gpu, phi_gpu

        # Final download
        wp.synchronize()
        return phi_gpu.numpy().astype(np.float32)

    _HAS_WARP_FIELD = True

except ImportError:
    _HAS_WARP_FIELD = False
