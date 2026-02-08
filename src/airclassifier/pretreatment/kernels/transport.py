"""
Material Transport Kernels
==========================

Conveyor belt advection of scalar fields (T, M) along the positive
X-axis.

Phase 1: First-order upwind (NumPy).
Phase 2: Van Leer TVD with flux limiter (NumPy) — §4.4.1.
Phase 3: @wp.kernel on GPU.
"""

from __future__ import annotations

import numpy as np


# ── Phase 1: first-order upwind ──────────────────────────────────────

def advect_material_np(
    field: np.ndarray,
    v_belt_m_per_s: float,
    dx: float,
    dt: float,
    inlet_value: float | np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Advect a 3-D scalar field along +X (conveyor direction).

    First-order upwind scheme::

        phi_new[i] = phi[i] - C * (phi[i] - phi[i-1])

    where ``C = v_belt * dt / dx`` is the Courant number (must be < 1).

    Args:
        field: 3-D scalar field to advect (e.g. T or M).
        v_belt_m_per_s: Belt velocity [m/s].
        dx: Cell size in X direction [m].
        dt: Timestep [s].
        inlet_value: Value at infeed boundary (scalar or 2-D array
            of shape ``(ny, nz)``).
        out: Optional pre-allocated output of same shape.

    Returns:
        Advected field.
    """
    C = v_belt_m_per_s * dt / dx  # Courant number
    if C > 1.0:
        raise ValueError(
            f"Courant number {C:.3f} > 1 — reduce dt or increase dx."
        )

    if out is None:
        out = np.empty_like(field)

    # Infeed (i=0): inject fresh material
    out[0, :, :] = inlet_value

    # Interior + outfeed: upwind
    out[1:, :, :] = field[1:, :, :] - C * (field[1:, :, :] - field[:-1, :, :])

    return out


# ── Phase 2: Van Leer TVD with flux limiter ──────────────────────────

def _van_leer_limiter(r: np.ndarray) -> np.ndarray:
    """Van Leer flux limiter: psi(r) = (r + |r|) / (1 + |r|)."""
    return (r + np.abs(r)) / (1.0 + np.abs(r))


def advect_material_tvd_np(
    field: np.ndarray,
    v_belt_m_per_s: float,
    dx: float,
    dt: float,
    inlet_value: float | np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Advect a 3-D scalar field along +X using Van Leer TVD.

    Second-order flux-limited scheme from the engineering guide §4.4.1::

        r = (phi[i] - phi[i-1]) / (phi[i+1] - phi[i])
        psi(r) = (r + |r|) / (1 + |r|)         # Van Leer limiter
        phi_new[i] = phi[i] - C * (phi[i] - phi[i-1])
                     - 0.5 * C * (1 - C) * psi(r) * (phi[i+1] - phi[i])
                     + 0.5 * C * (1 - C) * psi(1/r) * (phi[i] - phi[i-1])

    Simplifies to the MUSCL form with Van Leer limiter.

    Courant number C = v_belt * dt / dx must be in (0, 1).

    Args:
        field: 3-D scalar field to advect.
        v_belt_m_per_s: Belt velocity [m/s].
        dx: Cell size in X direction [m].
        dt: Timestep [s].
        inlet_value: Value at infeed boundary.
        out: Optional pre-allocated output.

    Returns:
        Advected field.
    """
    C = v_belt_m_per_s * dt / dx
    if C > 1.0:
        raise ValueError(
            f"Courant number {C:.3f} > 1 — reduce dt or increase dx."
        )
    if C <= 0.0:
        if out is None:
            return field.copy()
        out[:] = field
        return out

    if out is None:
        out = np.empty_like(field)

    nx = field.shape[0]

    # Infeed (i=0): inject fresh material
    out[0, :, :] = inlet_value

    # i=1: only upwind available (no i-2 for the slope ratio)
    out[1, :, :] = field[1, :, :] - C * (field[1, :, :] - field[0, :, :])

    if nx > 3:
        # Interior i = 2 .. nx-2 : full TVD stencil  [i-1, i, i+1]
        phi_m1 = field[1:-2, :, :]    # phi[i-1]   for i in [2..nx-2]
        phi_0 = field[2:-1, :, :]     # phi[i]
        phi_p1 = field[3:, :, :]      # phi[i+1]   (only exists up to nx-2)

        # But we need phi[i-1], phi[i], phi[i+1] for i=2..nx-2
        # Let me redo indices carefully.
        # For i in range(2, nx-1):
        #   phi_im1 = field[i-1]
        #   phi_i   = field[i]
        #   phi_ip1 = field[i+1]  (exists for i up to nx-2)
        phi_im1 = field[1:-2, :, :]   # i-1 for i in [2, nx-1)
        phi_i = field[2:-1, :, :]     # i
        phi_ip1 = field[3:, :, :]     # i+1  — shape is (nx-3, ...)

        # Need same-size slices: use i in [2, nx-2) so i+1 exists
        # That means phi_im1 = field[1:-2], phi_i = field[2:-1], phi_ip1 = field[3:]
        # all have shape (nx-3, ny, nz)

        delta_fwd = phi_ip1 - phi_i   # phi[i+1] - phi[i]
        delta_bwd = phi_i - phi_im1   # phi[i] - phi[i-1]

        # Slope ratio r = delta_bwd / delta_fwd (with zero-division guard)
        r = np.where(
            np.abs(delta_fwd) > 1e-30,
            delta_bwd / delta_fwd,
            np.where(delta_bwd > 0, 1e10, np.where(delta_bwd < 0, -1e10, 0.0)),
        )

        psi = _van_leer_limiter(r)

        # TVD flux: upwind + anti-diffusive correction
        out[2:-1, :, :] = (
            phi_i
            - C * delta_bwd
            - 0.5 * C * (1.0 - C) * (psi * delta_fwd - psi * delta_bwd)
        )

    # Outfeed (i=nx-1): fall back to upwind (no phi[i+1])
    if nx > 1:
        out[-1, :, :] = field[-1, :, :] - C * (field[-1, :, :] - field[-2, :, :])

    return out


# ── Warp kernel stub (Phase 3) ──────────────────────────────────────

# @wp.kernel
# def advect_material(...):  ...
