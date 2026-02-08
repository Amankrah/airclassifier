"""
Material Transport Kernels
==========================

Conveyor belt advection of scalar fields (T, M) along the positive
X-axis using a first-order upwind scheme.

Phase 1: NumPy vectorised implementation.
Phase 2+: @wp.kernel on GPU.
"""

from __future__ import annotations

import numpy as np


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

    nx = field.shape[0]

    # Infeed (i=0): inject fresh material
    if np.isscalar(inlet_value):
        out[0, :, :] = inlet_value
    else:
        out[0, :, :] = inlet_value

    # Interior + outfeed: upwind
    out[1:, :, :] = field[1:, :, :] - C * (field[1:, :, :] - field[:-1, :, :])

    return out


# ── Warp kernel stub (Phase 2+) ─────────────────────────────────────

# @wp.kernel
# def advect_material(...):  ...
