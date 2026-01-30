"""
GPU kernels for fluid dynamics computations.

Provides Warp-accelerated kernels for advection, diffusion,
and pressure projection operations.
"""

from .advection import (
    trilinear_interp,
    advect_semi_lagrangian_scalar,
    advect_upwind_scalar,
    advect_maccormack_predict,
    advect_maccormack_correct,
)

from .diffusion import (
    diffuse_explicit,
    diffuse_jacobi_iteration,
    diffuse_with_variable_diffusivity,
    compute_diffusion_stability_limit,
)

from .projection import (
    compute_divergence,
    pressure_poisson_jacobi,
    pressure_poisson_sor,
    project_velocity,
    compute_max_divergence,
)

__all__ = [
    # Advection
    "trilinear_interp",
    "advect_semi_lagrangian_scalar",
    "advect_upwind_scalar",
    "advect_maccormack_predict",
    "advect_maccormack_correct",
    # Diffusion
    "diffuse_explicit",
    "diffuse_jacobi_iteration",
    "diffuse_with_variable_diffusivity",
    "compute_diffusion_stability_limit",
    # Projection
    "compute_divergence",
    "pressure_poisson_jacobi",
    "pressure_poisson_sor",
    "project_velocity",
    "compute_max_divergence",
]
