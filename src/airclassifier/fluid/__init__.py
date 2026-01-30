"""
Fluid dynamics module for cyclone air classifier.

Provides flow field representations, boundary conditions,
CFD solvers, turbulence models, and fluid-particle coupling.
"""

from .flow_field import (
    CycloneFlowParams,
    CycloneFlowField,
    WarpFlowParams,
    wp_velocity_at,
    compute_fluid_velocities,
    create_warp_flow_params,
)

# Sub-modules
from . import solvers
from . import turbulence

__all__ = [
    # Analytical flow field
    "CycloneFlowParams",
    "CycloneFlowField",
    "WarpFlowParams",
    "wp_velocity_at",
    "compute_fluid_velocities",
    "create_warp_flow_params",
    # Sub-modules
    "solvers",
    "turbulence",
]
