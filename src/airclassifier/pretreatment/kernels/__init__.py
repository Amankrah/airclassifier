"""
Warp GPU Kernels
================

NVIDIA Warp JIT-compiled kernels for the pretreatment simulation.
All kernels operate on pre-allocated 3D arrays and are CUDA-graph
compatible (no allocations inside the kernel body).

Kernels:
    dielectric_heating  P_v computation, loss factor update
    heat_transfer       Conduction with variable k, convection BC, RF source
    drying              Moisture diffusion, evaporation
    transport           Material advection on conveyor (upwind scheme)
                        + ConveyorDriveController (motor, VFD ramp, kinematics)
    field_solve         Laplace solver kernels (Jacobi / CG)
"""

from .transport import (                        # noqa: F401
    ConveyorDriveController,
    ConveyorDriveState,
    rotate_mesh_around_z_axis,
)
