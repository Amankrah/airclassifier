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
    field_solve         Laplace solver kernels (Jacobi / CG)
"""
