"""
Material Transport Kernels
==========================

Warp kernels for conveyor belt advection of scalar fields (T, M)
along the positive X-axis.

Uses first-order upwind scheme. Courant number must be < 1.
Infeed boundary injects fresh material at the inlet conditions.
"""

# TODO: Import warp when implementing
# import warp as wp


# @wp.kernel
def advect_material():
    """Advect a scalar field along the positive X-axis (conveyor direction).

    First-order upwind scheme.

    See engineering guide section 8.5 for full implementation.
    """
    # TODO: Implement as @wp.kernel
    raise NotImplementedError
