"""
Field Solve Kernels
===================

Warp kernels for the Laplace equation solver:

    div(eps' * grad(phi)) = 0

Phase 2: Jacobi iterative solver on structured grid.
Phase 3: Warp FEM assembly + sparse CG solve.
"""

# TODO: Import warp when implementing
# import warp as wp


# @wp.kernel
def jacobi_iteration():
    """One Jacobi iteration for the Laplace equation.

    phi_new[i,j,k] = weighted average of neighbors / eps' weights.
    """
    # TODO: Implement as @wp.kernel
    raise NotImplementedError


# @wp.kernel
def compute_gradient_sq():
    """Compute |grad(phi)|^2 from the potential field.

    |E|^2 = (dphi/dx)^2 + (dphi/dy)^2 + (dphi/dz)^2

    Uses central differences.
    """
    # TODO: Implement as @wp.kernel
    raise NotImplementedError
