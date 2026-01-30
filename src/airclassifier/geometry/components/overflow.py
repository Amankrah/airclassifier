"""
Overflow (gas outlet) component for cyclone air classifier.

The overflow represents the exit path for the clean gas stream carrying
fine particles through the vortex finder. This module handles the
boundary conditions and tracking of particles exiting through the top.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import warp as wp

from ...utils.constants import PI


@dataclass
class OverflowParams:
    """Parameters for the overflow (gas outlet) region."""

    # Vortex finder parameters (overflow goes through vortex finder)
    vortex_finder_diameter: float   # [m] Inner diameter of vortex finder
    vortex_finder_top_y: float      # [m] Y-coordinate of top of vortex finder

    # Cyclone center
    cyclone_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Exit duct (optional, after vortex finder)
    exit_duct_length: float = 0.2   # [m] Length of exit duct above vortex finder
    exit_duct_diameter: float = None  # [m] If None, same as vortex finder

    @property
    def vortex_finder_radius(self) -> float:
        """Radius of vortex finder."""
        return self.vortex_finder_diameter / 2.0

    @property
    def exit_area(self) -> float:
        """Cross-sectional area at exit."""
        r = self.vortex_finder_radius
        return PI * r * r

    def __post_init__(self):
        if self.exit_duct_diameter is None:
            self.exit_duct_diameter = self.vortex_finder_diameter


class Overflow:
    """
    Overflow region where clean gas and fine particles exit.

    This component manages:
    - Detection of particles exiting through the vortex finder
    - Boundary conditions for gas flow at the outlet
    - Tracking of overflow particle statistics
    """

    def __init__(self, params: OverflowParams):
        """
        Initialize overflow region.

        Args:
            params: OverflowParams defining the outlet geometry
        """
        self.params = params

        # Statistics tracking
        self._particles_exited = 0
        self._particle_sizes_exited: List[float] = []

    def is_particle_exiting(self, position: np.ndarray, velocity: np.ndarray) -> bool:
        """
        Check if a particle is exiting through the overflow.

        Args:
            position: 3D position of particle
            velocity: 3D velocity of particle

        Returns:
            True if particle is exiting through overflow
        """
        p = self.params

        # Must be above the vortex finder top
        if position[1] < p.vortex_finder_top_y:
            return False

        # Must be within vortex finder radius
        dx = position[0] - p.cyclone_center[0]
        dz = position[2] - p.cyclone_center[2]
        r = np.sqrt(dx * dx + dz * dz)

        if r > p.vortex_finder_radius:
            return False

        # Must be moving upward
        return velocity[1] > 0

    def record_exited_particle(self, diameter: float):
        """
        Record a particle that has exited through overflow.

        Args:
            diameter: Particle diameter [m]
        """
        self._particles_exited += 1
        self._particle_sizes_exited.append(diameter)

    def reset_statistics(self):
        """Reset the overflow statistics."""
        self._particles_exited = 0
        self._particle_sizes_exited = []

    def get_exit_plane(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the plane defining the overflow exit.

        Returns:
            Tuple of (center_point, normal_vector)
        """
        p = self.params
        center = np.array([
            p.cyclone_center[0],
            p.vortex_finder_top_y,
            p.cyclone_center[2]
        ])
        normal = np.array([0.0, 1.0, 0.0])  # Pointing up (outward)

        return center, normal

    def get_exit_boundary_condition(self) -> dict:
        """
        Get boundary condition parameters for the overflow exit.

        Returns:
            Dictionary with boundary condition parameters
        """
        return {
            "type": "pressure_outlet",
            "pressure": 0.0,  # Gauge pressure (relative to ambient)
            "center": self.params.cyclone_center,
            "radius": self.params.vortex_finder_radius,
            "y_position": self.params.vortex_finder_top_y,
            "normal": np.array([0.0, 1.0, 0.0])
        }

    @property
    def particles_exited(self) -> int:
        """Number of particles that have exited through overflow."""
        return self._particles_exited

    @property
    def particle_sizes_exited(self) -> List[float]:
        """List of diameters of particles that exited through overflow."""
        return self._particle_sizes_exited.copy()

    @property
    def mean_exit_diameter(self) -> Optional[float]:
        """Mean diameter of particles that exited (None if no particles)."""
        if not self._particle_sizes_exited:
            return None
        return np.mean(self._particle_sizes_exited)


# =============================================================================
# WARP KERNEL FOR OVERFLOW CHECK
# =============================================================================

@wp.func
def is_in_overflow(
    p: wp.vec3,
    v: wp.vec3,
    center: wp.vec3,
    vf_radius: float,
    vf_top_y: float
) -> bool:
    """
    Check if a particle is exiting through the overflow.

    Args:
        p: Particle position
        v: Particle velocity
        center: Cyclone center (x, y, z)
        vf_radius: Vortex finder inner radius
        vf_top_y: Y-coordinate of vortex finder top

    Returns:
        True if particle is exiting through overflow
    """
    # Must be above vortex finder top
    if p[1] < vf_top_y:
        return False

    # Must be within vortex finder radius
    dx = p[0] - center[0]
    dz = p[2] - center[2]
    r_sq = dx * dx + dz * dz

    if r_sq > vf_radius * vf_radius:
        return False

    # Must be moving upward
    return v[1] > 0.0


@wp.kernel
def check_overflow_particles(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    is_active: wp.array(dtype=wp.int32),
    exit_flags: wp.array(dtype=wp.int32),
    center: wp.vec3,
    vf_radius: float,
    vf_top_y: float
):
    """
    Kernel to check which particles are exiting through overflow.

    Args:
        positions: Particle positions
        velocities: Particle velocities
        is_active: Active particle flags (1 = active, 0 = inactive)
        exit_flags: Output flags (1 = exiting, 0 = not exiting)
        center: Cyclone center
        vf_radius: Vortex finder radius
        vf_top_y: Vortex finder top Y
    """
    tid = wp.tid()

    if is_active[tid] == 0:
        exit_flags[tid] = 0
        return

    if is_in_overflow(positions[tid], velocities[tid], center, vf_radius, vf_top_y):
        exit_flags[tid] = 1
    else:
        exit_flags[tid] = 0
