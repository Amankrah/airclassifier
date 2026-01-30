"""
Dust outlet (underflow) component for cyclone air classifier.

The dust outlet is located at the bottom of the cone section where
coarse/heavy particles are collected after being separated from the
gas stream.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import warp as wp

from ..primitives import Tube, TubeParams
from ...utils.constants import PI


@dataclass
class DustOutletParams:
    """Parameters for the dust outlet."""

    # Outlet dimensions
    diameter: float         # [m] Inner diameter of outlet
    length: float           # [m] Length of outlet pipe
    wall_thickness: float = 0.005  # [m] Wall thickness

    # Position (connected to bottom of cone)
    cone_bottom_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Optional: apex cone at transition
    include_apex_cone: bool = False
    apex_cone_height: float = 0.05  # [m] Height of transition cone

    # Mesh resolution
    resolution_radial: int = 24
    resolution_axial: int = 8

    @property
    def inner_radius(self) -> float:
        """Inner radius of outlet."""
        return self.diameter / 2.0

    @property
    def outer_radius(self) -> float:
        """Outer radius including wall."""
        return self.inner_radius + self.wall_thickness

    @property
    def cross_sectional_area(self) -> float:
        """Flow cross-sectional area."""
        return PI * self.inner_radius ** 2


class DustOutlet:
    """
    Dust outlet (underflow) for collecting separated coarse particles.

    The dust outlet is positioned at the apex of the cone section.
    Particles that reach this region have been successfully separated
    from the gas stream by centrifugal action.
    """

    def __init__(self, params: DustOutletParams):
        """
        Initialize dust outlet.

        Args:
            params: DustOutletParams defining the geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

        # Create tube component extending downward from cone bottom
        self._tube = Tube(TubeParams(
            outer_radius=params.outer_radius,
            inner_radius=params.inner_radius,
            length=params.length,
            center=params.cone_bottom_center,
            axis="y",
            direction=-1,  # Extends downward
            resolution_radial=params.resolution_radial,
            resolution_axial=params.resolution_axial
        ))

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for the dust outlet.

        Returns:
            Tuple of (vertices, indices, normals)
        """
        self._vertices, self._indices, self._normals = self._tube.generate_mesh()
        return self._vertices, self._indices, self._normals

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the dust outlet geometry."""
        if self._vertices is None:
            self.generate_mesh()

        points = wp.array(self._vertices, dtype=wp.vec3, device=device)
        indices = wp.array(self._indices, dtype=wp.int32, device=device)

        return wp.Mesh(points=points, indices=indices)

    def is_particle_collected(self, point: np.ndarray) -> bool:
        """
        Check if a particle at the given position should be considered collected.

        Args:
            point: 3D position of particle

        Returns:
            True if particle has entered the dust outlet region
        """
        p = self.params

        # Check radial distance from outlet axis
        dx = point[0] - p.cone_bottom_center[0]
        dz = point[2] - p.cone_bottom_center[2]
        r = np.sqrt(dx * dx + dz * dz)

        # Must be within outlet radius
        if r > p.inner_radius:
            return False

        # Must be at or below the cone bottom
        return point[1] <= p.cone_bottom_center[1]

    def get_outlet_plane(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the plane defining the outlet (bottom of dust outlet pipe).

        Returns:
            Tuple of (center_point, normal_vector)
        """
        p = self.params
        center = np.array([
            p.cone_bottom_center[0],
            p.cone_bottom_center[1] - p.length,
            p.cone_bottom_center[2]
        ])
        normal = np.array([0.0, -1.0, 0.0])  # Pointing down (outward)

        return center, normal

    def get_collection_plane_y(self) -> float:
        """
        Get Y-coordinate of the collection plane.

        Particles below this Y are considered collected.
        """
        return self.params.cone_bottom_center[1]

    @property
    def tube(self) -> Tube:
        """Get the tube component."""
        return self._tube

    @property
    def vertices(self) -> np.ndarray:
        """Get mesh vertices."""
        if self._vertices is None:
            self.generate_mesh()
        return self._vertices

    @property
    def indices(self) -> np.ndarray:
        """Get mesh triangle indices."""
        if self._indices is None:
            self.generate_mesh()
        return self._indices


# =============================================================================
# WARP KERNEL FOR PARTICLE COLLECTION CHECK
# =============================================================================

@wp.func
def is_in_dust_outlet(
    p: wp.vec3,
    outlet_center: wp.vec3,
    outlet_radius: float
) -> bool:
    """
    Check if a particle position is in the dust outlet region.

    Args:
        p: Particle position
        outlet_center: Center of dust outlet (at cone bottom)
        outlet_radius: Inner radius of dust outlet

    Returns:
        True if particle is in the collection region
    """
    # Check if below outlet plane
    if p[1] > outlet_center[1]:
        return False

    # Check radial distance
    dx = p[0] - outlet_center[0]
    dz = p[2] - outlet_center[2]
    r_sq = dx * dx + dz * dz

    return r_sq <= outlet_radius * outlet_radius
