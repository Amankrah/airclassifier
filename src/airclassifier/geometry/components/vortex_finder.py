"""
Vortex finder component for cyclone air classifier.

The vortex finder is the central tube at the top of the cyclone through
which the clean gas (and fine particles) exits. It extends down into
the cyclone to establish the inner vortex.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import warp as wp

from ..primitives import Tube, TubeParams
from ...utils.constants import PI


@dataclass
class VortexFinderParams:
    """Parameters for the vortex finder."""

    # Tube dimensions
    diameter: float         # [m] Inner diameter of vortex finder
    length: float           # [m] Total length (insertion depth into cyclone)
    wall_thickness: float = 0.005  # [m] Wall thickness

    # Position relative to cyclone
    cyclone_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    protrusion_above: float = 0.1  # [m] How far tube extends above cyclone top

    # Mesh resolution
    resolution_radial: int = 32
    resolution_axial: int = 12

    @property
    def inner_radius(self) -> float:
        """Inner radius of vortex finder."""
        return self.diameter / 2.0

    @property
    def outer_radius(self) -> float:
        """Outer radius including wall."""
        return self.inner_radius + self.wall_thickness

    @property
    def outer_diameter(self) -> float:
        """Outer diameter including wall."""
        return self.diameter + 2.0 * self.wall_thickness

    @property
    def cross_sectional_area(self) -> float:
        """Flow cross-sectional area (inner)."""
        return PI * self.inner_radius ** 2

    @property
    def insertion_depth(self) -> float:
        """How far the vortex finder extends into the cyclone."""
        return self.length - self.protrusion_above


class VortexFinder:
    """
    Vortex finder (gas outlet tube) for cyclone air classifier.

    The vortex finder establishes the boundary between the outer
    (downward-spiraling) vortex and the inner (upward-spiraling) vortex.
    Its diameter and insertion depth are critical design parameters.
    """

    def __init__(self, params: VortexFinderParams):
        """
        Initialize vortex finder.

        Args:
            params: VortexFinderParams defining the geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

        # Create tube component
        # Positioned at cyclone center, extending downward
        tube_center = (
            params.cyclone_center[0],
            params.cyclone_center[1] + params.protrusion_above,
            params.cyclone_center[2]
        )

        self._tube = Tube(TubeParams(
            outer_radius=params.outer_radius,
            inner_radius=params.inner_radius,
            length=params.length,
            center=tube_center,
            axis="y",
            direction=-1,  # Extends downward
            resolution_radial=params.resolution_radial,
            resolution_axial=params.resolution_axial
        ))

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for the vortex finder.

        Returns:
            Tuple of (vertices, indices, normals)
        """
        self._vertices, self._indices, self._normals = self._tube.generate_mesh()
        return self._vertices, self._indices, self._normals

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the vortex finder geometry."""
        if self._vertices is None:
            self.generate_mesh()

        points = wp.array(self._vertices, dtype=wp.vec3, device=device)
        indices = wp.array(self._indices, dtype=wp.int32, device=device)

        return wp.Mesh(points=points, indices=indices)

    def is_inside_tube(self, point: np.ndarray) -> bool:
        """
        Check if a point is inside the vortex finder tube.

        Args:
            point: 3D point to check

        Returns:
            True if point is inside the tube's flow passage
        """
        p = self.params

        # Check radial distance
        dx = point[0] - p.cyclone_center[0]
        dz = point[2] - p.cyclone_center[2]
        r = np.sqrt(dx * dx + dz * dz)

        if r > p.inner_radius:
            return False

        # Check axial position
        y_top = p.cyclone_center[1] + p.protrusion_above
        y_bottom = y_top - p.length

        return y_bottom <= point[1] <= y_top

    def get_outlet_plane(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the plane defining the outlet (top of vortex finder).

        Returns:
            Tuple of (center_point, normal_vector)
        """
        p = self.params
        center = np.array([
            p.cyclone_center[0],
            p.cyclone_center[1] + p.protrusion_above,
            p.cyclone_center[2]
        ])
        normal = np.array([0.0, 1.0, 0.0])  # Pointing up (outward)

        return center, normal

    def get_bottom_edge_y(self) -> float:
        """Get Y-coordinate of the bottom edge of vortex finder."""
        return self.params.cyclone_center[1] + self.params.protrusion_above - self.params.length

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
# WARP SDF FUNCTION FOR VORTEX FINDER
# =============================================================================

@wp.func
def vortex_finder_sdf(
    p: wp.vec3,
    center: wp.vec3,
    inner_radius: float,
    outer_radius: float,
    length: float,
    protrusion: float
) -> float:
    """
    Signed distance function for vortex finder tube.

    Returns negative distance if inside the tube WALL (solid region),
    positive if outside or in the flow passage.

    Args:
        p: Query point
        center: Center of cyclone (top)
        inner_radius: Inner radius of tube
        outer_radius: Outer radius of tube
        length: Total length of tube
        protrusion: How far tube extends above cyclone

    Returns:
        Signed distance to tube wall
    """
    # Local coordinates
    local_x = p[0] - center[0]
    local_y = p[1] - center[1]
    local_z = p[2] - center[2]

    # Radial distance from axis
    r = wp.sqrt(local_x * local_x + local_z * local_z)

    # Y bounds
    y_top = protrusion
    y_bottom = protrusion - length

    # Check if in Y range
    if local_y > y_top:
        # Above tube
        if r <= outer_radius:
            return local_y - y_top
        else:
            return wp.sqrt((r - outer_radius) ** 2.0 + (local_y - y_top) ** 2.0)

    elif local_y < y_bottom:
        # Below tube
        if r <= outer_radius:
            return y_bottom - local_y
        else:
            return wp.sqrt((r - outer_radius) ** 2.0 + (y_bottom - local_y) ** 2.0)

    else:
        # In Y range - check radial position
        if r < inner_radius:
            # Inside flow passage
            return inner_radius - r
        elif r <= outer_radius:
            # Inside wall (negative)
            return -wp.min(r - inner_radius, outer_radius - r)
        else:
            # Outside tube
            return r - outer_radius
