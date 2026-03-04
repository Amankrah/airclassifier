"""
Cylinder primitive for cyclone geometry.

Provides mesh generation and signed distance field (SDF) functions
for cylindrical sections of the cyclone.
"""

from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING
import numpy as np

from ...utils.constants import PI, TWO_PI

# Lazy warp import to avoid issues with PyInstaller bundling
# Warp requires source code access for JIT compilation
_wp = None

def _get_warp():
    """Lazy load warp module."""
    global _wp
    if _wp is None:
        import warp as wp
        _wp = wp
    return _wp

if TYPE_CHECKING:
    import warp as wp


@dataclass
class CylinderParams:
    """Parameters defining a cylinder."""

    radius: float           # [m] Cylinder radius
    height: float           # [m] Cylinder height
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Center of base
    axis: str = "y"         # Axis along which cylinder extends ("x", "y", or "z")
    resolution_radial: int = 32   # Number of radial segments
    resolution_axial: int = 16    # Number of axial segments

    @property
    def diameter(self) -> float:
        """Cylinder diameter."""
        return 2.0 * self.radius

    @property
    def volume(self) -> float:
        """Cylinder volume."""
        return PI * self.radius ** 2 * self.height

    @property
    def lateral_surface_area(self) -> float:
        """Lateral (curved) surface area."""
        return TWO_PI * self.radius * self.height

    @property
    def total_surface_area(self) -> float:
        """Total surface area including end caps."""
        end_cap_area = PI * self.radius ** 2
        return self.lateral_surface_area + 2 * end_cap_area


class Cylinder:
    """
    Cylinder primitive with mesh generation and SDF computation.

    The cylinder is defined by its radius, height, and axis orientation.
    By default, the cylinder is aligned with the Y-axis with the base
    centered at the origin.
    """

    def __init__(self, params: CylinderParams):
        """
        Initialize cylinder.

        Args:
            params: CylinderParams defining the cylinder geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triangle mesh for the cylinder surface.

        Returns:
            Tuple of (vertices, indices, normals) as numpy arrays
        """
        p = self.params
        n_radial = p.resolution_radial
        n_axial = p.resolution_axial

        vertices = []
        normals = []
        indices = []

        # Generate lateral surface vertices
        for j in range(n_axial + 1):
            t = j / n_axial
            h = t * p.height

            for i in range(n_radial):
                theta = (i / n_radial) * TWO_PI

                # Position on cylinder surface
                if p.axis == "y":
                    x = p.radius * np.cos(theta) + p.center[0]
                    y = h + p.center[1]
                    z = p.radius * np.sin(theta) + p.center[2]
                    nx, ny, nz = np.cos(theta), 0.0, np.sin(theta)
                elif p.axis == "z":
                    x = p.radius * np.cos(theta) + p.center[0]
                    y = p.radius * np.sin(theta) + p.center[1]
                    z = h + p.center[2]
                    nx, ny, nz = np.cos(theta), np.sin(theta), 0.0
                else:  # axis == "x"
                    x = h + p.center[0]
                    y = p.radius * np.cos(theta) + p.center[1]
                    z = p.radius * np.sin(theta) + p.center[2]
                    nx, ny, nz = 0.0, np.cos(theta), np.sin(theta)

                vertices.append([x, y, z])
                normals.append([nx, ny, nz])

        # Generate lateral surface triangles
        for j in range(n_axial):
            for i in range(n_radial):
                i_next = (i + 1) % n_radial
                v0 = j * n_radial + i
                v1 = j * n_radial + i_next
                v2 = (j + 1) * n_radial + i_next
                v3 = (j + 1) * n_radial + i

                # Two triangles per quad
                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

        # Generate end caps
        base_vertex_start = len(vertices)

        # Bottom cap center
        if p.axis == "y":
            cap_center_bottom = [p.center[0], p.center[1], p.center[2]]
            cap_normal_bottom = [0.0, -1.0, 0.0]
            cap_center_top = [p.center[0], p.center[1] + p.height, p.center[2]]
            cap_normal_top = [0.0, 1.0, 0.0]
        elif p.axis == "z":
            cap_center_bottom = [p.center[0], p.center[1], p.center[2]]
            cap_normal_bottom = [0.0, 0.0, -1.0]
            cap_center_top = [p.center[0], p.center[1], p.center[2] + p.height]
            cap_normal_top = [0.0, 0.0, 1.0]
        else:  # axis == "x"
            cap_center_bottom = [p.center[0], p.center[1], p.center[2]]
            cap_normal_bottom = [-1.0, 0.0, 0.0]
            cap_center_top = [p.center[0] + p.height, p.center[1], p.center[2]]
            cap_normal_top = [1.0, 0.0, 0.0]

        # Bottom cap
        vertices.append(cap_center_bottom)
        normals.append(cap_normal_bottom)
        bottom_center_idx = len(vertices) - 1

        for i in range(n_radial):
            theta = (i / n_radial) * TWO_PI
            if p.axis == "y":
                x = p.radius * np.cos(theta) + p.center[0]
                y = p.center[1]
                z = p.radius * np.sin(theta) + p.center[2]
            elif p.axis == "z":
                x = p.radius * np.cos(theta) + p.center[0]
                y = p.radius * np.sin(theta) + p.center[1]
                z = p.center[2]
            else:
                x = p.center[0]
                y = p.radius * np.cos(theta) + p.center[1]
                z = p.radius * np.sin(theta) + p.center[2]

            vertices.append([x, y, z])
            normals.append(cap_normal_bottom)

        for i in range(n_radial):
            i_next = (i + 1) % n_radial
            v0 = bottom_center_idx
            v1 = bottom_center_idx + 1 + i_next
            v2 = bottom_center_idx + 1 + i
            indices.extend([v0, v1, v2])

        # Top cap
        vertices.append(cap_center_top)
        normals.append(cap_normal_top)
        top_center_idx = len(vertices) - 1

        for i in range(n_radial):
            theta = (i / n_radial) * TWO_PI
            if p.axis == "y":
                x = p.radius * np.cos(theta) + p.center[0]
                y = p.center[1] + p.height
                z = p.radius * np.sin(theta) + p.center[2]
            elif p.axis == "z":
                x = p.radius * np.cos(theta) + p.center[0]
                y = p.radius * np.sin(theta) + p.center[1]
                z = p.center[2] + p.height
            else:
                x = p.center[0] + p.height
                y = p.radius * np.cos(theta) + p.center[1]
                z = p.radius * np.sin(theta) + p.center[2]

            vertices.append([x, y, z])
            normals.append(cap_normal_top)

        for i in range(n_radial):
            i_next = (i + 1) % n_radial
            v0 = top_center_idx
            v1 = top_center_idx + 1 + i
            v2 = top_center_idx + 1 + i_next
            indices.extend([v0, v1, v2])

        self._vertices = np.array(vertices, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.int32)
        self._normals = np.array(normals, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def to_warp_mesh(self, device: str = "cuda"):
        """
        Create a Warp mesh from the cylinder geometry.

        Args:
            device: Device to create mesh on ("cuda" or "cpu")

        Returns:
            wp.Mesh object
        """
        wp = _get_warp()
        if self._vertices is None:
            self.generate_mesh()

        points = wp.array(self._vertices, dtype=wp.vec3, device=device)
        indices = wp.array(self._indices, dtype=wp.int32, device=device)

        return wp.Mesh(points=points, indices=indices)

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

    @property
    def normals(self) -> np.ndarray:
        """Get vertex normals."""
        if self._normals is None:
            self.generate_mesh()
        return self._normals

    @property
    def volume(self) -> float:
        """Cylinder volume (proxy to params.volume)."""
        return self.params.volume

    @property
    def lateral_surface_area(self) -> float:
        """Lateral (curved) surface area (proxy to params.lateral_surface_area)."""
        return self.params.lateral_surface_area


# =============================================================================
# WARP SDF FUNCTIONS FOR CYLINDER
# =============================================================================
# NOTE: These functions are created lazily to avoid PyInstaller issues
# with Warp JIT compilation at import time.

_cylinder_sdf = None
_cylinder_sdf_hollow = None


def get_cylinder_sdf():
    """Get the cylinder SDF warp function (lazy loaded)."""
    global _cylinder_sdf
    if _cylinder_sdf is None:
        wp = _get_warp()

        @wp.func
        def cylinder_sdf(
            p: wp.vec3,
            center: wp.vec3,
            radius: float,
            height: float
        ) -> float:
            """
            Signed distance function for a Y-axis aligned cylinder.
            """
            # Vector from base center to point
            d = wp.vec3(p[0] - center[0], 0.0, p[2] - center[2])

            # Radial distance from axis
            r_dist = wp.length(d) - radius

            # Axial distance from cylinder bounds
            y_local = p[1] - center[1]
            y_dist = wp.max(-y_local, y_local - height)

            # Combine radial and axial distances
            if r_dist > 0.0 and y_dist > 0.0:
                # Outside both radially and axially (corner case)
                return wp.sqrt(r_dist * r_dist + y_dist * y_dist)
            else:
                # Inside at least one dimension
                return wp.max(r_dist, y_dist)

        _cylinder_sdf = cylinder_sdf
    return _cylinder_sdf


def get_cylinder_sdf_hollow():
    """Get the hollow cylinder SDF warp function (lazy loaded)."""
    global _cylinder_sdf_hollow
    if _cylinder_sdf_hollow is None:
        wp = _get_warp()

        @wp.func
        def cylinder_sdf_hollow(
            p: wp.vec3,
            center: wp.vec3,
            outer_radius: float,
            inner_radius: float,
            height: float
        ) -> float:
            """
            Signed distance function for a hollow Y-axis aligned cylinder (tube).
            """
            # Vector from base center to point
            d = wp.vec3(p[0] - center[0], 0.0, p[2] - center[2])
            r = wp.length(d)

            # Radial distance from wall
            r_dist_outer = r - outer_radius
            r_dist_inner = inner_radius - r
            r_dist = wp.max(r_dist_outer, r_dist_inner)

            # Axial distance from cylinder bounds
            y_local = p[1] - center[1]
            y_dist = wp.max(-y_local, y_local - height)

            return wp.max(r_dist, y_dist)

        _cylinder_sdf_hollow = cylinder_sdf_hollow
    return _cylinder_sdf_hollow


# Backward-compatible aliases (functions return lazy-loaded warp funcs)
def cylinder_sdf(*args, **kwargs):
    """Backward-compatible wrapper for cylinder_sdf."""
    return get_cylinder_sdf()(*args, **kwargs)


def cylinder_sdf_hollow(*args, **kwargs):
    """Backward-compatible wrapper for cylinder_sdf_hollow."""
    return get_cylinder_sdf_hollow()(*args, **kwargs)
