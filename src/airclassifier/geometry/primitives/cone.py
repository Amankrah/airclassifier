"""
Cone/frustum primitive for cyclone geometry.

Provides mesh generation and signed distance field (SDF) functions
for conical sections of the cyclone.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import warp as wp

from ...utils.constants import PI, TWO_PI


@dataclass
class ConeParams:
    """Parameters defining a cone or frustum (truncated cone)."""

    top_radius: float       # [m] Radius at top (larger end for cyclone cone)
    bottom_radius: float    # [m] Radius at bottom (smaller end, can be 0 for full cone)
    height: float           # [m] Height of the cone
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Center of top base
    axis: str = "y"         # Axis along which cone extends (pointing down from top)
    resolution_radial: int = 32   # Number of radial segments
    resolution_axial: int = 16    # Number of axial segments

    @property
    def is_frustum(self) -> bool:
        """Check if this is a frustum (truncated cone) vs full cone."""
        return self.bottom_radius > 1e-10

    @property
    def slant_height(self) -> float:
        """Slant height of the cone."""
        dr = self.top_radius - self.bottom_radius
        return np.sqrt(self.height ** 2 + dr ** 2)

    @property
    def half_angle(self) -> float:
        """Half-angle of the cone in radians."""
        dr = self.top_radius - self.bottom_radius
        return np.arctan2(dr, self.height)

    @property
    def volume(self) -> float:
        """Volume of the frustum/cone."""
        r1, r2, h = self.top_radius, self.bottom_radius, self.height
        return (PI * h / 3.0) * (r1 ** 2 + r1 * r2 + r2 ** 2)

    @property
    def lateral_surface_area(self) -> float:
        """Lateral (curved) surface area."""
        r1, r2 = self.top_radius, self.bottom_radius
        s = self.slant_height
        return PI * (r1 + r2) * s


class Cone:
    """
    Cone/frustum primitive with mesh generation and SDF computation.

    For cyclone geometry, the cone typically has the larger radius at top
    (connecting to cylinder) and smaller radius at bottom (dust outlet).
    """

    def __init__(self, params: ConeParams):
        """
        Initialize cone.

        Args:
            params: ConeParams defining the cone geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triangle mesh for the cone surface.

        Returns:
            Tuple of (vertices, indices, normals) as numpy arrays
        """
        p = self.params
        n_radial = p.resolution_radial
        n_axial = p.resolution_axial

        vertices = []
        normals = []
        indices = []

        # Calculate normal direction (perpendicular to slant surface)
        dr = p.top_radius - p.bottom_radius
        slant = np.sqrt(p.height ** 2 + dr ** 2)
        # Normal components: radial = h/slant, axial = dr/slant
        if slant > 1e-10:
            n_radial_component = p.height / slant
            n_axial_component = dr / slant
        else:
            n_radial_component = 1.0
            n_axial_component = 0.0

        # Generate lateral surface vertices
        for j in range(n_axial + 1):
            t = j / n_axial
            # Interpolate radius from top to bottom
            r = p.top_radius * (1.0 - t) + p.bottom_radius * t
            # Height goes from 0 (top) to -height (bottom)
            h = -t * p.height

            for i in range(n_radial):
                theta = (i / n_radial) * TWO_PI

                if p.axis == "y":
                    x = r * np.cos(theta) + p.center[0]
                    y = h + p.center[1]
                    z = r * np.sin(theta) + p.center[2]
                    # Normal pointing outward and slightly upward
                    nx = n_radial_component * np.cos(theta)
                    ny = n_axial_component
                    nz = n_radial_component * np.sin(theta)
                elif p.axis == "z":
                    x = r * np.cos(theta) + p.center[0]
                    y = r * np.sin(theta) + p.center[1]
                    z = h + p.center[2]
                    nx = n_radial_component * np.cos(theta)
                    ny = n_radial_component * np.sin(theta)
                    nz = n_axial_component
                else:  # axis == "x"
                    x = h + p.center[0]
                    y = r * np.cos(theta) + p.center[1]
                    z = r * np.sin(theta) + p.center[2]
                    nx = n_axial_component
                    ny = n_radial_component * np.cos(theta)
                    nz = n_radial_component * np.sin(theta)

                vertices.append([x, y, z])
                # Normalize normal vector
                n_len = np.sqrt(nx * nx + ny * ny + nz * nz)
                if n_len > 1e-10:
                    normals.append([nx / n_len, ny / n_len, nz / n_len])
                else:
                    normals.append([0.0, 1.0, 0.0])

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

        # Generate top cap (larger end)
        if p.axis == "y":
            cap_center_top = [p.center[0], p.center[1], p.center[2]]
            cap_normal_top = [0.0, 1.0, 0.0]
        elif p.axis == "z":
            cap_center_top = [p.center[0], p.center[1], p.center[2]]
            cap_normal_top = [0.0, 0.0, 1.0]
        else:
            cap_center_top = [p.center[0], p.center[1], p.center[2]]
            cap_normal_top = [1.0, 0.0, 0.0]

        vertices.append(cap_center_top)
        normals.append(cap_normal_top)
        top_center_idx = len(vertices) - 1

        for i in range(n_radial):
            theta = (i / n_radial) * TWO_PI
            if p.axis == "y":
                x = p.top_radius * np.cos(theta) + p.center[0]
                y = p.center[1]
                z = p.top_radius * np.sin(theta) + p.center[2]
            elif p.axis == "z":
                x = p.top_radius * np.cos(theta) + p.center[0]
                y = p.top_radius * np.sin(theta) + p.center[1]
                z = p.center[2]
            else:
                x = p.center[0]
                y = p.top_radius * np.cos(theta) + p.center[1]
                z = p.top_radius * np.sin(theta) + p.center[2]

            vertices.append([x, y, z])
            normals.append(cap_normal_top)

        for i in range(n_radial):
            i_next = (i + 1) % n_radial
            v0 = top_center_idx
            v1 = top_center_idx + 1 + i
            v2 = top_center_idx + 1 + i_next
            indices.extend([v0, v1, v2])

        # Generate bottom cap if frustum
        if p.is_frustum:
            if p.axis == "y":
                cap_center_bottom = [p.center[0], p.center[1] - p.height, p.center[2]]
                cap_normal_bottom = [0.0, -1.0, 0.0]
            elif p.axis == "z":
                cap_center_bottom = [p.center[0], p.center[1], p.center[2] - p.height]
                cap_normal_bottom = [0.0, 0.0, -1.0]
            else:
                cap_center_bottom = [p.center[0] - p.height, p.center[1], p.center[2]]
                cap_normal_bottom = [-1.0, 0.0, 0.0]

            vertices.append(cap_center_bottom)
            normals.append(cap_normal_bottom)
            bottom_center_idx = len(vertices) - 1

            for i in range(n_radial):
                theta = (i / n_radial) * TWO_PI
                if p.axis == "y":
                    x = p.bottom_radius * np.cos(theta) + p.center[0]
                    y = p.center[1] - p.height
                    z = p.bottom_radius * np.sin(theta) + p.center[2]
                elif p.axis == "z":
                    x = p.bottom_radius * np.cos(theta) + p.center[0]
                    y = p.bottom_radius * np.sin(theta) + p.center[1]
                    z = p.center[2] - p.height
                else:
                    x = p.center[0] - p.height
                    y = p.bottom_radius * np.cos(theta) + p.center[1]
                    z = p.bottom_radius * np.sin(theta) + p.center[2]

                vertices.append([x, y, z])
                normals.append(cap_normal_bottom)

            for i in range(n_radial):
                i_next = (i + 1) % n_radial
                v0 = bottom_center_idx
                v1 = bottom_center_idx + 1 + i_next
                v2 = bottom_center_idx + 1 + i
                indices.extend([v0, v1, v2])

        self._vertices = np.array(vertices, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.int32)
        self._normals = np.array(normals, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """
        Create a Warp mesh from the cone geometry.

        Args:
            device: Device to create mesh on ("cuda" or "cpu")

        Returns:
            wp.Mesh object
        """
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


# =============================================================================
# WARP SDF FUNCTIONS FOR CONE/FRUSTUM
# =============================================================================

@wp.func
def cone_sdf(
    p: wp.vec3,
    apex: wp.vec3,
    half_angle: float,
    height: float
) -> float:
    """
    Signed distance function for a Y-axis aligned cone with apex at top.

    Args:
        p: Query point
        apex: Position of cone apex (tip)
        half_angle: Half-angle of cone in radians
        height: Height of cone from apex to base

    Returns:
        Signed distance (negative inside, positive outside)
    """
    # Transform point to cone's local space
    q = wp.vec3(p[0] - apex[0], apex[1] - p[1], p[2] - apex[2])

    # Radial distance from axis
    r = wp.sqrt(q[0] * q[0] + q[2] * q[2])

    # Cone parameters
    sin_a = wp.sin(half_angle)
    cos_a = wp.cos(half_angle)

    # Distance to cone surface
    # Project onto cone surface direction
    d = wp.vec2(r, q[1])
    cone_dir = wp.vec2(sin_a, cos_a)

    # Distance along cone normal
    dist_to_surface = wp.dot(d, wp.vec2(cos_a, -sin_a))

    # Clamp to valid cone region (between apex and base)
    if q[1] < 0.0:
        # Above apex
        return wp.length(wp.vec3(q[0], q[1], q[2]))
    elif q[1] > height:
        # Below base
        base_radius = height * wp.tan(half_angle)
        if r < base_radius:
            return q[1] - height
        else:
            edge_dist = r - base_radius
            return wp.sqrt(edge_dist * edge_dist + (q[1] - height) * (q[1] - height))
    else:
        return dist_to_surface


@wp.func
def frustum_sdf(
    p: wp.vec3,
    center: wp.vec3,
    top_radius: float,
    bottom_radius: float,
    height: float
) -> float:
    """
    Signed distance function for a Y-axis aligned frustum (truncated cone).

    The frustum has its top (larger end) at center, extending downward.

    Args:
        p: Query point
        center: Center of top base
        top_radius: Radius at top (larger end)
        bottom_radius: Radius at bottom (smaller end)
        height: Height of frustum

    Returns:
        Signed distance (negative inside, positive outside)
    """
    # Local coordinates relative to top center
    local_x = p[0] - center[0]
    local_y = center[1] - p[1]  # Positive going down
    local_z = p[2] - center[2]

    # Radial distance from axis
    r = wp.sqrt(local_x * local_x + local_z * local_z)

    # Interpolated radius at this height
    if local_y < 0.0:
        # Above top
        t = 0.0
    elif local_y > height:
        # Below bottom
        t = 1.0
    else:
        t = local_y / height

    r_at_height = top_radius * (1.0 - t) + bottom_radius * t

    # Radial distance from surface
    r_dist = r - r_at_height

    # Axial bounds
    y_dist_top = -local_y
    y_dist_bottom = local_y - height

    # Combine distances
    if local_y < 0.0:
        # Above top cap
        if r <= top_radius:
            return -local_y
        else:
            return wp.sqrt((r - top_radius) ** 2.0 + local_y ** 2.0)
    elif local_y > height:
        # Below bottom cap
        if r <= bottom_radius:
            return local_y - height
        else:
            return wp.sqrt((r - bottom_radius) ** 2.0 + (local_y - height) ** 2.0)
    else:
        # In height range - compute distance to slant surface
        dr = top_radius - bottom_radius
        slant = wp.sqrt(height * height + dr * dr)
        # Normal direction: (height, dr) / slant
        # Distance = (r - r_at_height) * height / slant
        # But we need the true perpendicular distance to the slant
        dist_to_slant = r_dist * height / slant

        return dist_to_slant
