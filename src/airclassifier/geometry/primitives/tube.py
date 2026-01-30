"""
Tube (hollow cylinder) primitive for cyclone geometry.

Used for inlet pipes, vortex finder, and dust outlet.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import warp as wp

from ...utils.constants import PI, TWO_PI


@dataclass
class TubeParams:
    """Parameters defining a tube (hollow cylinder)."""

    outer_radius: float     # [m] Outer radius
    inner_radius: float     # [m] Inner radius (must be < outer_radius)
    length: float           # [m] Tube length
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Center of one end
    axis: str = "y"         # Axis along which tube extends
    direction: int = 1      # 1 = positive axis direction, -1 = negative
    resolution_radial: int = 32   # Number of radial segments
    resolution_axial: int = 8     # Number of axial segments

    def __post_init__(self):
        if self.inner_radius >= self.outer_radius:
            raise ValueError("Inner radius must be less than outer radius")

    @property
    def wall_thickness(self) -> float:
        """Wall thickness."""
        return self.outer_radius - self.inner_radius

    @property
    def outer_diameter(self) -> float:
        """Outer diameter."""
        return 2.0 * self.outer_radius

    @property
    def inner_diameter(self) -> float:
        """Inner diameter (bore)."""
        return 2.0 * self.inner_radius

    @property
    def cross_sectional_area(self) -> float:
        """Cross-sectional area of the flow passage (inner)."""
        return PI * self.inner_radius ** 2

    @property
    def wall_area(self) -> float:
        """Cross-sectional area of the wall material."""
        return PI * (self.outer_radius ** 2 - self.inner_radius ** 2)

    @property
    def inner_surface_area(self) -> float:
        """Inner surface area."""
        return TWO_PI * self.inner_radius * self.length

    @property
    def outer_surface_area(self) -> float:
        """Outer surface area."""
        return TWO_PI * self.outer_radius * self.length


class Tube:
    """
    Tube (hollow cylinder) primitive with mesh generation.

    Used for:
    - Inlet pipe (tangential entry)
    - Vortex finder (central outlet)
    - Dust outlet pipe
    """

    def __init__(self, params: TubeParams):
        """
        Initialize tube.

        Args:
            params: TubeParams defining the tube geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

    def generate_mesh(self, include_inner: bool = True, include_outer: bool = True,
                      include_ends: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triangle mesh for the tube surface.

        Args:
            include_inner: Include inner surface
            include_outer: Include outer surface
            include_ends: Include end annular faces

        Returns:
            Tuple of (vertices, indices, normals) as numpy arrays
        """
        p = self.params
        n_radial = p.resolution_radial
        n_axial = p.resolution_axial

        vertices = []
        normals = []
        indices = []

        def get_position_and_normal(theta: float, t: float, radius: float, is_outer: bool):
            """Get position and normal for a point on tube surface."""
            h = t * p.length * p.direction

            if p.axis == "y":
                x = radius * np.cos(theta) + p.center[0]
                y = h + p.center[1]
                z = radius * np.sin(theta) + p.center[2]
                sign = 1.0 if is_outer else -1.0
                nx, ny, nz = sign * np.cos(theta), 0.0, sign * np.sin(theta)
            elif p.axis == "z":
                x = radius * np.cos(theta) + p.center[0]
                y = radius * np.sin(theta) + p.center[1]
                z = h + p.center[2]
                sign = 1.0 if is_outer else -1.0
                nx, ny, nz = sign * np.cos(theta), sign * np.sin(theta), 0.0
            else:  # axis == "x"
                x = h + p.center[0]
                y = radius * np.cos(theta) + p.center[1]
                z = radius * np.sin(theta) + p.center[2]
                sign = 1.0 if is_outer else -1.0
                nx, ny, nz = 0.0, sign * np.cos(theta), sign * np.sin(theta)

            return [x, y, z], [nx, ny, nz]

        # Outer surface
        if include_outer:
            outer_start = len(vertices)
            for j in range(n_axial + 1):
                t = j / n_axial
                for i in range(n_radial):
                    theta = (i / n_radial) * TWO_PI
                    pos, norm = get_position_and_normal(theta, t, p.outer_radius, True)
                    vertices.append(pos)
                    normals.append(norm)

            for j in range(n_axial):
                for i in range(n_radial):
                    i_next = (i + 1) % n_radial
                    v0 = outer_start + j * n_radial + i
                    v1 = outer_start + j * n_radial + i_next
                    v2 = outer_start + (j + 1) * n_radial + i_next
                    v3 = outer_start + (j + 1) * n_radial + i
                    indices.extend([v0, v1, v2])
                    indices.extend([v0, v2, v3])

        # Inner surface
        if include_inner:
            inner_start = len(vertices)
            for j in range(n_axial + 1):
                t = j / n_axial
                for i in range(n_radial):
                    theta = (i / n_radial) * TWO_PI
                    pos, norm = get_position_and_normal(theta, t, p.inner_radius, False)
                    vertices.append(pos)
                    normals.append(norm)

            for j in range(n_axial):
                for i in range(n_radial):
                    i_next = (i + 1) % n_radial
                    v0 = inner_start + j * n_radial + i
                    v1 = inner_start + j * n_radial + i_next
                    v2 = inner_start + (j + 1) * n_radial + i_next
                    v3 = inner_start + (j + 1) * n_radial + i

                    # Reverse winding for inner surface
                    indices.extend([v0, v2, v1])
                    indices.extend([v0, v3, v2])

        # End faces (annular rings)
        if include_ends:
            # Start end (t=0)
            self._add_annular_face(
                vertices, normals, indices,
                p.center, p.outer_radius, p.inner_radius,
                p.axis, -p.direction, n_radial
            )

            # End end (t=1)
            if p.axis == "y":
                end_center = [p.center[0], p.center[1] + p.length * p.direction, p.center[2]]
            elif p.axis == "z":
                end_center = [p.center[0], p.center[1], p.center[2] + p.length * p.direction]
            else:
                end_center = [p.center[0] + p.length * p.direction, p.center[1], p.center[2]]

            self._add_annular_face(
                vertices, normals, indices,
                end_center, p.outer_radius, p.inner_radius,
                p.axis, p.direction, n_radial
            )

        self._vertices = np.array(vertices, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.int32)
        self._normals = np.array(normals, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def _add_annular_face(self, vertices, normals, indices,
                          center, outer_r, inner_r, axis, direction, n_radial):
        """Add an annular (ring) face to the mesh."""
        start_idx = len(vertices)

        # Normal direction
        if axis == "y":
            normal = [0.0, float(direction), 0.0]
        elif axis == "z":
            normal = [0.0, 0.0, float(direction)]
        else:
            normal = [float(direction), 0.0, 0.0]

        # Outer ring vertices
        for i in range(n_radial):
            theta = (i / n_radial) * TWO_PI
            if axis == "y":
                pos = [outer_r * np.cos(theta) + center[0], center[1], outer_r * np.sin(theta) + center[2]]
            elif axis == "z":
                pos = [outer_r * np.cos(theta) + center[0], outer_r * np.sin(theta) + center[1], center[2]]
            else:
                pos = [center[0], outer_r * np.cos(theta) + center[1], outer_r * np.sin(theta) + center[2]]
            vertices.append(pos)
            normals.append(normal)

        # Inner ring vertices
        for i in range(n_radial):
            theta = (i / n_radial) * TWO_PI
            if axis == "y":
                pos = [inner_r * np.cos(theta) + center[0], center[1], inner_r * np.sin(theta) + center[2]]
            elif axis == "z":
                pos = [inner_r * np.cos(theta) + center[0], inner_r * np.sin(theta) + center[1], center[2]]
            else:
                pos = [center[0], inner_r * np.cos(theta) + center[1], inner_r * np.sin(theta) + center[2]]
            vertices.append(pos)
            normals.append(normal)

        # Triangles connecting outer and inner rings
        for i in range(n_radial):
            i_next = (i + 1) % n_radial
            outer_v0 = start_idx + i
            outer_v1 = start_idx + i_next
            inner_v0 = start_idx + n_radial + i
            inner_v1 = start_idx + n_radial + i_next

            if direction > 0:
                indices.extend([outer_v0, outer_v1, inner_v1])
                indices.extend([outer_v0, inner_v1, inner_v0])
            else:
                indices.extend([outer_v0, inner_v1, outer_v1])
                indices.extend([outer_v0, inner_v0, inner_v1])

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """
        Create a Warp mesh from the tube geometry.

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


@dataclass
class RectangularDuctParams:
    """Parameters for rectangular duct (used for tangential inlet)."""

    width: float            # [m] Width of the duct
    height: float           # [m] Height of the duct
    length: float           # [m] Length of the duct
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Center of inlet end
    direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # Direction duct extends

    @property
    def cross_sectional_area(self) -> float:
        """Cross-sectional area."""
        return self.width * self.height

    @property
    def hydraulic_diameter(self) -> float:
        """Hydraulic diameter for flow calculations."""
        return 4.0 * self.cross_sectional_area / (2.0 * (self.width + self.height))


class RectangularDuct:
    """
    Rectangular duct for tangential inlet.

    The duct is defined by its width, height, length, and direction.
    """

    def __init__(self, params: RectangularDuctParams):
        """
        Initialize rectangular duct.

        Args:
            params: RectangularDuctParams defining the duct geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triangle mesh for the rectangular duct.

        Returns:
            Tuple of (vertices, indices, normals) as numpy arrays
        """
        p = self.params
        w, h, l = p.width / 2.0, p.height / 2.0, p.length

        # Normalize direction
        d = np.array(p.direction)
        d = d / np.linalg.norm(d)

        # Create orthonormal basis
        if abs(d[1]) < 0.9:
            up = np.array([0.0, 1.0, 0.0])
        else:
            up = np.array([1.0, 0.0, 0.0])

        right = np.cross(d, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, d)

        center = np.array(p.center)

        # Define 8 corners of the duct
        corners = [
            center + right * w + up * h,           # 0: front top right
            center - right * w + up * h,           # 1: front top left
            center - right * w - up * h,           # 2: front bottom left
            center + right * w - up * h,           # 3: front bottom right
            center + d * l + right * w + up * h,   # 4: back top right
            center + d * l - right * w + up * h,   # 5: back top left
            center + d * l - right * w - up * h,   # 6: back bottom left
            center + d * l + right * w - up * h,   # 7: back bottom right
        ]

        vertices = [c.tolist() for c in corners]

        # Define faces (6 faces, 2 triangles each)
        faces = [
            # Front face (inlet)
            ([0, 1, 2, 3], -d),
            # Back face (connects to cyclone)
            ([4, 7, 6, 5], d),
            # Top face
            ([0, 4, 5, 1], up),
            # Bottom face
            ([2, 6, 7, 3], -up),
            # Right face
            ([0, 3, 7, 4], right),
            # Left face
            ([1, 5, 6, 2], -right),
        ]

        indices = []
        normals = [[0.0, 0.0, 0.0]] * 8

        for face_verts, normal in faces:
            # Two triangles per face
            indices.extend([face_verts[0], face_verts[1], face_verts[2]])
            indices.extend([face_verts[0], face_verts[2], face_verts[3]])

        # Simple normals (can be improved with per-face normals)
        for i, (face_verts, normal) in enumerate(faces):
            for v in face_verts:
                normals[v] = normal.tolist()

        self._vertices = np.array(vertices, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.int32)
        self._normals = np.array(normals, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the duct geometry."""
        if self._vertices is None:
            self.generate_mesh()

        points = wp.array(self._vertices, dtype=wp.vec3, device=device)
        indices = wp.array(self._indices, dtype=wp.int32, device=device)

        return wp.Mesh(points=points, indices=indices)
