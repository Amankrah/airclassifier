"""
Tangential inlet component for cyclone air classifier.

The inlet introduces air and particles tangentially into the cyclone,
creating the swirling motion essential for particle separation.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import warp as wp

from ..primitives import RectangularDuct, RectangularDuctParams
from ...utils.constants import PI


@dataclass
class InletParams:
    """Parameters for the tangential inlet."""

    # Inlet dimensions
    width: float            # [m] Width of inlet (tangential direction)
    height: float           # [m] Height of inlet (axial direction)
    length: float           # [m] Length of inlet duct

    # Position relative to cyclone
    cyclone_diameter: float # [m] Diameter of cyclone cylinder
    inlet_top_offset: float # [m] Distance from top of cyclone to top of inlet

    # Cyclone center position
    cyclone_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Entry angle (0 = purely tangential, positive = downward spiral)
    entry_angle: float = 0.0  # [radians]

    # Angular position of inlet (0 = +X direction)
    angular_position: float = 0.0  # [radians]

    @property
    def cross_sectional_area(self) -> float:
        """Inlet cross-sectional area."""
        return self.width * self.height

    @property
    def hydraulic_diameter(self) -> float:
        """Hydraulic diameter for flow calculations."""
        return 4.0 * self.cross_sectional_area / (2.0 * (self.width + self.height))

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (height/width)."""
        return self.height / self.width

    @property
    def area(self) -> float:
        """Inlet area (alias for cross_sectional_area)."""
        return self.cross_sectional_area


class TangentialInlet:
    """
    Tangential inlet for introducing air and particles into the cyclone.

    The inlet is positioned tangentially to the cyclone body, typically
    at the top of the cylindrical section.
    """

    def __init__(self, params: InletParams):
        """
        Initialize tangential inlet.

        Args:
            params: InletParams defining the inlet geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

        # Calculate inlet position and orientation
        self._calculate_position()

    def _calculate_position(self):
        """Calculate the inlet position and direction vectors."""
        p = self.params
        r = p.cyclone_diameter / 2.0

        # Angular position
        theta = p.angular_position

        # Inlet center position (on cyclone surface)
        self.surface_point = np.array([
            p.cyclone_center[0] + r * np.cos(theta),
            p.cyclone_center[1] - p.inlet_top_offset - p.height / 2.0,
            p.cyclone_center[2] + r * np.sin(theta)
        ])

        # Tangent direction (perpendicular to radius, in XZ plane)
        # Positive tangent = counterclockwise when viewed from above
        self.tangent = np.array([-np.sin(theta), 0.0, np.cos(theta)])

        # Radial direction (outward from cyclone axis)
        self.radial = np.array([np.cos(theta), 0.0, np.sin(theta)])

        # Inlet direction (pointing inward, with possible downward angle)
        inlet_horizontal = -self.radial
        inlet_vertical = np.array([0.0, -np.sin(p.entry_angle), 0.0])
        self.inlet_direction = inlet_horizontal * np.cos(p.entry_angle) + inlet_vertical
        self.inlet_direction = self.inlet_direction / np.linalg.norm(self.inlet_direction)

        # Inlet start point (outer end of inlet duct)
        self.inlet_start = self.surface_point - self.inlet_direction * p.length

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for the inlet duct.

        Returns:
            Tuple of (vertices, indices, normals)
        """
        p = self.params

        # Build the rectangular duct
        vertices = []
        indices = []
        normals = []

        # Get orthonormal basis for the duct cross-section
        forward = self.inlet_direction
        up = np.array([0.0, 1.0, 0.0])

        # Handle case where inlet direction is nearly vertical
        if abs(np.dot(forward, up)) > 0.9:
            up = np.array([1.0, 0.0, 0.0])

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # Half dimensions
        hw = p.width / 2.0
        hh = p.height / 2.0

        # Define corners at inlet start (outer) and surface (inner)
        start = self.inlet_start
        end = self.surface_point

        corners_start = [
            start + right * hw + up * hh,   # 0: top right
            start - right * hw + up * hh,   # 1: top left
            start - right * hw - up * hh,   # 2: bottom left
            start + right * hw - up * hh,   # 3: bottom right
        ]

        corners_end = [
            end + right * hw + up * hh,     # 4: top right
            end - right * hw + up * hh,     # 5: top left
            end - right * hw - up * hh,     # 6: bottom left
            end + right * hw - up * hh,     # 7: bottom right
        ]

        vertices = corners_start + corners_end

        # Define faces
        # Outer face (inlet opening)
        indices.extend([0, 2, 1])
        indices.extend([0, 3, 2])

        # Top face
        indices.extend([0, 1, 5])
        indices.extend([0, 5, 4])

        # Bottom face
        indices.extend([2, 3, 7])
        indices.extend([2, 7, 6])

        # Left face
        indices.extend([1, 2, 6])
        indices.extend([1, 6, 5])

        # Right face
        indices.extend([0, 4, 7])
        indices.extend([0, 7, 3])

        # Inner face is open (connects to cyclone)

        # Simple normals
        normals = [
            (right + up).tolist(),
            (-right + up).tolist(),
            (-right - up).tolist(),
            (right - up).tolist(),
            (right + up).tolist(),
            (-right + up).tolist(),
            (-right - up).tolist(),
            (right - up).tolist(),
        ]

        self._vertices = np.array(vertices, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.int32)
        self._normals = np.array(normals, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def get_inlet_velocity_direction(self) -> np.ndarray:
        """
        Get the direction of inlet velocity.

        For tangential inlet, this is primarily in the tangent direction
        with possible axial component.

        Returns:
            Unit vector in the direction of inlet flow
        """
        p = self.params

        # Combine tangential and possible axial component
        v = self.tangent.copy()
        if abs(p.entry_angle) > 1e-6:
            v[1] = -np.sin(p.entry_angle)
            v = v / np.linalg.norm(v)

        return v

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the inlet geometry."""
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
    def entry_point(self) -> np.ndarray:
        """Get the point where flow enters the cyclone."""
        return self.surface_point.copy()

    @property
    def outer_point(self) -> np.ndarray:
        """Get the outer opening of the inlet."""
        return self.inlet_start.copy()
