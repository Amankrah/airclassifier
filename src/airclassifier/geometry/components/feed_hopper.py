"""
Feed hopper component for flour/powder storage and discharge.

The feed hopper provides controlled storage and gravity discharge
of flour into the classification system. Design follows mass flow
principles for consistent discharge without ratholing.

Principle:
- Cylindrical section provides main storage volume
- Conical section ensures mass flow discharge
- Cone angle must exceed material's angle of repose + 10-15 deg
"""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import warp as wp

from ...utils.constants import PI, TWO_PI


@dataclass
class FeedHopperParams:
    """Parameters for feed hopper/silo."""

    # Geometry
    top_diameter: float          # [m] Top opening diameter
    bottom_diameter: float       # [m] Bottom discharge diameter
    cylindrical_height: float    # [m] Height of cylindrical section
    conical_height: float        # [m] Height of conical discharge section

    # Optional lid/cover
    has_lid: bool = True         # Include top cover
    lid_height: float = 0.05     # [m] Lid thickness

    # Design parameters
    wall_thickness: float = 0.003  # [m] Wall thickness (3mm default)

    # Position
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Bottom center of discharge

    # Mesh resolution
    resolution_radial: int = 32
    resolution_axial: int = 24

    @property
    def top_radius(self) -> float:
        """Top radius."""
        return self.top_diameter / 2

    @property
    def bottom_radius(self) -> float:
        """Bottom discharge radius."""
        return self.bottom_diameter / 2

    @property
    def total_height(self) -> float:
        """Total hopper height."""
        h = self.conical_height + self.cylindrical_height
        if self.has_lid:
            h += self.lid_height
        return h

    @property
    def cone_half_angle(self) -> float:
        """Half-angle of the conical section in radians."""
        dr = self.top_radius - self.bottom_radius
        return np.arctan2(dr, self.conical_height)

    @property
    def cone_half_angle_degrees(self) -> float:
        """Half-angle in degrees (for checking mass flow criteria)."""
        return np.degrees(self.cone_half_angle)

    @property
    def cylindrical_volume(self) -> float:
        """Volume of cylindrical section [m^3]."""
        return PI * self.top_radius ** 2 * self.cylindrical_height

    @property
    def conical_volume(self) -> float:
        """Volume of conical section [m^3]."""
        r1, r2 = self.top_radius, self.bottom_radius
        h = self.conical_height
        return (PI * h / 3.0) * (r1 ** 2 + r1 * r2 + r2 ** 2)

    @property
    def total_volume(self) -> float:
        """Total internal volume [m^3]."""
        return self.cylindrical_volume + self.conical_volume

    def capacity_kg(self, bulk_density: float = 500.0) -> float:
        """
        Calculate capacity in kg for given bulk density.

        Args:
            bulk_density: Material bulk density [kg/m^3] (default 500 for flour)

        Returns:
            Capacity in kg
        """
        return self.total_volume * bulk_density


class FeedHopper:
    """
    Feed hopper for powder/flour storage.

    Components:
    - Cylindrical storage section
    - Conical discharge section (mass flow design)
    - Optional lid/cover

    Coordinate system:
    - Origin at center of bottom discharge opening
    - Y-axis pointing upward
    """

    def __init__(self, params: FeedHopperParams):
        """
        Initialize feed hopper.

        Args:
            params: FeedHopperParams defining the geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for the feed hopper.

        Returns:
            Tuple of (vertices, indices, normals)
        """
        p = self.params
        vertices = []
        indices = []
        normals = []

        n_radial = p.resolution_radial
        n_axial = p.resolution_axial

        # Generate conical section
        y_cone_bottom = p.center[1]
        y_cone_top = p.center[1] + p.conical_height

        n_cone = n_axial // 2
        for i in range(n_cone + 1):
            t = i / n_cone
            y = y_cone_bottom + t * p.conical_height
            r = p.bottom_radius + (p.top_radius - p.bottom_radius) * t

            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + r * np.cos(theta)
                z = p.center[2] + r * np.sin(theta)

                vertices.append([x, y, z])

                # Normal for conical surface (angled outward)
                dr = p.top_radius - p.bottom_radius
                slant = np.sqrt(p.conical_height ** 2 + dr ** 2)
                n_y = dr / slant
                n_r = p.conical_height / slant
                normals.append([n_r * np.cos(theta), n_y, n_r * np.sin(theta)])

        # Generate triangles for cone
        for i in range(n_cone):
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = i * n_radial + j
                v1 = i * n_radial + j_next
                v2 = (i + 1) * n_radial + j_next
                v3 = (i + 1) * n_radial + j

                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

        # Generate cylindrical section
        y_cyl_bottom = y_cone_top
        y_cyl_top = y_cyl_bottom + p.cylindrical_height

        cyl_start_idx = len(vertices)
        n_cyl = n_axial // 2
        for i in range(n_cyl + 1):
            t = i / n_cyl
            y = y_cyl_bottom + t * p.cylindrical_height
            r = p.top_radius

            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + r * np.cos(theta)
                z = p.center[2] + r * np.sin(theta)

                vertices.append([x, y, z])
                normals.append([np.cos(theta), 0.0, np.sin(theta)])

        # Generate triangles for cylinder
        for i in range(n_cyl):
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = cyl_start_idx + i * n_radial + j
                v1 = cyl_start_idx + i * n_radial + j_next
                v2 = cyl_start_idx + (i + 1) * n_radial + j_next
                v3 = cyl_start_idx + (i + 1) * n_radial + j

                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

        # Generate lid if requested
        if p.has_lid:
            self._add_lid(vertices, indices, normals, y_cyl_top)

        # Generate bottom discharge ring
        self._add_discharge_ring(vertices, indices, normals, y_cone_bottom)

        self._vertices = np.array(vertices, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.int32)
        self._normals = np.array(normals, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def _add_lid(self, vertices: List, indices: List, normals: List, y_top: float):
        """Add lid/cover to hopper."""
        p = self.params
        n_radial = p.resolution_radial

        # Top surface of lid
        lid_start_idx = len(vertices)
        y_lid_top = y_top + p.lid_height

        # Center vertex
        vertices.append([p.center[0], y_lid_top, p.center[2]])
        normals.append([0.0, 1.0, 0.0])

        # Edge vertices
        for j in range(n_radial):
            theta = (j / n_radial) * TWO_PI
            x = p.center[0] + p.top_radius * np.cos(theta)
            z = p.center[2] + p.top_radius * np.sin(theta)
            vertices.append([x, y_lid_top, z])
            normals.append([0.0, 1.0, 0.0])

        # Triangles for lid top
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            indices.extend([lid_start_idx, lid_start_idx + 1 + j, lid_start_idx + 1 + j_next])

    def _add_discharge_ring(self, vertices: List, indices: List, normals: List, y_bottom: float):
        """Add discharge ring at bottom."""
        p = self.params
        n_radial = p.resolution_radial // 2

        ring_start_idx = len(vertices)
        ring_height = p.bottom_diameter * 0.2

        # Inner and outer rings
        for i in range(2):
            y = y_bottom - i * ring_height
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + p.bottom_radius * np.cos(theta)
                z = p.center[2] + p.bottom_radius * np.sin(theta)
                vertices.append([x, y, z])
                normals.append([np.cos(theta), 0.0, np.sin(theta)])

        # Triangles for discharge ring
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = ring_start_idx + j
            v1 = ring_start_idx + j_next
            v2 = ring_start_idx + n_radial + j_next
            v3 = ring_start_idx + n_radial + j

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

    def is_mass_flow_design(self, material_angle_of_repose: float = 35.0) -> bool:
        """
        Check if hopper meets mass flow criteria.

        Args:
            material_angle_of_repose: Material's angle of repose in degrees

        Returns:
            True if design should achieve mass flow
        """
        # Mass flow requires cone angle > angle of repose + 10-15 deg
        required_angle = material_angle_of_repose + 12
        return self.params.cone_half_angle_degrees > required_angle

    def get_discharge_center(self) -> Tuple[float, float, float]:
        """Get center position of discharge opening."""
        p = self.params
        return (p.center[0], p.center[1], p.center[2])

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the hopper geometry."""
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


def create_standard_feed_hopper(
    capacity_kg: float = 500,
    bulk_density: float = 500,
    discharge_diameter: float = 0.15
) -> FeedHopper:
    """
    Create a standard feed hopper sized for given capacity.

    Args:
        capacity_kg: Design capacity [kg]
        bulk_density: Material bulk density [kg/m^3]
        discharge_diameter: Discharge opening diameter [m]

    Returns:
        FeedHopper instance
    """
    # Calculate required volume
    required_volume = capacity_kg / bulk_density

    # Standard proportions
    # Assume cylinder height = 1.5 * diameter
    # Cone angle = 45 degrees for mass flow with most powders

    # Estimate top diameter from volume
    # V_total = V_cyl + V_cone
    # With aspect ratio ~ 2:1 (height:diameter)
    # Approximate: V ~ 0.8 * pi * r^2 * h with h = 2 * r
    # V ~ 1.6 * pi * r^3

    r_estimate = (required_volume / (1.6 * PI)) ** (1/3)
    top_diameter = 2 * r_estimate

    # Ensure minimum size
    top_diameter = max(top_diameter, discharge_diameter * 3)

    params = FeedHopperParams(
        top_diameter=top_diameter,
        bottom_diameter=discharge_diameter,
        cylindrical_height=top_diameter * 0.75,
        conical_height=top_diameter * 0.6,
        has_lid=True,
    )

    return FeedHopper(params)
