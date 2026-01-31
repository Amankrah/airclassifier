"""
Screw feeder component for controlled powder dosing.

The screw feeder (auger conveyor) provides controlled, adjustable
feed rates of powder into the classification system. Variable speed
drive allows precise control of feed rate.

Principle:
- Rotating helical screw moves material along trough
- Feed rate controlled by RPM and fill level
- Variable pitch option for uniform withdrawal from hopper
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import warp as wp

from ...utils.constants import PI, TWO_PI


@dataclass
class ScrewFeederParams:
    """Parameters for screw feeder / auger conveyor."""

    # Screw geometry
    screw_diameter: float        # [m] Screw/auger outer diameter
    shaft_diameter: float        # [m] Central shaft diameter
    screw_pitch: float           # [m] Pitch of screw flights (axial distance per revolution)
    flight_thickness: float      # [m] Thickness of helical flight

    # Trough geometry
    trough_length: float         # [m] Length of trough
    trough_clearance: float      # [m] Gap between screw OD and trough

    # Inlet/outlet
    inlet_length: float          # [m] Inlet opening length (along trough)
    inlet_width: float           # [m] Inlet opening width
    outlet_diameter: float       # [m] Outlet diameter

    # Variable pitch option
    variable_pitch: bool = False  # Use variable pitch for uniform withdrawal
    pitch_start_ratio: float = 0.7  # Starting pitch ratio if variable

    # Operating parameters
    rpm: float = 30.0            # [rpm] Rotation speed
    fill_level: float = 0.30    # Fill level (0-1, typically 0.15-0.45)

    # Position
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Center of inlet
    axis: str = "x"              # Screw axis direction

    # Mesh resolution
    resolution_radial: int = 16
    resolution_axial: int = 32
    helix_segments: int = 36     # Segments per revolution of helix

    @property
    def screw_radius(self) -> float:
        """Screw outer radius."""
        return self.screw_diameter / 2

    @property
    def shaft_radius(self) -> float:
        """Shaft radius."""
        return self.shaft_diameter / 2

    @property
    def trough_radius(self) -> float:
        """Trough inner radius (U-shaped cross-section bottom)."""
        return self.screw_radius + self.trough_clearance

    @property
    def num_flights(self) -> float:
        """Number of complete screw flights."""
        return self.trough_length / self.screw_pitch

    @property
    def volumetric_capacity(self) -> float:
        """Theoretical volumetric capacity [m^3/h] at design RPM and fill."""
        # Q = pi/4 * (D^2 - d^2) * p * n * fill * 60
        D = self.screw_diameter
        d = self.shaft_diameter
        return PI / 4 * (D ** 2 - d ** 2) * self.screw_pitch * self.rpm * self.fill_level * 60

    def capacity_kg_h(self, bulk_density: float = 500.0) -> float:
        """Mass capacity [kg/h] for given bulk density."""
        return self.volumetric_capacity * bulk_density


class ScrewFeeder:
    """
    Screw feeder / auger conveyor for powder dosing.

    Components:
    - U-shaped trough
    - Helical screw with central shaft
    - Inlet hopper connection
    - Outlet discharge

    Coordinate system:
    - Origin at center of inlet
    - Screw axis along specified direction (default X)
    """

    def __init__(self, params: ScrewFeederParams):
        """
        Initialize screw feeder.

        Args:
            params: ScrewFeederParams defining the geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for the screw feeder.

        Returns:
            Tuple of (vertices, indices, normals)
        """
        p = self.params
        vertices = []
        indices = []
        normals = []

        # Generate trough (U-shaped channel)
        self._generate_trough(vertices, indices, normals)

        # Generate screw (shaft + helical flights)
        self._generate_screw(vertices, indices, normals)

        # Generate inlet hopper section
        self._generate_inlet(vertices, indices, normals)

        # Generate outlet
        self._generate_outlet(vertices, indices, normals)

        self._vertices = np.array(vertices, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.int32)
        self._normals = np.array(normals, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def _generate_trough(self, vertices: List, indices: List, normals: List):
        """Generate U-shaped trough."""
        p = self.params
        n_radial = p.resolution_radial
        n_axial = p.resolution_axial

        start_idx = len(vertices)
        r = p.trough_radius
        wall_thickness = 0.003  # 3mm walls

        # Trough is a half-cylinder (bottom) plus side walls
        # Generate outer surface of half-cylinder
        for i in range(n_axial + 1):
            t_axial = i / n_axial
            axial_pos = t_axial * p.trough_length

            for j in range(n_radial // 2 + 1):
                # Half circle from -90 to +90 degrees (bottom half)
                theta = PI / 2 + (j / (n_radial // 2)) * PI

                if p.axis == "x":
                    x = p.center[0] + axial_pos
                    y = p.center[1] + (r + wall_thickness) * np.sin(theta)
                    z = p.center[2] + (r + wall_thickness) * np.cos(theta)
                    nx, ny, nz = 0.0, np.sin(theta), np.cos(theta)
                elif p.axis == "y":
                    x = p.center[0] + (r + wall_thickness) * np.cos(theta)
                    y = p.center[1] + axial_pos
                    z = p.center[2] + (r + wall_thickness) * np.sin(theta)
                    nx, ny, nz = np.cos(theta), 0.0, np.sin(theta)
                else:  # z-axis
                    x = p.center[0] + (r + wall_thickness) * np.cos(theta)
                    y = p.center[1] + (r + wall_thickness) * np.sin(theta)
                    z = p.center[2] + axial_pos
                    nx, ny, nz = np.cos(theta), np.sin(theta), 0.0

                vertices.append([x, y, z])
                normals.append([nx, ny, nz])

        # Generate triangles for trough
        n_circ = n_radial // 2 + 1
        for i in range(n_axial):
            for j in range(n_circ - 1):
                v0 = start_idx + i * n_circ + j
                v1 = start_idx + i * n_circ + j + 1
                v2 = start_idx + (i + 1) * n_circ + j + 1
                v3 = start_idx + (i + 1) * n_circ + j

                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

        # Add side walls (vertical extensions above the half-cylinder)
        wall_height = p.screw_radius * 0.5
        self._add_trough_walls(vertices, indices, normals, wall_height)

    def _add_trough_walls(self, vertices: List, indices: List, normals: List,
                          wall_height: float):
        """Add vertical side walls to trough."""
        p = self.params
        n_axial = p.resolution_axial

        r = p.trough_radius
        wall_thickness = 0.003

        for side in [-1, 1]:
            start_idx = len(vertices)

            for i in range(n_axial + 1):
                t_axial = i / n_axial
                axial_pos = t_axial * p.trough_length

                # Bottom of wall (at trough edge)
                # Top of wall
                for h_idx, height in enumerate([0, wall_height]):
                    if p.axis == "x":
                        x = p.center[0] + axial_pos
                        y = p.center[1] + height
                        z = p.center[2] + side * (r + wall_thickness)
                        nx, ny, nz = 0.0, 0.0, side
                    elif p.axis == "y":
                        x = p.center[0] + side * (r + wall_thickness)
                        y = p.center[1] + axial_pos
                        z = p.center[2] + height
                        nx, ny, nz = side, 0.0, 0.0
                    else:
                        x = p.center[0] + side * (r + wall_thickness)
                        y = p.center[1] + height
                        z = p.center[2] + axial_pos
                        nx, ny, nz = side, 0.0, 0.0

                    vertices.append([x, y, z])
                    normals.append([nx, ny, nz])

            # Triangles for this wall
            for i in range(n_axial):
                v0 = start_idx + i * 2
                v1 = start_idx + i * 2 + 1
                v2 = start_idx + (i + 1) * 2 + 1
                v3 = start_idx + (i + 1) * 2

                if side > 0:
                    indices.extend([v0, v1, v2])
                    indices.extend([v0, v2, v3])
                else:
                    indices.extend([v0, v2, v1])
                    indices.extend([v0, v3, v2])

    def _generate_screw(self, vertices: List, indices: List, normals: List):
        """Generate helical screw with shaft."""
        p = self.params
        n_radial = p.resolution_radial

        # Central shaft
        shaft_start = len(vertices)
        for i in range(2):
            axial_pos = i * p.trough_length

            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI

                if p.axis == "x":
                    x = p.center[0] + axial_pos
                    y = p.center[1] + p.shaft_radius * np.sin(theta)
                    z = p.center[2] + p.shaft_radius * np.cos(theta)
                    nx, ny, nz = 0.0, np.sin(theta), np.cos(theta)
                elif p.axis == "y":
                    x = p.center[0] + p.shaft_radius * np.cos(theta)
                    y = p.center[1] + axial_pos
                    z = p.center[2] + p.shaft_radius * np.sin(theta)
                    nx, ny, nz = np.cos(theta), 0.0, np.sin(theta)
                else:
                    x = p.center[0] + p.shaft_radius * np.cos(theta)
                    y = p.center[1] + p.shaft_radius * np.sin(theta)
                    z = p.center[2] + axial_pos
                    nx, ny, nz = np.cos(theta), np.sin(theta), 0.0

                vertices.append([x, y, z])
                normals.append([nx, ny, nz])

        # Shaft triangles
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = shaft_start + j
            v1 = shaft_start + j_next
            v2 = shaft_start + n_radial + j_next
            v3 = shaft_start + n_radial + j

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

        # Helical flights
        self._generate_helical_flights(vertices, indices, normals)

    def _generate_helical_flights(self, vertices: List, indices: List, normals: List):
        """Generate helical screw flights."""
        p = self.params

        num_turns = p.num_flights
        total_segments = int(num_turns * p.helix_segments)

        flight_start = len(vertices)

        for i in range(total_segments + 1):
            t = i / total_segments
            axial_pos = t * p.trough_length

            # Calculate pitch at this position (for variable pitch)
            if p.variable_pitch:
                # Pitch increases along length
                local_pitch_ratio = p.pitch_start_ratio + (1 - p.pitch_start_ratio) * t
            else:
                local_pitch_ratio = 1.0

            theta = t * num_turns * TWO_PI

            # Inner edge (at shaft)
            r_inner = p.shaft_radius + 0.002
            # Outer edge
            r_outer = p.screw_radius

            for r_idx, r in enumerate([r_inner, r_outer]):
                if p.axis == "x":
                    x = p.center[0] + axial_pos
                    y = p.center[1] + r * np.sin(theta)
                    z = p.center[2] + r * np.cos(theta)
                    # Normal perpendicular to helix
                    nx = -local_pitch_ratio * p.screw_pitch / (TWO_PI * r)
                    ny = np.cos(theta)
                    nz = -np.sin(theta)
                elif p.axis == "y":
                    x = p.center[0] + r * np.cos(theta)
                    y = p.center[1] + axial_pos
                    z = p.center[2] + r * np.sin(theta)
                    nx = -np.sin(theta)
                    ny = -local_pitch_ratio * p.screw_pitch / (TWO_PI * r)
                    nz = np.cos(theta)
                else:
                    x = p.center[0] + r * np.cos(theta)
                    y = p.center[1] + r * np.sin(theta)
                    z = p.center[2] + axial_pos
                    nx = -np.sin(theta)
                    ny = np.cos(theta)
                    nz = -local_pitch_ratio * p.screw_pitch / (TWO_PI * r)

                # Normalize
                n_len = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
                if n_len > 0:
                    nx, ny, nz = nx / n_len, ny / n_len, nz / n_len

                vertices.append([x, y, z])
                normals.append([nx, ny, nz])

        # Triangles for helix
        for i in range(total_segments):
            v0 = flight_start + i * 2
            v1 = flight_start + i * 2 + 1
            v2 = flight_start + (i + 1) * 2 + 1
            v3 = flight_start + (i + 1) * 2

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

    def _generate_inlet(self, vertices: List, indices: List, normals: List):
        """Generate inlet hopper connection."""
        p = self.params

        start_idx = len(vertices)
        inlet_height = p.inlet_width * 0.5

        # Rectangular inlet opening
        hw = p.inlet_width / 2
        hl = p.inlet_length / 2

        if p.axis == "x":
            # Inlet at start, pointing up (+Y)
            x_center = p.center[0] + hl
            y_base = p.center[1] + p.trough_radius
            z_center = p.center[2]

            corners = [
                [x_center - hl, y_base, z_center - hw],
                [x_center + hl, y_base, z_center - hw],
                [x_center + hl, y_base, z_center + hw],
                [x_center - hl, y_base, z_center + hw],
                [x_center - hl, y_base + inlet_height, z_center - hw],
                [x_center + hl, y_base + inlet_height, z_center - hw],
                [x_center + hl, y_base + inlet_height, z_center + hw],
                [x_center - hl, y_base + inlet_height, z_center + hw],
            ]
        else:
            # Simplified for other axes
            corners = [[p.center[0], p.center[1], p.center[2]] for _ in range(8)]

        for corner in corners:
            vertices.append(corner)
            normals.append([0.0, 1.0, 0.0])

        # Side faces of inlet
        # Front
        indices.extend([start_idx, start_idx + 4, start_idx + 5])
        indices.extend([start_idx, start_idx + 5, start_idx + 1])
        # Back
        indices.extend([start_idx + 2, start_idx + 6, start_idx + 7])
        indices.extend([start_idx + 2, start_idx + 7, start_idx + 3])
        # Left
        indices.extend([start_idx + 3, start_idx + 7, start_idx + 4])
        indices.extend([start_idx + 3, start_idx + 4, start_idx])
        # Right
        indices.extend([start_idx + 1, start_idx + 5, start_idx + 6])
        indices.extend([start_idx + 1, start_idx + 6, start_idx + 2])

    def _generate_outlet(self, vertices: List, indices: List, normals: List):
        """Generate outlet discharge."""
        p = self.params
        n_radial = p.resolution_radial // 2

        start_idx = len(vertices)
        r = p.outlet_diameter / 2
        outlet_length = p.outlet_diameter

        # Outlet at end of trough, pointing down
        if p.axis == "x":
            x_center = p.center[0] + p.trough_length - p.outlet_diameter
            y_center = p.center[1] - p.trough_radius
            z_center = p.center[2]

            for i in range(2):
                y = y_center - i * outlet_length
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x = x_center + r * np.cos(theta)
                    z = z_center + r * np.sin(theta)
                    vertices.append([x, y, z])
                    normals.append([np.cos(theta), 0.0, np.sin(theta)])
        else:
            # Simplified
            for _ in range(n_radial * 2):
                vertices.append([p.center[0], p.center[1], p.center[2]])
                normals.append([0.0, -1.0, 0.0])

        # Triangles
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = start_idx + j
            v1 = start_idx + j_next
            v2 = start_idx + n_radial + j_next
            v3 = start_idx + n_radial + j

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

    def get_feed_rate(self, rpm: float = None, bulk_density: float = 500.0) -> float:
        """
        Calculate feed rate for given RPM.

        Args:
            rpm: Rotation speed (uses params.rpm if None)
            bulk_density: Material bulk density [kg/m^3]

        Returns:
            Feed rate [kg/h]
        """
        if rpm is None:
            rpm = self.params.rpm

        p = self.params
        D = p.screw_diameter
        d = p.shaft_diameter

        vol_rate = PI / 4 * (D ** 2 - d ** 2) * p.screw_pitch * rpm * p.fill_level * 60
        return vol_rate * bulk_density

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the screw feeder geometry."""
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


def create_standard_screw_feeder(
    screw_diameter: float = 0.10,
    feed_rate_kg_h: float = 500,
    bulk_density: float = 500
) -> ScrewFeeder:
    """
    Create a standard screw feeder sized for given feed rate.

    Args:
        screw_diameter: Screw diameter [m]
        feed_rate_kg_h: Target feed rate [kg/h]
        bulk_density: Material bulk density [kg/m^3]

    Returns:
        ScrewFeeder instance
    """
    # Standard proportions
    shaft_diameter = screw_diameter * 0.3
    screw_pitch = screw_diameter * 0.8  # Standard pitch

    # Calculate length for reasonable fill level and RPM
    rpm = 30
    fill_level = 0.30

    # Q = pi/4 * (D^2 - d^2) * p * n * fill * 60 * rho
    D = screw_diameter
    d = shaft_diameter
    vol_rate_needed = feed_rate_kg_h / bulk_density  # m3/h

    # Length doesn't directly affect capacity, but we size for ~3 pitches
    trough_length = screw_pitch * 3

    params = ScrewFeederParams(
        screw_diameter=screw_diameter,
        shaft_diameter=shaft_diameter,
        screw_pitch=screw_pitch,
        flight_thickness=0.003,
        trough_length=trough_length,
        trough_clearance=0.003,
        inlet_length=screw_diameter * 1.5,
        inlet_width=screw_diameter * 1.2,
        outlet_diameter=screw_diameter * 0.8,
        rpm=rpm,
        fill_level=fill_level,
    )

    return ScrewFeeder(params)
