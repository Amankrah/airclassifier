"""
Rotary airlock valve component for pressure sealing and metering.

The rotary airlock provides a seal between pressure zones while
allowing continuous material flow. Essential for feeding material
into pneumatic conveying systems and maintaining system pressure.

Principle:
- Rotating vanes create pockets that transport material
- Seal maintained by small clearance between vanes and housing
- Volumetric metering based on pocket size and rotation speed
"""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import warp as wp

from ...utils.constants import PI, TWO_PI


@dataclass
class RotaryAirlockParams:
    """Parameters for rotary airlock valve."""

    # Rotor geometry
    rotor_diameter: float        # [m] Rotor diameter
    rotor_length: float          # [m] Rotor length (axial width)
    num_vanes: int               # Number of vanes (typically 6-10)
    vane_thickness: float        # [m] Vane thickness
    vane_tip_clearance: float    # [m] Gap between vane tip and housing

    # Housing
    housing_thickness: float = 0.010  # [m] Housing wall thickness

    # Connections
    inlet_width: float = None    # [m] Inlet opening width (default = rotor_length)
    inlet_length: float = None   # [m] Inlet opening length (along circumference)
    outlet_width: float = None   # [m] Outlet opening width
    outlet_length: float = None  # [m] Outlet opening length

    # Operating parameters
    rpm: float = 20.0            # [rpm] Rotation speed

    # Position and orientation
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Center of rotor
    axis: str = "z"              # Rotation axis

    # Mesh resolution
    resolution_radial: int = 24
    resolution_axial: int = 8

    def __post_init__(self):
        # Set defaults based on rotor dimensions
        if self.inlet_width is None:
            self.inlet_width = self.rotor_length * 0.9
        if self.inlet_length is None:
            self.inlet_length = self.rotor_diameter * 0.4
        if self.outlet_width is None:
            self.outlet_width = self.rotor_length * 0.9
        if self.outlet_length is None:
            self.outlet_length = self.rotor_diameter * 0.4

    @property
    def rotor_radius(self) -> float:
        """Rotor radius."""
        return self.rotor_diameter / 2

    @property
    def housing_inner_radius(self) -> float:
        """Housing inner radius."""
        return self.rotor_radius + self.vane_tip_clearance

    @property
    def housing_outer_radius(self) -> float:
        """Housing outer radius."""
        return self.housing_inner_radius + self.housing_thickness

    @property
    def pocket_angle(self) -> float:
        """Angular span of each pocket [radians]."""
        return TWO_PI / self.num_vanes

    @property
    def pocket_volume(self) -> float:
        """Volume of single pocket [m^3]."""
        # Approximate as sector minus hub
        r_outer = self.rotor_radius - self.vane_thickness / 2
        r_inner = self.rotor_radius * 0.3  # Hub radius estimate
        sector_area = 0.5 * self.pocket_angle * (r_outer ** 2 - r_inner ** 2)
        return sector_area * self.rotor_length

    @property
    def volumetric_capacity(self) -> float:
        """Volumetric capacity [m^3/h] at design RPM."""
        pockets_per_hour = self.rpm * 60 * self.num_vanes
        # Account for ~60% fill efficiency
        return self.pocket_volume * pockets_per_hour * 0.6

    def capacity_kg_h(self, bulk_density: float = 500.0) -> float:
        """Mass capacity [kg/h] for given bulk density."""
        return self.volumetric_capacity * bulk_density


class RotaryAirlock:
    """
    Rotary airlock valve for pressure sealing and metering.

    Components:
    - Cylindrical housing
    - Rotor with radial vanes
    - Inlet opening (top)
    - Outlet opening (bottom)
    - End plates

    Coordinate system:
    - Origin at rotor center
    - Rotation axis along specified axis (default Z)
    """

    def __init__(self, params: RotaryAirlockParams):
        """
        Initialize rotary airlock.

        Args:
            params: RotaryAirlockParams defining the geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for the rotary airlock.

        Returns:
            Tuple of (vertices, indices, normals)
        """
        p = self.params
        vertices = []
        indices = []
        normals = []

        # Generate housing (outer cylinder)
        self._generate_housing(vertices, indices, normals)

        # Generate rotor with vanes
        self._generate_rotor(vertices, indices, normals)

        # Generate end plates
        self._generate_end_plates(vertices, indices, normals)

        # Generate inlet and outlet flanges
        self._generate_inlet_outlet(vertices, indices, normals)

        self._vertices = np.array(vertices, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.int32)
        self._normals = np.array(normals, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def _generate_housing(self, vertices: List, indices: List, normals: List):
        """Generate cylindrical housing."""
        p = self.params
        n_radial = p.resolution_radial
        n_axial = p.resolution_axial

        start_idx = len(vertices)
        r = p.housing_outer_radius
        half_length = p.rotor_length / 2 + p.housing_thickness

        # Generate housing cylinder (outer surface)
        for i in range(n_axial + 1):
            t = i / n_axial
            if p.axis == "z":
                z = p.center[2] - half_length + t * 2 * half_length
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x = p.center[0] + r * np.cos(theta)
                    y = p.center[1] + r * np.sin(theta)
                    vertices.append([x, y, z])
                    normals.append([np.cos(theta), np.sin(theta), 0.0])
            elif p.axis == "y":
                y = p.center[1] - half_length + t * 2 * half_length
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x = p.center[0] + r * np.cos(theta)
                    z = p.center[2] + r * np.sin(theta)
                    vertices.append([x, y, z])
                    normals.append([np.cos(theta), 0.0, np.sin(theta)])
            else:  # x-axis
                x = p.center[0] - half_length + t * 2 * half_length
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    y = p.center[1] + r * np.cos(theta)
                    z = p.center[2] + r * np.sin(theta)
                    vertices.append([x, y, z])
                    normals.append([0.0, np.cos(theta), np.sin(theta)])

        # Generate triangles
        for i in range(n_axial):
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = start_idx + i * n_radial + j
                v1 = start_idx + i * n_radial + j_next
                v2 = start_idx + (i + 1) * n_radial + j_next
                v3 = start_idx + (i + 1) * n_radial + j

                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

    def _generate_rotor(self, vertices: List, indices: List, normals: List):
        """Generate rotor with vanes."""
        p = self.params
        n_radial = p.resolution_radial

        start_idx = len(vertices)

        # Hub (inner cylinder)
        hub_radius = p.rotor_radius * 0.25
        half_length = p.rotor_length / 2

        # Hub cylinder
        for i in range(2):
            t = i
            if p.axis == "z":
                z = p.center[2] - half_length + t * p.rotor_length
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x = p.center[0] + hub_radius * np.cos(theta)
                    y = p.center[1] + hub_radius * np.sin(theta)
                    vertices.append([x, y, z])
                    normals.append([np.cos(theta), np.sin(theta), 0.0])
            elif p.axis == "y":
                y = p.center[1] - half_length + t * p.rotor_length
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x = p.center[0] + hub_radius * np.cos(theta)
                    z = p.center[2] + hub_radius * np.sin(theta)
                    vertices.append([x, y, z])
                    normals.append([np.cos(theta), 0.0, np.sin(theta)])
            else:
                x = p.center[0] - half_length + t * p.rotor_length
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    y = p.center[1] + hub_radius * np.cos(theta)
                    z = p.center[2] + hub_radius * np.sin(theta)
                    vertices.append([x, y, z])
                    normals.append([0.0, np.cos(theta), np.sin(theta)])

        # Hub triangles
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = start_idx + j
            v1 = start_idx + j_next
            v2 = start_idx + n_radial + j_next
            v3 = start_idx + n_radial + j

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

        # Generate vanes
        vane_start = len(vertices)
        vane_outer_radius = p.rotor_radius - p.vane_tip_clearance * 0.5

        for v in range(p.num_vanes):
            vane_angle = v * p.pocket_angle

            # Each vane is a thin rectangular plate
            for side in [-1, 1]:
                angle = vane_angle + side * p.vane_thickness / (2 * vane_outer_radius)

                for r_pos in [hub_radius, vane_outer_radius]:
                    for axial in [-half_length, half_length]:
                        if p.axis == "z":
                            x = p.center[0] + r_pos * np.cos(angle)
                            y = p.center[1] + r_pos * np.sin(angle)
                            z = p.center[2] + axial
                            nx = np.cos(angle + side * PI / 2)
                            ny = np.sin(angle + side * PI / 2)
                            nz = 0.0
                        elif p.axis == "y":
                            x = p.center[0] + r_pos * np.cos(angle)
                            y = p.center[1] + axial
                            z = p.center[2] + r_pos * np.sin(angle)
                            nx = np.cos(angle + side * PI / 2)
                            ny = 0.0
                            nz = np.sin(angle + side * PI / 2)
                        else:
                            x = p.center[0] + axial
                            y = p.center[1] + r_pos * np.cos(angle)
                            z = p.center[2] + r_pos * np.sin(angle)
                            nx = 0.0
                            ny = np.cos(angle + side * PI / 2)
                            nz = np.sin(angle + side * PI / 2)

                        vertices.append([x, y, z])
                        normals.append([nx, ny, nz])

            # Triangles for this vane (both sides)
            base = vane_start + v * 8
            # Side 1
            indices.extend([base, base + 2, base + 3])
            indices.extend([base, base + 3, base + 1])
            # Side 2
            indices.extend([base + 4, base + 5, base + 7])
            indices.extend([base + 4, base + 7, base + 6])

    def _generate_end_plates(self, vertices: List, indices: List, normals: List):
        """Generate end plates."""
        p = self.params
        n_radial = p.resolution_radial // 2

        half_length = p.rotor_length / 2 + p.housing_thickness

        for side in [-1, 1]:
            start_idx = len(vertices)

            if p.axis == "z":
                z = p.center[2] + side * half_length
                nz = side
                # Center
                vertices.append([p.center[0], p.center[1], z])
                normals.append([0.0, 0.0, nz])
                # Ring
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x = p.center[0] + p.housing_outer_radius * np.cos(theta)
                    y = p.center[1] + p.housing_outer_radius * np.sin(theta)
                    vertices.append([x, y, z])
                    normals.append([0.0, 0.0, nz])
            elif p.axis == "y":
                y = p.center[1] + side * half_length
                ny = side
                vertices.append([p.center[0], y, p.center[2]])
                normals.append([0.0, ny, 0.0])
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x = p.center[0] + p.housing_outer_radius * np.cos(theta)
                    z = p.center[2] + p.housing_outer_radius * np.sin(theta)
                    vertices.append([x, y, z])
                    normals.append([0.0, ny, 0.0])
            else:
                x = p.center[0] + side * half_length
                nx = side
                vertices.append([x, p.center[1], p.center[2]])
                normals.append([nx, 0.0, 0.0])
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    y = p.center[1] + p.housing_outer_radius * np.cos(theta)
                    z = p.center[2] + p.housing_outer_radius * np.sin(theta)
                    vertices.append([x, y, z])
                    normals.append([nx, 0.0, 0.0])

            # Triangles
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                if side > 0:
                    indices.extend([start_idx, start_idx + 1 + j, start_idx + 1 + j_next])
                else:
                    indices.extend([start_idx, start_idx + 1 + j_next, start_idx + 1 + j])

    def _generate_inlet_outlet(self, vertices: List, indices: List, normals: List):
        """Generate inlet and outlet flanges."""
        p = self.params
        flange_height = p.housing_thickness * 2

        # Inlet (top, +Y for z-axis rotation)
        self._add_flange(vertices, indices, normals,
                        is_inlet=True, flange_height=flange_height)

        # Outlet (bottom, -Y for z-axis rotation)
        self._add_flange(vertices, indices, normals,
                        is_inlet=False, flange_height=flange_height)

    def _add_flange(self, vertices: List, indices: List, normals: List,
                   is_inlet: bool, flange_height: float):
        """Add inlet or outlet flange."""
        p = self.params
        start_idx = len(vertices)

        width = p.inlet_width if is_inlet else p.outlet_width
        length = p.inlet_length if is_inlet else p.outlet_length

        hw = width / 2
        hl = length / 2
        r = p.housing_outer_radius

        sign = 1 if is_inlet else -1

        if p.axis == "z":
            # Flange on +Y (inlet) or -Y (outlet) side
            y_base = p.center[1] + sign * r
            y_outer = y_base + sign * flange_height

            # 4 corners of rectangular flange
            corners = [
                [p.center[0] - hw, y_outer, p.center[2] - hl],
                [p.center[0] + hw, y_outer, p.center[2] - hl],
                [p.center[0] + hw, y_outer, p.center[2] + hl],
                [p.center[0] - hw, y_outer, p.center[2] + hl],
            ]
            normal = [0.0, sign, 0.0]

        elif p.axis == "y":
            # Flange on +Z (inlet) or -Z (outlet) side
            z_base = p.center[2] + sign * r
            z_outer = z_base + sign * flange_height

            corners = [
                [p.center[0] - hw, p.center[1] - hl, z_outer],
                [p.center[0] + hw, p.center[1] - hl, z_outer],
                [p.center[0] + hw, p.center[1] + hl, z_outer],
                [p.center[0] - hw, p.center[1] + hl, z_outer],
            ]
            normal = [0.0, 0.0, sign]

        else:  # x-axis
            # Flange on +Y (inlet) or -Y (outlet) side
            y_base = p.center[1] + sign * r
            y_outer = y_base + sign * flange_height

            corners = [
                [p.center[0] - hl, y_outer, p.center[2] - hw],
                [p.center[0] + hl, y_outer, p.center[2] - hw],
                [p.center[0] + hl, y_outer, p.center[2] + hw],
                [p.center[0] - hl, y_outer, p.center[2] + hw],
            ]
            normal = [0.0, sign, 0.0]

        for corner in corners:
            vertices.append(corner)
            normals.append(normal)

        # Triangles
        if sign > 0:
            indices.extend([start_idx, start_idx + 1, start_idx + 2])
            indices.extend([start_idx, start_idx + 2, start_idx + 3])
        else:
            indices.extend([start_idx, start_idx + 2, start_idx + 1])
            indices.extend([start_idx, start_idx + 3, start_idx + 2])

    def get_air_leakage_rate(self, pressure_diff: float = 5000.0) -> float:
        """
        Estimate air leakage through tip clearance.

        Args:
            pressure_diff: Pressure difference across valve [Pa]

        Returns:
            Estimated leakage rate [m^3/s]
        """
        p = self.params
        # Simplified orifice flow estimate
        # Q = Cd * A * sqrt(2 * dP / rho)
        Cd = 0.6  # Discharge coefficient
        rho = 1.2  # Air density
        A = PI * p.rotor_diameter * p.vane_tip_clearance  # Annular clearance area

        return Cd * A * np.sqrt(2 * pressure_diff / rho)

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the airlock geometry."""
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


def create_standard_rotary_airlock(
    rotor_diameter: float = 0.20,
    capacity_m3_h: float = 5.0
) -> RotaryAirlock:
    """
    Create a standard rotary airlock valve.

    Args:
        rotor_diameter: Rotor diameter [m]
        capacity_m3_h: Target volumetric capacity [m^3/h]

    Returns:
        RotaryAirlock instance
    """
    # Calculate rotor length for target capacity
    # capacity = pocket_vol * pockets_per_hour * efficiency
    # pocket_vol ~ 0.1 * pi * r^2 * L (rough estimate)

    r = rotor_diameter / 2
    num_vanes = 8
    rpm = 20
    efficiency = 0.6
    pockets_per_hour = rpm * 60 * num_vanes

    # Solve for length
    required_pocket_vol = capacity_m3_h / (pockets_per_hour * efficiency)
    # pocket_vol ~ sector_area * length ~ 0.1 * pi * r^2 * length
    rotor_length = required_pocket_vol / (0.1 * PI * r ** 2)

    # Reasonable limits
    rotor_length = max(rotor_length, rotor_diameter * 0.5)
    rotor_length = min(rotor_length, rotor_diameter * 2.0)

    params = RotaryAirlockParams(
        rotor_diameter=rotor_diameter,
        rotor_length=rotor_length,
        num_vanes=num_vanes,
        vane_thickness=0.005,
        vane_tip_clearance=0.0003,  # 0.3mm typical
        rpm=rpm,
    )

    return RotaryAirlock(params)
