"""
Venturi eductor component for particle entrainment.

The venturi eductor uses the Venturi effect to entrain particles
from a hopper into the airstream. High-velocity air through the
throat creates a low-pressure zone that draws in particles.

Principle:
- Motive air accelerates through converging section
- Low pressure at throat draws in particles from side inlet
- Mixed flow expands and decelerates in diverging section
"""

from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np
import warp as wp

from ...utils.constants import PI, TWO_PI
from ..connection_ports import ConnectionPort, PortType


@dataclass
class VenturiEducatorParams:
    """Parameters for venturi eductor."""

    # Main venturi dimensions
    inlet_diameter: float        # [m] Air inlet diameter
    throat_diameter: float       # [m] Throat (minimum) diameter
    outlet_diameter: float       # [m] Mixed flow outlet diameter

    # Angles
    convergent_angle: float      # [rad] Inlet convergent half-angle (typically 10-15°)
    divergent_angle: float       # [rad] Outlet divergent half-angle (typically 3-7°)

    # Solids inlet
    solids_inlet_diameter: float # [m] Particle feed inlet diameter
    solids_inlet_angle: float    # [rad] Angle of solids entry (from horizontal)
    solids_inlet_position: float # [m] Distance from throat start to solids entry

    # Section lengths (if not specified, calculated from angles)
    throat_length: float = None  # [m] Length of throat section
    convergent_length: float = None  # [m] Will be calculated if None
    divergent_length: float = None   # [m] Will be calculated if None

    # Position
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Center of inlet
    axis: str = "x"              # Axis along which eductor extends

    # Mesh resolution
    resolution_radial: int = 24
    resolution_axial: int = 20

    def __post_init__(self):
        # Calculate section lengths from angles if not provided
        if self.convergent_length is None:
            dr = (self.inlet_diameter - self.throat_diameter) / 2
            self.convergent_length = dr / np.tan(self.convergent_angle)

        if self.throat_length is None:
            self.throat_length = self.throat_diameter  # Typical: L = D

        if self.divergent_length is None:
            dr = (self.outlet_diameter - self.throat_diameter) / 2
            self.divergent_length = dr / np.tan(self.divergent_angle)

    @property
    def total_length(self) -> float:
        """Total length of eductor."""
        return self.convergent_length + self.throat_length + self.divergent_length

    @property
    def inlet_area(self) -> float:
        """Inlet cross-sectional area."""
        return PI * (self.inlet_diameter / 2) ** 2

    @property
    def throat_area(self) -> float:
        """Throat cross-sectional area."""
        return PI * (self.throat_diameter / 2) ** 2

    @property
    def area_ratio(self) -> float:
        """Ratio of inlet to throat area."""
        return self.inlet_area / self.throat_area

    @property
    def throat_start_position(self) -> float:
        """Axial position where throat begins."""
        return self.convergent_length

    @property
    def throat_end_position(self) -> float:
        """Axial position where throat ends."""
        return self.convergent_length + self.throat_length


class VenturiEducator:
    """
    Venturi eductor for entraining particles into airstream.

    Components:
    - Convergent section: Accelerates air
    - Throat: Minimum area, maximum velocity, low pressure
    - Solids inlet: Particle entry at throat
    - Divergent section: Pressure recovery

    Coordinate system:
    - Origin at center of air inlet
    - Flow direction along positive axis
    """

    def __init__(self, params: VenturiEducatorParams):
        """
        Initialize venturi eductor.

        Args:
            params: VenturiEducatorParams defining the geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for the venturi eductor.

        Returns:
            Tuple of (vertices, indices, normals)
        """
        p = self.params
        vertices = []
        normals = []
        indices = []

        n_radial = p.resolution_radial
        n_axial = p.resolution_axial

        # Define axial stations and radii
        stations = self._get_axial_stations()

        # Generate main venturi body (surface of revolution)
        for i, (x, r) in enumerate(stations):
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI

                # Position based on axis
                if p.axis == "x":
                    vx = x + p.center[0]
                    vy = r * np.cos(theta) + p.center[1]
                    vz = r * np.sin(theta) + p.center[2]
                    nx, ny, nz = 0.0, np.cos(theta), np.sin(theta)
                elif p.axis == "y":
                    vx = r * np.cos(theta) + p.center[0]
                    vy = x + p.center[1]
                    vz = r * np.sin(theta) + p.center[2]
                    nx, ny, nz = np.cos(theta), 0.0, np.sin(theta)
                else:  # z-axis
                    vx = r * np.cos(theta) + p.center[0]
                    vy = r * np.sin(theta) + p.center[1]
                    vz = x + p.center[2]
                    nx, ny, nz = np.cos(theta), np.sin(theta), 0.0

                vertices.append([vx, vy, vz])

                # Calculate surface normal (accounting for taper)
                if i > 0 and i < len(stations) - 1:
                    dx = stations[i + 1][0] - stations[i - 1][0]
                    dr = stations[i + 1][1] - stations[i - 1][1]
                    # Normal perpendicular to surface
                    n_axial_component = -dr / np.sqrt(dx * dx + dr * dr) if dx != 0 else 0
                    n_radial_component = dx / np.sqrt(dx * dx + dr * dr) if dx != 0 else 1
                else:
                    n_axial_component = 0
                    n_radial_component = 1

                if p.axis == "x":
                    normals.append([n_axial_component, n_radial_component * np.cos(theta),
                                   n_radial_component * np.sin(theta)])
                elif p.axis == "y":
                    normals.append([n_radial_component * np.cos(theta), n_axial_component,
                                   n_radial_component * np.sin(theta)])
                else:
                    normals.append([n_radial_component * np.cos(theta),
                                   n_radial_component * np.sin(theta), n_axial_component])

        # Generate triangles for main body
        n_stations = len(stations)
        for i in range(n_stations - 1):
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = i * n_radial + j
                v1 = i * n_radial + j_next
                v2 = (i + 1) * n_radial + j_next
                v3 = (i + 1) * n_radial + j

                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

        # Add solids inlet tube
        self._add_solids_inlet(vertices, indices, normals)

        self._vertices = np.array(vertices, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.int32)
        self._normals = np.array(normals, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def _get_axial_stations(self) -> List[Tuple[float, float]]:
        """
        Get axial stations with corresponding radii for the venturi profile.

        Returns:
            List of (axial_position, radius) tuples
        """
        p = self.params
        stations = []

        n = p.resolution_axial

        # Convergent section
        n_conv = n // 3
        for i in range(n_conv + 1):
            t = i / n_conv
            x = t * p.convergent_length
            r_inlet = p.inlet_diameter / 2
            r_throat = p.throat_diameter / 2
            r = r_inlet + (r_throat - r_inlet) * t
            stations.append((x, r))

        # Throat section
        n_throat = n // 6
        x_start = p.convergent_length
        for i in range(1, n_throat + 1):
            t = i / n_throat
            x = x_start + t * p.throat_length
            r = p.throat_diameter / 2
            stations.append((x, r))

        # Divergent section
        n_div = n - n_conv - n_throat
        x_start = p.convergent_length + p.throat_length
        for i in range(1, n_div + 1):
            t = i / n_div
            x = x_start + t * p.divergent_length
            r_throat = p.throat_diameter / 2
            r_outlet = p.outlet_diameter / 2
            r = r_throat + (r_outlet - r_throat) * t
            stations.append((x, r))

        return stations

    def _add_solids_inlet(self, vertices: List, indices: List, normals: List):
        """Add solids inlet tube geometry."""
        p = self.params

        # Position of solids inlet
        x_inlet = p.throat_start_position + p.solids_inlet_position
        r_main = p.throat_diameter / 2
        r_inlet = p.solids_inlet_diameter / 2

        # Inlet tube length
        tube_length = p.throat_diameter * 2

        n_radial = p.resolution_radial // 2
        start_idx = len(vertices)

        # Generate tube along inlet angle
        for i in range(2):  # Start and end of tube
            t = i  # 0 at main body, 1 at outer end

            if p.axis == "x":
                x_pos = x_inlet + p.center[0]
                # Tube extends in Y-Z plane at angle
                y_base = p.center[1] + r_main + t * tube_length * np.cos(p.solids_inlet_angle)
                z_base = p.center[2] + t * tube_length * np.sin(p.solids_inlet_angle)

                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    # Tube cross-section perpendicular to tube axis
                    vy = y_base + r_inlet * np.cos(theta) * np.sin(p.solids_inlet_angle)
                    vz = z_base + r_inlet * np.sin(theta)
                    vx = x_pos + r_inlet * np.cos(theta) * np.cos(p.solids_inlet_angle)

                    vertices.append([vx, vy, vz])
                    # Normal pointing outward from tube
                    normals.append([np.cos(theta) * np.cos(p.solids_inlet_angle),
                                   np.cos(theta) * np.sin(p.solids_inlet_angle),
                                   np.sin(theta)])

        # Generate triangles for inlet tube
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = start_idx + j
            v1 = start_idx + j_next
            v2 = start_idx + n_radial + j_next
            v3 = start_idx + n_radial + j

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

    def get_velocity_at_throat(self, inlet_velocity: float) -> float:
        """
        Calculate velocity at throat using continuity.

        Args:
            inlet_velocity: Velocity at inlet [m/s]

        Returns:
            Velocity at throat [m/s]
        """
        return inlet_velocity * self.params.area_ratio

    def get_pressure_drop_at_throat(self, inlet_velocity: float,
                                    air_density: float = 1.2) -> float:
        """
        Estimate pressure drop at throat (Bernoulli, ideal).

        Args:
            inlet_velocity: Velocity at inlet [m/s]
            air_density: Air density [kg/m³]

        Returns:
            Pressure drop at throat [Pa]
        """
        v_throat = self.get_velocity_at_throat(inlet_velocity)
        # Bernoulli: P1 + 0.5*rho*v1² = P2 + 0.5*rho*v2²
        # dP = 0.5 * rho * (v_throat² - v_inlet²)
        return 0.5 * air_density * (v_throat ** 2 - inlet_velocity ** 2)

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the eductor geometry."""
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
    def ports(self) -> Dict[str, ConnectionPort]:
        """
        Get connection ports for the venturi eductor.
        
        Ports:
        - air_inlet: Main air inlet at start (circular, -X direction)
        - solids_inlet: Particle feed inlet at throat (angled, typically +Y component)
        - outlet: Mixed flow outlet at end (circular, +X direction)
        """
        p = self.params
        
        # Air inlet at origin (start of venturi)
        air_inlet = ConnectionPort(
            position=p.center,
            direction=(-1.0, 0.0, 0.0) if p.axis == "x" else (0.0, -1.0, 0.0) if p.axis == "y" else (0.0, 0.0, -1.0),
            diameter=p.inlet_diameter,
            port_type=PortType.CIRCULAR,
            name="air_inlet"
        )
        
        # Solids inlet at throat
        x_throat = p.throat_start_position + p.solids_inlet_position
        if p.axis == "x":
            solids_pos = (
                p.center[0] + x_throat,
                p.center[1] + p.throat_diameter / 2,
                p.center[2]
            )
            solids_dir = (0.0, 1.0, 0.0)  # Points upward (feed from above)
        elif p.axis == "y":
            solids_pos = (
                p.center[0] + p.throat_diameter / 2,
                p.center[1] + x_throat,
                p.center[2]
            )
            solids_dir = (1.0, 0.0, 0.0)
        else:
            solids_pos = (
                p.center[0] + p.throat_diameter / 2,
                p.center[1],
                p.center[2] + x_throat
            )
            solids_dir = (1.0, 0.0, 0.0)
        
        solids_inlet = ConnectionPort(
            position=solids_pos,
            direction=solids_dir,
            diameter=p.solids_inlet_diameter,
            port_type=PortType.CIRCULAR,
            name="solids_inlet"
        )
        
        # Outlet at end of venturi
        if p.axis == "x":
            outlet_pos = (p.center[0] + p.total_length, p.center[1], p.center[2])
            outlet_dir = (1.0, 0.0, 0.0)
        elif p.axis == "y":
            outlet_pos = (p.center[0], p.center[1] + p.total_length, p.center[2])
            outlet_dir = (0.0, 1.0, 0.0)
        else:
            outlet_pos = (p.center[0], p.center[1], p.center[2] + p.total_length)
            outlet_dir = (0.0, 0.0, 1.0)
        
        outlet = ConnectionPort(
            position=outlet_pos,
            direction=outlet_dir,
            diameter=p.outlet_diameter,
            port_type=PortType.CIRCULAR,
            name="outlet"
        )
        
        return {
            'air_inlet': air_inlet,
            'solids_inlet': solids_inlet,
            'outlet': outlet
        }


def create_standard_venturi_eductor(
    inlet_diameter: float = 0.1,
    throat_ratio: float = 0.5
) -> VenturiEducator:
    """
    Create a standard venturi eductor with typical proportions.

    Args:
        inlet_diameter: Inlet diameter [m]
        throat_ratio: Throat to inlet diameter ratio (0.3-0.6 typical)

    Returns:
        VenturiEducator instance
    """
    throat_d = inlet_diameter * throat_ratio

    params = VenturiEducatorParams(
        inlet_diameter=inlet_diameter,
        throat_diameter=throat_d,
        outlet_diameter=inlet_diameter * 0.9,  # Slightly smaller outlet
        convergent_angle=np.radians(12),       # 12° half-angle
        divergent_angle=np.radians(5),         # 5° half-angle for good recovery
        solids_inlet_diameter=throat_d * 0.8,
        solids_inlet_angle=np.radians(45),     # 45° entry
        solids_inlet_position=throat_d * 0.3,  # Just into throat
    )

    return VenturiEducator(params)
