"""
Inlet air filter component for clean air supply.

The inlet air filter removes particulates from ambient air before
it enters the classification system. This protects the product
from contamination and extends equipment life.

Types:
- Panel filters: Simple, low cost, moderate efficiency
- Bag/pocket filters: High surface area, good efficiency
- Cartridge filters: Compact, high efficiency
- HEPA filters: Very high efficiency for critical applications
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import warp as wp

from ...utils.constants import PI, TWO_PI


@dataclass
class InletAirFilterParams:
    """Parameters for inlet air filter housing."""

    # Housing geometry
    housing_width: float         # [m] Housing width
    housing_height: float        # [m] Housing height
    housing_depth: float         # [m] Housing depth (flow direction)

    # Filter element
    filter_type: str = "panel"   # "panel", "bag", "cartridge", "HEPA"
    num_elements: int = 1        # Number of filter elements
    element_thickness: float = 0.05  # [m] Filter element thickness

    # Connections
    inlet_diameter: float = None  # [m] Inlet duct diameter
    outlet_diameter: float = None  # [m] Outlet duct diameter

    # Performance
    filter_area: float = None    # [m²] Total filter area
    efficiency_class: str = "G4"  # Filter class: G1-G4, M5-M6, F7-F9, E10-E12, H13-H14
    clean_pressure_drop: float = 100  # [Pa] Clean filter pressure drop
    max_pressure_drop: float = 400    # [Pa] Max pressure drop (change-out indicator)

    # Position
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Mesh resolution
    resolution: int = 16

    def __post_init__(self):
        if self.inlet_diameter is None:
            # Size inlet for ~5 m/s velocity
            self.inlet_diameter = self.housing_width * 0.6
        if self.outlet_diameter is None:
            self.outlet_diameter = self.inlet_diameter
        if self.filter_area is None:
            # Approximate filter area (face area x depth factor)
            face_area = self.housing_width * self.housing_height
            if self.filter_type == "bag":
                self.filter_area = face_area * 3  # Bags have more area
            elif self.filter_type == "cartridge":
                self.filter_area = face_area * 5  # Cartridges even more
            else:
                self.filter_area = face_area * 0.8

    @property
    def face_area(self) -> float:
        """Face area of filter housing [m²]."""
        return self.housing_width * self.housing_height

    @property
    def face_velocity(self) -> float:
        """Estimate face velocity for given flow [m/s]."""
        # Typical face velocity depends on filter type
        if self.filter_type == "HEPA":
            return 0.05  # Very low for HEPA
        elif self.filter_type == "cartridge":
            return 0.5
        elif self.filter_type == "bag":
            return 0.8
        else:
            return 2.5  # Panel filters

    def max_flow_rate(self) -> float:
        """Maximum recommended flow rate [m³/h]."""
        return self.face_area * self.face_velocity * 3600


class InletAirFilter:
    """
    Inlet air filter housing with filter elements.

    Components:
    - Rectangular housing
    - Inlet plenum
    - Filter element(s)
    - Outlet plenum
    - Access door

    Coordinate system:
    - Origin at center of housing
    - Flow direction along +X axis
    """

    def __init__(self, params: InletAirFilterParams):
        """
        Initialize inlet air filter.

        Args:
            params: InletAirFilterParams defining the geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for the air filter housing.

        Returns:
            Tuple of (vertices, indices, normals)
        """
        p = self.params
        vertices = []
        indices = []
        normals = []

        # Generate main housing
        self._generate_housing(vertices, indices, normals)

        # Generate filter element(s)
        self._generate_filter_elements(vertices, indices, normals)

        # Generate inlet duct
        self._generate_inlet(vertices, indices, normals)

        # Generate outlet duct
        self._generate_outlet(vertices, indices, normals)

        # Generate access door
        self._generate_access_door(vertices, indices, normals)

        self._vertices = np.array(vertices, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.int32)
        self._normals = np.array(normals, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def _generate_housing(self, vertices: List, indices: List, normals: List):
        """Generate rectangular housing box."""
        p = self.params

        hw = p.housing_width / 2
        hh = p.housing_height / 2
        hd = p.housing_depth / 2

        wall_thickness = 0.002  # 2mm walls

        # Outer box vertices (8 corners)
        start_idx = len(vertices)

        corners = [
            # Back face (inlet side, -X)
            [p.center[0] - hd, p.center[1] - hh, p.center[2] - hw],
            [p.center[0] - hd, p.center[1] + hh, p.center[2] - hw],
            [p.center[0] - hd, p.center[1] + hh, p.center[2] + hw],
            [p.center[0] - hd, p.center[1] - hh, p.center[2] + hw],
            # Front face (outlet side, +X)
            [p.center[0] + hd, p.center[1] - hh, p.center[2] - hw],
            [p.center[0] + hd, p.center[1] + hh, p.center[2] - hw],
            [p.center[0] + hd, p.center[1] + hh, p.center[2] + hw],
            [p.center[0] + hd, p.center[1] - hh, p.center[2] + hw],
        ]

        # Face normals
        face_data = [
            ([0, 1, 2, 3], [-1, 0, 0]),  # Back (-X)
            ([4, 7, 6, 5], [1, 0, 0]),   # Front (+X)
            ([0, 4, 5, 1], [0, 0, -1]),  # Left (-Z)
            ([3, 2, 6, 7], [0, 0, 1]),   # Right (+Z)
            ([0, 3, 7, 4], [0, -1, 0]),  # Bottom (-Y)
            ([1, 5, 6, 2], [0, 1, 0]),   # Top (+Y)
        ]

        for corner in corners:
            vertices.append(corner)
            normals.append([0, 0, 1])  # Will be overwritten per face

        # Generate faces
        for face_indices, normal in face_data:
            face_start = len(vertices)
            for idx in face_indices:
                vertices.append(corners[idx])
                normals.append(normal)

            # Two triangles per face
            indices.extend([face_start, face_start + 1, face_start + 2])
            indices.extend([face_start, face_start + 2, face_start + 3])

    def _generate_filter_elements(self, vertices: List, indices: List, normals: List):
        """Generate filter element representations."""
        p = self.params

        # Filter element is a rectangular sheet inside housing
        # Position depends on number of elements

        element_spacing = p.housing_depth / (p.num_elements + 1)

        hw = p.housing_width / 2 * 0.95  # Slightly smaller than housing
        hh = p.housing_height / 2 * 0.95
        half_thick = p.element_thickness / 2

        for e in range(p.num_elements):
            start_idx = len(vertices)

            x_pos = p.center[0] - p.housing_depth / 2 + (e + 1) * element_spacing

            # Filter element as a thin box
            corners = [
                [x_pos - half_thick, p.center[1] - hh, p.center[2] - hw],
                [x_pos - half_thick, p.center[1] + hh, p.center[2] - hw],
                [x_pos - half_thick, p.center[1] + hh, p.center[2] + hw],
                [x_pos - half_thick, p.center[1] - hh, p.center[2] + hw],
                [x_pos + half_thick, p.center[1] - hh, p.center[2] - hw],
                [x_pos + half_thick, p.center[1] + hh, p.center[2] - hw],
                [x_pos + half_thick, p.center[1] + hh, p.center[2] + hw],
                [x_pos + half_thick, p.center[1] - hh, p.center[2] + hw],
            ]

            for corner in corners:
                vertices.append(corner)
                normals.append([1, 0, 0])

            # Back face
            indices.extend([start_idx, start_idx + 2, start_idx + 1])
            indices.extend([start_idx, start_idx + 3, start_idx + 2])
            # Front face
            indices.extend([start_idx + 4, start_idx + 5, start_idx + 6])
            indices.extend([start_idx + 4, start_idx + 6, start_idx + 7])

    def _generate_inlet(self, vertices: List, indices: List, normals: List):
        """Generate inlet duct connection."""
        p = self.params
        n_radial = p.resolution

        start_idx = len(vertices)
        r = p.inlet_diameter / 2
        duct_length = p.inlet_diameter

        x_start = p.center[0] - p.housing_depth / 2
        x_end = x_start - duct_length

        for i in range(2):
            x = x_start if i == 0 else x_end
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                y = p.center[1] + r * np.sin(theta)
                z = p.center[2] + r * np.cos(theta)
                vertices.append([x, y, z])
                normals.append([0, np.sin(theta), np.cos(theta)])

        # Triangles
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = start_idx + j
            v1 = start_idx + j_next
            v2 = start_idx + n_radial + j_next
            v3 = start_idx + n_radial + j

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

    def _generate_outlet(self, vertices: List, indices: List, normals: List):
        """Generate outlet duct connection."""
        p = self.params
        n_radial = p.resolution

        start_idx = len(vertices)
        r = p.outlet_diameter / 2
        duct_length = p.outlet_diameter

        x_start = p.center[0] + p.housing_depth / 2
        x_end = x_start + duct_length

        for i in range(2):
            x = x_start if i == 0 else x_end
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                y = p.center[1] + r * np.sin(theta)
                z = p.center[2] + r * np.cos(theta)
                vertices.append([x, y, z])
                normals.append([0, np.sin(theta), np.cos(theta)])

        # Triangles
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = start_idx + j
            v1 = start_idx + j_next
            v2 = start_idx + n_radial + j_next
            v3 = start_idx + n_radial + j

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

    def _generate_access_door(self, vertices: List, indices: List, normals: List):
        """Generate access door on side of housing."""
        p = self.params

        # Door on the +Z side
        door_width = p.housing_depth * 0.8
        door_height = p.housing_height * 0.7

        start_idx = len(vertices)

        hw = door_width / 2
        hh = door_height / 2
        z = p.center[2] + p.housing_width / 2 + 0.005  # Slightly proud

        # Door corners
        corners = [
            [p.center[0] - hw, p.center[1] - hh, z],
            [p.center[0] + hw, p.center[1] - hh, z],
            [p.center[0] + hw, p.center[1] + hh, z],
            [p.center[0] - hw, p.center[1] + hh, z],
        ]

        for corner in corners:
            vertices.append(corner)
            normals.append([0, 0, 1])

        indices.extend([start_idx, start_idx + 1, start_idx + 2])
        indices.extend([start_idx, start_idx + 2, start_idx + 3])

    def get_pressure_drop(self, flow_rate: float, loading: float = 0.0) -> float:
        """
        Estimate pressure drop across filter.

        Args:
            flow_rate: Air flow rate [m³/h]
            loading: Dust loading factor (0=clean, 1=fully loaded)

        Returns:
            Estimated pressure drop [Pa]
        """
        p = self.params
        dp_clean = p.clean_pressure_drop

        # Pressure drop increases with loading
        dp_loaded = p.max_pressure_drop

        return dp_clean + loading * (dp_loaded - dp_clean)

    def get_efficiency(self) -> dict:
        """
        Get filter efficiency based on class.

        Returns:
            Dict with particle size and efficiency
        """
        # Approximate efficiencies by filter class
        efficiency_data = {
            "G1": {"0.4um": 0.0, "1um": 0.05, "5um": 0.50, "10um": 0.80},
            "G2": {"0.4um": 0.0, "1um": 0.10, "5um": 0.60, "10um": 0.85},
            "G3": {"0.4um": 0.0, "1um": 0.15, "5um": 0.70, "10um": 0.90},
            "G4": {"0.4um": 0.0, "1um": 0.20, "5um": 0.80, "10um": 0.95},
            "M5": {"0.4um": 0.20, "1um": 0.50, "5um": 0.90, "10um": 0.99},
            "M6": {"0.4um": 0.35, "1um": 0.65, "5um": 0.95, "10um": 0.99},
            "F7": {"0.4um": 0.50, "1um": 0.80, "5um": 0.99, "10um": 0.99},
            "F8": {"0.4um": 0.65, "1um": 0.90, "5um": 0.99, "10um": 0.99},
            "F9": {"0.4um": 0.80, "1um": 0.95, "5um": 0.99, "10um": 0.99},
            "H13": {"0.3um": 0.9997, "0.4um": 0.9999, "1um": 0.9999, "5um": 0.9999},
            "H14": {"0.3um": 0.99995, "0.4um": 0.99999, "1um": 0.99999, "5um": 0.99999},
        }

        return efficiency_data.get(self.params.efficiency_class, {"unknown": 0})

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the filter housing geometry."""
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
    def ports(self) -> dict:
        """
        Get connection ports for the filter.
        
        Air flow direction: -X (inlet) → +X (outlet)
        
        Ports:
        - 'inlet': Inlet side of filter housing (air enters)
        - 'outlet': Outlet side of filter housing (clean air exits)
        
        Returns:
            Dictionary of port name to ConnectionPort
        """
        from ..connection_ports import ConnectionPort, PortType
        
        p = self.params
        half_depth = p.housing_depth / 2
        
        # Inlet port: at -X face of housing
        inlet_pos = (p.center[0] - half_depth, p.center[1], p.center[2])
        
        # Outlet port: at +X face of housing
        outlet_pos = (p.center[0] + half_depth, p.center[1], p.center[2])
        
        return {
            'inlet': ConnectionPort(
                position=inlet_pos,
                direction=(-1.0, 0.0, 0.0),  # Air enters from -X
                diameter=p.inlet_diameter,
                port_type=PortType.CIRCULAR,
                name="filter_inlet",
                compatible_types=[PortType.CIRCULAR, PortType.FLANGED],
            ),
            'outlet': ConnectionPort(
                position=outlet_pos,
                direction=(1.0, 0.0, 0.0),  # Clean air exits toward +X
                diameter=p.outlet_diameter,
                port_type=PortType.CIRCULAR,
                name="filter_outlet",
                compatible_types=[PortType.CIRCULAR, PortType.FLANGED],
            ),
        }


def create_standard_inlet_filter(
    flow_rate: float = 3000,
    filter_type: str = "panel",
    efficiency_class: str = "G4"
) -> InletAirFilter:
    """
    Create a standard inlet air filter sized for given flow rate.

    Args:
        flow_rate: Design flow rate [m³/h]
        filter_type: Filter type ("panel", "bag", "cartridge", "HEPA")
        efficiency_class: Filter efficiency class

    Returns:
        InletAirFilter instance
    """
    # Size based on face velocity
    if filter_type == "HEPA":
        face_velocity = 0.05  # m/s
    elif filter_type == "cartridge":
        face_velocity = 0.5
    elif filter_type == "bag":
        face_velocity = 0.8
    else:  # panel
        face_velocity = 2.5

    face_area = flow_rate / 3600 / face_velocity

    # Assume square face
    side = np.sqrt(face_area)

    # Depth depends on filter type
    if filter_type == "bag":
        depth = side * 0.8
    elif filter_type == "cartridge":
        depth = side * 0.6
    else:
        depth = side * 0.3

    params = InletAirFilterParams(
        housing_width=side,
        housing_height=side,
        housing_depth=depth,
        filter_type=filter_type,
        efficiency_class=efficiency_class,
    )

    return InletAirFilter(params)
