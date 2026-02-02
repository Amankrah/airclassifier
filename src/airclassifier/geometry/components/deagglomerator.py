"""
De-agglomerator / lump breaker component for breaking powder clumps.

The de-agglomerator breaks up agglomerated powder lumps before
classification. Uses rotating pins/paddles and a sizing screen
to ensure consistent particle size entering the classifier.

Principle:
- Rotating pins break up lumps by impact
- Screen allows properly sized particles to pass
- Oversize material recirculated or rejected
"""

from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np
import warp as wp

from ...utils.constants import PI, TWO_PI
from ..connection_ports import ConnectionPort, PortType


@dataclass
class DeagglomeratorParams:
    """Parameters for de-agglomerator / lump breaker."""

    # Rotor geometry
    rotor_diameter: float        # [m] Rotor diameter (to pin tips)
    rotor_length: float          # [m] Rotor length (width)
    shaft_diameter: float        # [m] Central shaft diameter

    # Pins/paddles
    num_pin_rows: int            # Number of rows of pins along length
    pins_per_row: int            # Number of pins per row (circumferential)
    pin_diameter: float          # [m] Pin/paddle diameter
    pin_length: float            # [m] Pin length (radial)

    # Housing
    housing_diameter: float      # [m] Housing inner diameter
    housing_length: float        # [m] Housing length

    # Screen
    screen_diameter: float       # [m] Screen outer diameter
    screen_aperture: float       # [m] Screen hole/slot size
    screen_open_area: float      # [%] Open area fraction (0-1)

    # Connections
    inlet_diameter: float        # [m] Inlet diameter
    outlet_diameter: float       # [m] Outlet diameter

    # Operating parameters
    rpm: float = 1500.0          # [rpm] Rotation speed (high speed)

    # Position
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Center of housing
    axis: str = "x"              # Rotation axis

    # Mesh resolution
    resolution_radial: int = 24
    resolution_axial: int = 16

    @property
    def rotor_radius(self) -> float:
        """Rotor radius to pin tips."""
        return self.rotor_diameter / 2

    @property
    def shaft_radius(self) -> float:
        """Shaft radius."""
        return self.shaft_diameter / 2

    @property
    def housing_radius(self) -> float:
        """Housing radius."""
        return self.housing_diameter / 2

    @property
    def screen_radius(self) -> float:
        """Screen radius."""
        return self.screen_diameter / 2

    @property
    def pin_tip_speed(self) -> float:
        """Pin tip speed [m/s]."""
        return PI * self.rotor_diameter * self.rpm / 60

    @property
    def clearance(self) -> float:
        """Clearance between pin tips and screen."""
        return self.screen_radius - self.rotor_radius


class Deagglomerator:
    """
    De-agglomerator / lump breaker for powder processing.

    Components:
    - Cylindrical housing
    - Rotor with pin rows
    - Sizing screen (cylindrical)
    - Inlet chute
    - Outlet

    Coordinate system:
    - Origin at center of housing
    - Rotation axis along specified direction
    """

    def __init__(self, params: DeagglomeratorParams):
        """
        Initialize de-agglomerator.

        Args:
            params: DeagglomeratorParams defining the geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None
        
        # Animation state for rotor
        self._rotor_angle = 0.0  # Current rotation angle [radians]
        
        # Cached separate meshes for animation
        self._static_vertices = None
        self._static_indices = None
        self._static_normals = None
        self._rotor_vertices = None
        self._rotor_indices = None
        self._rotor_normals = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for the de-agglomerator.

        Returns:
            Tuple of (vertices, indices, normals)
        """
        p = self.params
        vertices = []
        indices = []
        normals = []

        # Generate housing
        self._generate_housing(vertices, indices, normals)

        # Generate screen (perforated cylinder inside housing)
        self._generate_screen(vertices, indices, normals)

        # Generate rotor with pins
        self._generate_rotor(vertices, indices, normals)

        # Generate inlet
        self._generate_inlet(vertices, indices, normals)

        # Generate outlet
        self._generate_outlet(vertices, indices, normals)

        # Generate end plates
        self._generate_end_plates(vertices, indices, normals)

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
        r = p.housing_radius
        half_length = p.housing_length / 2
        wall_thickness = 0.005

        # Outer surface of housing
        for i in range(n_axial + 1):
            t = i / n_axial
            axial_pos = -half_length + t * p.housing_length

            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI

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

    def _generate_screen(self, vertices: List, indices: List, normals: List):
        """Generate perforated screen cylinder."""
        p = self.params
        n_radial = p.resolution_radial
        n_axial = p.resolution_axial // 2

        start_idx = len(vertices)
        r = p.screen_radius
        half_length = p.rotor_length / 2

        # Screen is a cylinder with "holes" represented visually
        # We simplify by just creating the cylinder
        for i in range(n_axial + 1):
            t = i / n_axial
            axial_pos = -half_length + t * p.rotor_length

            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI

                if p.axis == "x":
                    x = p.center[0] + axial_pos
                    y = p.center[1] + r * np.sin(theta)
                    z = p.center[2] + r * np.cos(theta)
                    # Normal pointing inward (toward rotor)
                    nx, ny, nz = 0.0, -np.sin(theta), -np.cos(theta)
                elif p.axis == "y":
                    x = p.center[0] + r * np.cos(theta)
                    y = p.center[1] + axial_pos
                    z = p.center[2] + r * np.sin(theta)
                    nx, ny, nz = -np.cos(theta), 0.0, -np.sin(theta)
                else:
                    x = p.center[0] + r * np.cos(theta)
                    y = p.center[1] + r * np.sin(theta)
                    z = p.center[2] + axial_pos
                    nx, ny, nz = -np.cos(theta), -np.sin(theta), 0.0

                vertices.append([x, y, z])
                normals.append([nx, ny, nz])

        # Generate triangles
        for i in range(n_axial):
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = start_idx + i * n_radial + j
                v1 = start_idx + i * n_radial + j_next
                v2 = start_idx + (i + 1) * n_radial + j_next
                v3 = start_idx + (i + 1) * n_radial + j

                # Reverse winding for inward-facing surface
                indices.extend([v0, v2, v1])
                indices.extend([v0, v3, v2])

    def _generate_rotor(self, vertices: List, indices: List, normals: List):
        """Generate rotor with pins."""
        p = self.params
        n_radial = p.resolution_radial

        # Central shaft
        shaft_start = len(vertices)
        half_length = p.rotor_length / 2

        for i in range(2):
            axial_pos = -half_length + i * p.rotor_length

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

        # Generate pins
        self._generate_pins(vertices, indices, normals)

    def _generate_pins(self, vertices: List, indices: List, normals: List):
        """Generate pin/paddle array on rotor."""
        p = self.params
        n_pin_segments = 6  # Segments around each pin

        half_length = p.rotor_length / 2
        row_spacing = p.rotor_length / (p.num_pin_rows + 1)

        for row in range(p.num_pin_rows):
            axial_pos = -half_length + (row + 1) * row_spacing

            for pin in range(p.pins_per_row):
                pin_angle = (pin / p.pins_per_row) * TWO_PI

                pin_start = len(vertices)

                # Pin is a small cylinder extending radially from shaft
                r_inner = p.shaft_radius
                r_outer = p.shaft_radius + p.pin_length
                pin_radius = p.pin_diameter / 2

                # Generate pin cylinder
                for r_idx, r in enumerate([r_inner, r_outer]):
                    for seg in range(n_pin_segments):
                        seg_angle = (seg / n_pin_segments) * TWO_PI

                        if p.axis == "x":
                            # Pin extends in Y-Z plane
                            x = p.center[0] + axial_pos
                            base_y = p.center[1] + r * np.sin(pin_angle)
                            base_z = p.center[2] + r * np.cos(pin_angle)
                            # Pin cross-section perpendicular to radial direction
                            y = base_y + pin_radius * np.cos(seg_angle) * np.cos(pin_angle)
                            z = base_z - pin_radius * np.cos(seg_angle) * np.sin(pin_angle)
                            x = x + pin_radius * np.sin(seg_angle)
                        elif p.axis == "y":
                            y = p.center[1] + axial_pos
                            base_x = p.center[0] + r * np.cos(pin_angle)
                            base_z = p.center[2] + r * np.sin(pin_angle)
                            x = base_x + pin_radius * np.cos(seg_angle) * np.cos(pin_angle)
                            z = base_z - pin_radius * np.cos(seg_angle) * np.sin(pin_angle)
                            y = y + pin_radius * np.sin(seg_angle)
                        else:
                            z = p.center[2] + axial_pos
                            base_x = p.center[0] + r * np.cos(pin_angle)
                            base_y = p.center[1] + r * np.sin(pin_angle)
                            x = base_x + pin_radius * np.cos(seg_angle) * np.cos(pin_angle)
                            y = base_y - pin_radius * np.cos(seg_angle) * np.sin(pin_angle)
                            z = z + pin_radius * np.sin(seg_angle)

                        vertices.append([x, y, z])
                        # Simplified normal
                        normals.append([np.cos(pin_angle), np.sin(pin_angle), 0.0])

                # Pin triangles
                for seg in range(n_pin_segments):
                    seg_next = (seg + 1) % n_pin_segments
                    v0 = pin_start + seg
                    v1 = pin_start + seg_next
                    v2 = pin_start + n_pin_segments + seg_next
                    v3 = pin_start + n_pin_segments + seg

                    indices.extend([v0, v1, v2])
                    indices.extend([v0, v2, v3])

    def _generate_inlet(self, vertices: List, indices: List, normals: List):
        """Generate circular inlet neck with flange."""
        p = self.params
        n_radial = max(16, p.resolution_radial // 2)

        start_idx = len(vertices)
        r = p.inlet_diameter / 2
        inlet_length = p.inlet_diameter * 0.6  # Neck length
        flange_radius = r * 1.3

        # Inlet on top of housing
        if p.axis == "x":
            x_center = p.center[0]
            y_base = p.center[1] + p.housing_radius
            y_top = y_base + inlet_length
            z_center = p.center[2]

            # Bottom ring (at housing)
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = x_center + r * np.cos(theta)
                z = z_center + r * np.sin(theta)
                vertices.append([x, y_base, z])
                normals.append([np.cos(theta), 0.0, np.sin(theta)])
            
            # Top ring (neck end)
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = x_center + r * np.cos(theta)
                z = z_center + r * np.sin(theta)
                vertices.append([x, y_top, z])
                normals.append([np.cos(theta), 0.0, np.sin(theta)])
            
            # Flange ring at top
            flange_start = len(vertices)
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = x_center + flange_radius * np.cos(theta)
                z = z_center + flange_radius * np.sin(theta)
                vertices.append([x, y_top, z])
                normals.append([0.0, 1.0, 0.0])
        else:
            for _ in range(n_radial * 3):
                vertices.append([p.center[0], p.center[1], p.center[2]])
                normals.append([0.0, 1.0, 0.0])
            flange_start = start_idx + n_radial * 2

        # Neck cylinder triangles
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = start_idx + j
            v1 = start_idx + j_next
            v2 = start_idx + n_radial + j_next
            v3 = start_idx + n_radial + j

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])
        
        # Flange face
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = start_idx + n_radial + j
            v1 = start_idx + n_radial + j_next
            v2 = flange_start + j_next
            v3 = flange_start + j

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

    def _generate_outlet(self, vertices: List, indices: List, normals: List):
        """Generate circular outlet neck with flange."""
        p = self.params
        n_radial = max(16, p.resolution_radial // 2)

        start_idx = len(vertices)
        r = p.outlet_diameter / 2
        outlet_length = p.outlet_diameter * 0.6  # Neck length
        flange_radius = r * 1.3

        # Outlet at bottom of housing
        if p.axis == "x":
            x_center = p.center[0]
            y_base = p.center[1] - p.housing_radius
            y_bottom = y_base - outlet_length
            z_center = p.center[2]

            # Top ring (at housing)
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = x_center + r * np.cos(theta)
                z = z_center + r * np.sin(theta)
                vertices.append([x, y_base, z])
                normals.append([np.cos(theta), 0.0, np.sin(theta)])
            
            # Bottom ring (outlet end)
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = x_center + r * np.cos(theta)
                z = z_center + r * np.sin(theta)
                vertices.append([x, y_bottom, z])
                normals.append([np.cos(theta), 0.0, np.sin(theta)])
            
            # Flange ring at bottom
            flange_start = len(vertices)
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = x_center + flange_radius * np.cos(theta)
                z = z_center + flange_radius * np.sin(theta)
                vertices.append([x, y_bottom, z])
                normals.append([0.0, -1.0, 0.0])
        else:
            for _ in range(n_radial * 3):
                vertices.append([p.center[0], p.center[1], p.center[2]])
                normals.append([0.0, -1.0, 0.0])
            flange_start = start_idx + n_radial * 2

        # Neck cylinder triangles
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = start_idx + j
            v1 = start_idx + j_next
            v2 = start_idx + n_radial + j_next
            v3 = start_idx + n_radial + j

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])
        
        # Flange face
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = start_idx + n_radial + j
            v1 = start_idx + n_radial + j_next
            v2 = flange_start + j_next
            v3 = flange_start + j

            indices.extend([v0, v2, v1])  # Reversed for facing down
            indices.extend([v0, v3, v2])
            indices.extend([v0, v2, v3])

    def _generate_end_plates(self, vertices: List, indices: List, normals: List):
        """Generate end plates."""
        p = self.params
        n_radial = p.resolution_radial // 2

        half_length = p.housing_length / 2

        for side in [-1, 1]:
            start_idx = len(vertices)

            if p.axis == "x":
                x = p.center[0] + side * half_length
                nx = side
                # Center
                vertices.append([x, p.center[1], p.center[2]])
                normals.append([nx, 0.0, 0.0])
                # Ring
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    y = p.center[1] + p.housing_radius * np.sin(theta)
                    z = p.center[2] + p.housing_radius * np.cos(theta)
                    vertices.append([x, y, z])
                    normals.append([nx, 0.0, 0.0])
            elif p.axis == "y":
                y = p.center[1] + side * half_length
                ny = side
                vertices.append([p.center[0], y, p.center[2]])
                normals.append([0.0, ny, 0.0])
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x = p.center[0] + p.housing_radius * np.cos(theta)
                    z = p.center[2] + p.housing_radius * np.sin(theta)
                    vertices.append([x, y, z])
                    normals.append([0.0, ny, 0.0])
            else:
                z = p.center[2] + side * half_length
                nz = side
                vertices.append([p.center[0], p.center[1], z])
                normals.append([0.0, 0.0, nz])
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x = p.center[0] + p.housing_radius * np.cos(theta)
                    y = p.center[1] + p.housing_radius * np.sin(theta)
                    vertices.append([x, y, z])
                    normals.append([0.0, 0.0, nz])

            # Triangles
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                if side > 0:
                    indices.extend([start_idx, start_idx + 1 + j, start_idx + 1 + j_next])
                else:
                    indices.extend([start_idx, start_idx + 1 + j_next, start_idx + 1 + j])

    def get_tip_speed(self) -> float:
        """Get pin tip speed [m/s]."""
        return self.params.pin_tip_speed

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the de-agglomerator geometry."""
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

    # =========================================================================
    # ANIMATION METHODS
    # =========================================================================
    
    def update_rotation(self, dt: float, rpm: float):
        """
        Update rotor rotation angle based on time step and RPM.
        
        Args:
            dt: Time step [seconds]
            rpm: Rotational speed [RPM]
        """
        omega = rpm * TWO_PI / 60.0  # Convert to rad/s
        self._rotor_angle += omega * dt
        self._rotor_angle = self._rotor_angle % TWO_PI
    
    def get_rotor_angle(self) -> float:
        """Get current rotor rotation angle [radians]."""
        return self._rotor_angle
    
    def set_rotor_angle(self, angle: float):
        """Set rotor rotation angle directly [radians]."""
        self._rotor_angle = angle % TWO_PI
    
    def get_static_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get mesh for static parts (housing, screen, inlet/outlet).
        
        Returns:
            Tuple of (vertices, indices, normals)
        """
        if self._static_vertices is None:
            self._generate_separated_meshes()
        return self._static_vertices, self._static_indices, self._static_normals
    
    def get_rotor_mesh(self, angle: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get mesh for rotor with pins at specified rotation angle.
        
        Args:
            angle: Rotation angle [radians]. Uses current angle if None.
            
        Returns:
            Tuple of (vertices, indices, normals) with rotation applied
        """
        if self._rotor_vertices is None:
            self._generate_separated_meshes()
        
        if angle is None:
            angle = self._rotor_angle
        
        if abs(angle) < 1e-6:
            return self._rotor_vertices.copy(), self._rotor_indices.copy(), self._rotor_normals.copy()
        
        # Apply rotation around X-axis (rotor axis)
        p = self.params
        cx, cy, cz = p.center
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        rotated_vertices = self._rotor_vertices.copy()
        rotated_normals = self._rotor_normals.copy()
        
        for i in range(len(rotated_vertices)):
            # Translate to origin (rotation around X-axis at center)
            y = rotated_vertices[i, 1] - cy
            z = rotated_vertices[i, 2] - cz
            
            # Rotate around X-axis (high speed!)
            new_y = y * cos_a - z * sin_a
            new_z = y * sin_a + z * cos_a
            
            # Translate back
            rotated_vertices[i, 1] = new_y + cy
            rotated_vertices[i, 2] = new_z + cz
            
            # Rotate normals
            ny = rotated_normals[i, 1]
            nz = rotated_normals[i, 2]
            rotated_normals[i, 1] = ny * cos_a - nz * sin_a
            rotated_normals[i, 2] = ny * sin_a + nz * cos_a
        
        return rotated_vertices, self._rotor_indices.copy(), rotated_normals
    
    def _generate_separated_meshes(self):
        """Generate separate meshes for static parts and animated rotor."""
        p = self.params
        
        # Static parts: housing, screen, inlet/outlet
        static_verts = []
        static_indices = []
        static_normals = []
        
        self._generate_housing(static_verts, static_indices, static_normals)
        self._generate_screen(static_verts, static_indices, static_normals)
        self._generate_inlet(static_verts, static_indices, static_normals)
        self._generate_outlet(static_verts, static_indices, static_normals)
        
        self._static_vertices = np.array(static_verts, dtype=np.float32)
        self._static_indices = np.array(static_indices, dtype=np.int32)
        self._static_normals = np.array(static_normals, dtype=np.float32)
        
        # Animated part: rotor with pins
        rotor_verts = []
        rotor_indices = []
        rotor_normals = []
        
        self._generate_rotor(rotor_verts, rotor_indices, rotor_normals)
        
        self._rotor_vertices = np.array(rotor_verts, dtype=np.float32)
        self._rotor_indices = np.array(rotor_indices, dtype=np.int32)
        self._rotor_normals = np.array(rotor_normals, dtype=np.float32)

    @property
    def ports(self) -> Dict[str, ConnectionPort]:
        """
        Get connection ports for this component.
        
        The port positions represent the ACTUAL CONNECTION SURFACES where
        components physically meet (at flange faces).
        
        Returns:
            Dictionary of port name to ConnectionPort:
            - 'inlet': Top inlet for material from feeder (circular with flange)
            - 'outlet': Bottom outlet to classifier/conveyor (circular with flange)
        """
        p = self.params
        
        # Neck lengths (must match _generate_inlet and _generate_outlet)
        inlet_neck_length = p.inlet_diameter * 0.6
        outlet_neck_length = p.outlet_diameter * 0.6
        
        if p.axis == "x":
            # Housing along X, inlet/outlet on Y axis
            inlet_pos = (0.0, p.housing_radius + inlet_neck_length, 0.0)
            inlet_dir = (0.0, 1.0, 0.0)  # Points up
            outlet_pos = (0.0, -(p.housing_radius + outlet_neck_length), 0.0)
            outlet_dir = (0.0, -1.0, 0.0)  # Points down
        elif p.axis == "y":
            # Housing along Y, inlet/outlet on Z axis
            inlet_pos = (0.0, 0.0, p.housing_radius + inlet_neck_length)
            inlet_dir = (0.0, 0.0, 1.0)
            outlet_pos = (0.0, 0.0, -(p.housing_radius + outlet_neck_length))
            outlet_dir = (0.0, 0.0, -1.0)
        else:  # z-axis
            # Housing along Z, inlet/outlet on Y axis
            inlet_pos = (0.0, p.housing_radius + inlet_neck_length, 0.0)
            inlet_dir = (0.0, 1.0, 0.0)
            outlet_pos = (0.0, -(p.housing_radius + outlet_neck_length), 0.0)
            outlet_dir = (0.0, -1.0, 0.0)
        
        return {
            'inlet': ConnectionPort(
                position=inlet_pos,
                direction=inlet_dir,
                diameter=p.inlet_diameter,
                port_type=PortType.FLANGED,
                name="deagglomerator_inlet",
                flange_diameter=p.inlet_diameter * 1.3,
                compatible_types=[PortType.CIRCULAR, PortType.GRAVITY, PortType.FLANGED],
            ),
            'outlet': ConnectionPort(
                position=outlet_pos,
                direction=outlet_dir,
                diameter=p.outlet_diameter,
                port_type=PortType.FLANGED,
                name="deagglomerator_outlet",
                flange_diameter=p.outlet_diameter * 1.3,
                compatible_types=[PortType.CIRCULAR, PortType.GRAVITY, PortType.FLANGED],
            ),
        }


def create_standard_deagglomerator(
    rotor_diameter: float = 0.20,
    screen_aperture: float = 0.002
) -> Deagglomerator:
    """
    Create a standard de-agglomerator / lump breaker.

    Args:
        rotor_diameter: Rotor diameter [m]
        screen_aperture: Screen hole size [m] (e.g., 2mm)

    Returns:
        Deagglomerator instance
    """
    params = DeagglomeratorParams(
        rotor_diameter=rotor_diameter,
        rotor_length=rotor_diameter * 0.6,
        shaft_diameter=rotor_diameter * 0.2,
        num_pin_rows=3,
        pins_per_row=6,
        pin_diameter=rotor_diameter * 0.05,
        pin_length=rotor_diameter * 0.35,
        housing_diameter=rotor_diameter * 1.3,
        housing_length=rotor_diameter * 0.8,
        screen_diameter=rotor_diameter * 1.1,
        screen_aperture=screen_aperture,
        screen_open_area=0.40,
        inlet_diameter=rotor_diameter * 0.4,
        outlet_diameter=rotor_diameter * 0.5,
        rpm=1500,
    )

    return Deagglomerator(params)
