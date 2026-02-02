"""
Flow control damper component for air flow regulation.

Dampers provide control over air flow rate and pressure in the
classification system. They can be used for isolation, flow control,
or pressure balancing.

Types:
- Butterfly: Single blade, simple, good for on/off or throttling
- Louver/Blade: Multiple parallel blades, better flow characteristics
- Iris: Radial leaves, good flow profile, expensive
"""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import warp as wp

from ...utils.constants import PI, TWO_PI


@dataclass
class DamperParams:
    """Parameters for flow control damper."""

    # Basic geometry
    diameter: float              # [m] Duct/damper diameter
    damper_type: str = "butterfly"  # "butterfly", "louver", "iris"

    # Butterfly specific
    blade_thickness: float = 0.003   # [m] Blade thickness

    # Louver specific
    num_blades: int = 4          # Number of blades (for louver type)
    blade_overlap: float = 0.1   # Overlap fraction between blades

    # Housing
    housing_length: float = None  # [m] Housing length (auto-calculated if None)
    flange_width: float = 0.03   # [m] Flange width

    # Actuator
    actuator_type: str = "manual"  # "manual", "pneumatic", "electric"
    actuator_size: float = 0.08    # [m] Actuator housing size

    # Current position
    position: float = 0.0        # Position 0=closed, 1=fully open

    # Position
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: str = "x"              # Flow axis direction

    # Mesh resolution
    resolution: int = 24

    def __post_init__(self):
        if self.housing_length is None:
            self.housing_length = self.diameter * 0.5

    @property
    def radius(self) -> float:
        """Damper radius."""
        return self.diameter / 2

    @property
    def flow_area_open(self) -> float:
        """Full open flow area [m²]."""
        return PI * self.radius ** 2

    def flow_area(self, position: float = None) -> float:
        """
        Effective flow area at given position.

        Args:
            position: Damper position (0=closed, 1=open)

        Returns:
            Effective flow area [m²]
        """
        pos = position if position is not None else self.position

        if self.damper_type == "butterfly":
            # Butterfly: area ~ sin(angle)
            angle = pos * PI / 2  # 0 to 90 degrees
            return self.flow_area_open * np.sin(angle)
        elif self.damper_type == "iris":
            # Iris: area ~ position^2 (radial)
            return self.flow_area_open * pos ** 2
        else:  # louver
            # Louver: approximately linear
            return self.flow_area_open * pos * 0.95

    def cv(self, position: float = None) -> float:
        """
        Flow coefficient at given position.

        Args:
            position: Damper position

        Returns:
            Flow coefficient Cv
        """
        # Cv ~ A * sqrt(1/k) where k is loss coefficient
        area = self.flow_area(position)
        return area * 27.3  # Approximate conversion


class FlowDamper:
    """
    Flow control damper for air flow regulation.

    Components:
    - Cylindrical housing (static)
    - Damper blade(s) (ANIMATED - rotates based on position)
    - Actuator housing (static)
    - Flanges (static)

    Coordinate system:
    - Origin at center of damper
    - Flow along specified axis
    
    Animation:
    - Butterfly blade rotates from 0° (closed) to 90° (open)
    - Use get_blade_mesh() for animated blade
    - Use get_static_mesh() for non-moving parts
    """

    def __init__(self, params: DamperParams):
        """
        Initialize flow damper.

        Args:
            params: DamperParams defining the geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None
        
        # Animation state
        self._current_position = params.position  # 0=closed, 1=open
        self._target_position = params.position
        
        # Cached separate meshes for animation
        self._static_vertices = None
        self._static_indices = None
        self._static_normals = None
        self._blade_base_vertices = None  # Blade at position=0
        self._blade_indices = None
        self._blade_base_normals = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for the flow damper.

        Returns:
            Tuple of (vertices, indices, normals)
        """
        p = self.params
        vertices = []
        indices = []
        normals = []

        # Generate housing cylinder
        self._generate_housing(vertices, indices, normals)

        # Generate flanges
        self._generate_flanges(vertices, indices, normals)

        # Generate blade(s) based on type
        if p.damper_type == "butterfly":
            self._generate_butterfly_blade(vertices, indices, normals)
        elif p.damper_type == "louver":
            self._generate_louver_blades(vertices, indices, normals)
        else:  # iris
            self._generate_iris_blades(vertices, indices, normals)

        # Generate actuator
        self._generate_actuator(vertices, indices, normals)

        self._vertices = np.array(vertices, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.int32)
        self._normals = np.array(normals, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def _generate_housing(self, vertices: List, indices: List, normals: List):
        """Generate cylindrical housing."""
        p = self.params
        n_radial = p.resolution
        n_axial = 4

        start_idx = len(vertices)
        r = p.radius + 0.005  # Slightly larger than blade
        half_length = p.housing_length / 2

        for i in range(n_axial + 1):
            t = i / n_axial
            axial_pos = -half_length + t * p.housing_length

            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI

                if p.axis == "x":
                    x = p.center[0] + axial_pos
                    y = p.center[1] + r * np.sin(theta)
                    z = p.center[2] + r * np.cos(theta)
                    nx, ny, nz = 0, np.sin(theta), np.cos(theta)
                elif p.axis == "y":
                    x = p.center[0] + r * np.cos(theta)
                    y = p.center[1] + axial_pos
                    z = p.center[2] + r * np.sin(theta)
                    nx, ny, nz = np.cos(theta), 0, np.sin(theta)
                else:  # z
                    x = p.center[0] + r * np.cos(theta)
                    y = p.center[1] + r * np.sin(theta)
                    z = p.center[2] + axial_pos
                    nx, ny, nz = np.cos(theta), np.sin(theta), 0

                vertices.append([x, y, z])
                normals.append([nx, ny, nz])

        # Triangles
        for i in range(n_axial):
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = start_idx + i * n_radial + j
                v1 = start_idx + i * n_radial + j_next
                v2 = start_idx + (i + 1) * n_radial + j_next
                v3 = start_idx + (i + 1) * n_radial + j

                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

    def _generate_flanges(self, vertices: List, indices: List, normals: List):
        """Generate inlet and outlet flanges."""
        p = self.params
        n_radial = p.resolution // 2

        r_inner = p.radius + 0.005
        r_outer = p.radius + p.flange_width
        flange_thickness = 0.01

        half_length = p.housing_length / 2

        for side in [-1, 1]:
            start_idx = len(vertices)

            if p.axis == "x":
                x_pos = p.center[0] + side * half_length
                for i in range(2):
                    r = r_inner if i == 0 else r_outer
                    for j in range(n_radial):
                        theta = (j / n_radial) * TWO_PI
                        vertices.append([x_pos, p.center[1] + r * np.sin(theta),
                                       p.center[2] + r * np.cos(theta)])
                        normals.append([side, 0, 0])
            elif p.axis == "y":
                y_pos = p.center[1] + side * half_length
                for i in range(2):
                    r = r_inner if i == 0 else r_outer
                    for j in range(n_radial):
                        theta = (j / n_radial) * TWO_PI
                        vertices.append([p.center[0] + r * np.cos(theta), y_pos,
                                       p.center[2] + r * np.sin(theta)])
                        normals.append([0, side, 0])
            else:  # z
                z_pos = p.center[2] + side * half_length
                for i in range(2):
                    r = r_inner if i == 0 else r_outer
                    for j in range(n_radial):
                        theta = (j / n_radial) * TWO_PI
                        vertices.append([p.center[0] + r * np.cos(theta),
                                       p.center[1] + r * np.sin(theta), z_pos])
                        normals.append([0, 0, side])

            # Triangles for flange (annular ring)
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = start_idx + j
                v1 = start_idx + j_next
                v2 = start_idx + n_radial + j_next
                v3 = start_idx + n_radial + j

                if side > 0:
                    indices.extend([v0, v1, v2])
                    indices.extend([v0, v2, v3])
                else:
                    indices.extend([v0, v2, v1])
                    indices.extend([v0, v3, v2])

    def _generate_butterfly_blade(self, vertices: List, indices: List, normals: List):
        """
        Generate butterfly damper blade as a solid disc.

        Real butterfly valve blade:
        - Single solid circular disc (3-6mm thick)
        - Mounted on central shaft
        - Rotates 0° (closed) to 90° (open)

        Geometry includes:
        - Front face (fan triangles)
        - Back face (fan triangles)
        - Edge rim (connecting front/back) - makes it look solid
        """
        p = self.params

        # Blade is a circular disk that rotates
        # At position 0 (closed), blade is perpendicular to flow
        # At position 1 (open), blade is parallel to flow

        blade_angle = p.position * PI / 2  # 0 to 90 degrees
        r = p.radius - 0.002  # Slightly smaller than housing
        half_thick = p.blade_thickness / 2

        n_radial = p.resolution // 2

        if p.axis != "x":
            # Simplified for non-X axes
            self._generate_blade_simple(vertices, indices, normals)
            return

        cx, cy, cz = p.center[0], p.center[1], p.center[2]

        # Pre-calculate rotation matrix components
        cos_a = np.cos(blade_angle)
        sin_a = np.sin(blade_angle)

        # Store edge vertex indices for rim generation
        front_edge_start = len(vertices)

        # ============================================================
        # 1. FRONT FACE (side = +1)
        # ============================================================
        front_center = len(vertices)

        # Center vertex (front)
        vertices.append([
            cx + half_thick * cos_a,
            cy,
            cz + half_thick * sin_a
        ])
        normals.append([cos_a, 0, sin_a])

        # Edge vertices (front)
        for j in range(n_radial):
            theta = (j / n_radial) * TWO_PI
            y = cy + r * np.sin(theta)
            z_local = r * np.cos(theta)

            # Apply rotation and thickness offset
            x = cx + z_local * sin_a + half_thick * cos_a
            z = cz + z_local * cos_a + half_thick * sin_a

            vertices.append([x, y, z])
            normals.append([cos_a, 0, sin_a])

        # Front face triangles (fan from center)
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            indices.extend([front_center, front_center + 1 + j, front_center + 1 + j_next])

        # ============================================================
        # 2. BACK FACE (side = -1)
        # ============================================================
        back_center = len(vertices)

        # Center vertex (back)
        vertices.append([
            cx - half_thick * cos_a,
            cy,
            cz - half_thick * sin_a
        ])
        normals.append([-cos_a, 0, -sin_a])

        # Edge vertices (back)
        back_edge_start = len(vertices)
        for j in range(n_radial):
            theta = (j / n_radial) * TWO_PI
            y = cy + r * np.sin(theta)
            z_local = r * np.cos(theta)

            # Apply rotation and thickness offset (negative for back)
            x = cx + z_local * sin_a - half_thick * cos_a
            z = cz + z_local * cos_a - half_thick * sin_a

            vertices.append([x, y, z])
            normals.append([-cos_a, 0, -sin_a])

        # Back face triangles (fan from center, reversed winding)
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            indices.extend([back_center, back_center + 1 + j_next, back_center + 1 + j])

        # ============================================================
        # 3. EDGE RIM (connects front and back faces)
        # ============================================================
        # This makes the blade look like a solid disc, not two floating circles
        front_edge_start = front_center + 1  # First edge vertex of front face

        for j in range(n_radial):
            j_next = (j + 1) % n_radial

            # Four vertices of this edge quad
            f0 = front_edge_start + j       # Front edge vertex j
            f1 = front_edge_start + j_next  # Front edge vertex j+1
            b0 = back_edge_start + j        # Back edge vertex j
            b1 = back_edge_start + j_next   # Back edge vertex j+1

            # Edge normal points radially outward
            theta = ((j + 0.5) / n_radial) * TWO_PI
            edge_ny = np.sin(theta)
            edge_nz_local = np.cos(theta)
            # Rotate edge normal by blade angle
            edge_nx = edge_nz_local * sin_a
            edge_nz = edge_nz_local * cos_a

            # Add edge quad vertices with radial normals
            rim_start = len(vertices)

            # Get positions from existing vertices
            vf0 = vertices[f0]
            vf1 = vertices[f1]
            vb0 = vertices[b0]
            vb1 = vertices[b1]

            vertices.append(vf0[:])
            normals.append([edge_nx, edge_ny, edge_nz])
            vertices.append(vf1[:])
            normals.append([edge_nx, edge_ny, edge_nz])
            vertices.append(vb1[:])
            normals.append([edge_nx, edge_ny, edge_nz])
            vertices.append(vb0[:])
            normals.append([edge_nx, edge_ny, edge_nz])

            # Two triangles for this edge quad
            indices.extend([rim_start, rim_start + 1, rim_start + 2])
            indices.extend([rim_start, rim_start + 2, rim_start + 3])

        # Blade shaft (pivot rod)
        self._add_blade_shaft(vertices, indices, normals)

    def _generate_blade_simple(self, vertices: List, indices: List, normals: List):
        """Simplified blade for non-X axis orientations."""
        p = self.params
        r = p.radius - 0.002
        n_radial = p.resolution // 2

        for side in [-1, 1]:
            blade_side_start = len(vertices)
            vertices.append([p.center[0], p.center[1], p.center[2]])
            normals.append([0, 0, side])

            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                vertices.append([
                    p.center[0] + r * np.cos(theta),
                    p.center[1] + r * np.sin(theta),
                    p.center[2]
                ])
                normals.append([0, 0, side])

            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                if side > 0:
                    indices.extend([blade_side_start, blade_side_start + 1 + j,
                                  blade_side_start + 1 + j_next])
                else:
                    indices.extend([blade_side_start, blade_side_start + 1 + j_next,
                                  blade_side_start + 1 + j])

    def _add_blade_shaft(self, vertices: List, indices: List, normals: List):
        """Add blade pivot shaft."""
        p = self.params

        shaft_start = len(vertices)
        shaft_radius = 0.008
        shaft_length = p.diameter * 0.6

        n_shaft = 8

        # Shaft along Y axis (perpendicular to blade and flow)
        for i in range(2):
            y = p.center[1] - shaft_length / 2 + i * shaft_length
            for j in range(n_shaft):
                theta = (j / n_shaft) * TWO_PI
                if p.axis == "x":
                    x = p.center[0] + shaft_radius * np.cos(theta)
                    z = p.center[2] + shaft_radius * np.sin(theta)
                    vertices.append([x, y, z])
                    normals.append([np.cos(theta), 0, np.sin(theta)])
                else:
                    vertices.append([p.center[0], y, p.center[2]])
                    normals.append([1, 0, 0])

        # Triangles
        for j in range(n_shaft):
            j_next = (j + 1) % n_shaft
            v0 = shaft_start + j
            v1 = shaft_start + j_next
            v2 = shaft_start + n_shaft + j_next
            v3 = shaft_start + n_shaft + j

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

    def _generate_louver_blades(self, vertices: List, indices: List, normals: List):
        """Generate multiple parallel louver blades."""
        p = self.params

        # Louver blades are rectangular, arranged parallel across the duct
        blade_spacing = p.diameter / (p.num_blades + 1)
        blade_width = blade_spacing * (1 + p.blade_overlap)
        blade_angle = p.position * PI / 2

        for blade_idx in range(p.num_blades):
            start_idx = len(vertices)

            # Position of this blade along diameter
            blade_offset = -p.diameter / 2 + (blade_idx + 1) * blade_spacing

            hw = blade_width / 2
            hh = p.radius - 0.005  # Blade height (radial extent)
            half_thick = p.blade_thickness / 2

            # Blade rotates around its long axis
            if p.axis == "x":
                y_center = p.center[1] + blade_offset

                # 4 corners x 2 sides = 8 vertices per blade
                for side in [-1, 1]:
                    for corner in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
                        local_x, local_z = corner

                        # Rotate blade around Y axis
                        x = p.center[0] + local_x * np.cos(blade_angle) + side * half_thick * np.sin(blade_angle)
                        z = p.center[2] + local_z

                        vertices.append([x, y_center, z])
                        normals.append([np.sin(blade_angle) * side, 0, np.cos(blade_angle) * side])

            else:
                # Simplified
                for _ in range(8):
                    vertices.append([p.center[0], p.center[1], p.center[2]])
                    normals.append([0, 0, 1])

            # Triangles for blade faces
            # Front face
            indices.extend([start_idx, start_idx + 1, start_idx + 2])
            indices.extend([start_idx, start_idx + 2, start_idx + 3])
            # Back face
            indices.extend([start_idx + 4, start_idx + 6, start_idx + 5])
            indices.extend([start_idx + 4, start_idx + 7, start_idx + 6])

    def _generate_iris_blades(self, vertices: List, indices: List, normals: List):
        """Generate iris damper with radial leaves."""
        p = self.params

        # Iris has overlapping leaves that open radially
        num_leaves = 8
        leaf_angle_span = TWO_PI / num_leaves * 1.3  # Overlap

        # Opening is controlled by rotating leaves
        opening_radius = p.radius * p.position

        for leaf_idx in range(num_leaves):
            start_idx = len(vertices)

            base_angle = (leaf_idx / num_leaves) * TWO_PI

            # Each leaf is a curved sector
            r_inner = opening_radius * 0.2
            r_outer = p.radius

            if p.axis == "x":
                for r_idx, r in enumerate([r_inner, r_outer]):
                    for a_idx in range(3):
                        angle = base_angle + (a_idx / 2) * leaf_angle_span
                        y = p.center[1] + r * np.sin(angle)
                        z = p.center[2] + r * np.cos(angle)
                        vertices.append([p.center[0], y, z])
                        normals.append([1, 0, 0])
            else:
                for _ in range(6):
                    vertices.append([p.center[0], p.center[1], p.center[2]])
                    normals.append([0, 0, 1])

            # Triangles for leaf
            indices.extend([start_idx, start_idx + 1, start_idx + 4])
            indices.extend([start_idx, start_idx + 4, start_idx + 3])
            indices.extend([start_idx + 1, start_idx + 2, start_idx + 5])
            indices.extend([start_idx + 1, start_idx + 5, start_idx + 4])

    def _generate_actuator(self, vertices: List, indices: List, normals: List):
        """
        Generate actuator assembly with proper mounting.

        Components:
        1. Mounting bracket - bolts to damper housing flange
        2. Drive stem - cylindrical connection from bracket to actuator
        3. Actuator body - main housing (box for electric, cylinder for pneumatic)
        4. Connected to blade shaft via internal gearing

        Layout (side view):
                    ┌─────────┐
                    │ ACTUATOR│ ← Main body
                    │  BODY   │
                    └────┬────┘
                         │     ← Drive stem
                    ┌────┴────┐
                    │ BRACKET │ ← Mounting bracket
            ════════╧═════════╧════════
                 DAMPER HOUSING
        """
        p = self.params

        if p.axis != "x":
            # Simplified for non-X axis (fallback)
            self._generate_actuator_simple(vertices, indices, normals)
            return

        # Actuator sizing based on damper diameter
        # Larger dampers need larger actuators for torque
        actuator_width = p.actuator_size
        actuator_depth = p.actuator_size * 0.8
        actuator_height = p.actuator_size * 0.6

        # Mounting bracket dimensions
        bracket_width = p.diameter * 0.4  # 40% of damper diameter
        bracket_depth = 0.025  # 25mm thick
        bracket_height = 0.015  # 15mm tall

        # Drive stem dimensions
        stem_diameter = 0.025  # 25mm diameter
        stem_height = 0.03  # 30mm tall

        # Positions
        x_center = p.center[0]
        z_center = p.center[2]
        housing_top_y = p.center[1] + p.radius + 0.005  # Top of housing

        # ============================================================
        # 1. MOUNTING BRACKET (sits on top of housing)
        # ============================================================
        self._generate_mounting_bracket(
            vertices, indices, normals,
            x_center, housing_top_y, z_center,
            bracket_width, bracket_depth, bracket_height
        )

        # ============================================================
        # 2. DRIVE STEM (cylindrical connection)
        # ============================================================
        stem_base_y = housing_top_y + bracket_height
        self._generate_drive_stem(
            vertices, indices, normals,
            x_center, stem_base_y, z_center,
            stem_diameter, stem_height
        )

        # ============================================================
        # 3. ACTUATOR BODY (main housing)
        # ============================================================
        body_base_y = stem_base_y + stem_height

        if p.actuator_type == "pneumatic":
            # Cylindrical body for pneumatic actuator (scotch yoke type)
            self._generate_pneumatic_actuator(
                vertices, indices, normals,
                x_center, body_base_y, z_center,
                actuator_width * 0.5, actuator_height
            )
        elif p.actuator_type == "electric":
            # Electric actuator with motor housing
            self._generate_electric_actuator(
                vertices, indices, normals,
                x_center, body_base_y, z_center,
                actuator_width, actuator_depth, actuator_height
            )
        else:
            # Manual actuator with handwheel
            self._generate_manual_actuator(
                vertices, indices, normals,
                x_center, body_base_y, z_center,
                actuator_width, actuator_depth, actuator_height
            )

    def _generate_mounting_bracket(self, vertices: List, indices: List, normals: List,
                                    cx: float, cy: float, cz: float,
                                    width: float, depth: float, height: float):
        """Generate mounting bracket that sits on damper housing."""
        start_idx = len(vertices)

        hw = width / 2
        hd = depth / 2

        # Bracket is a rectangular plate with curved sides to match housing
        # Simplified as a box for now
        corners = [
            # Bottom face (sits on housing)
            [cx - hw, cy, cz - hd],
            [cx + hw, cy, cz - hd],
            [cx + hw, cy, cz + hd],
            [cx - hw, cy, cz + hd],
            # Top face
            [cx - hw, cy + height, cz - hd],
            [cx + hw, cy + height, cz - hd],
            [cx + hw, cy + height, cz + hd],
            [cx - hw, cy + height, cz + hd],
        ]

        # Add vertices with proper normals
        face_normals = [
            [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0],  # Bottom
            [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],      # Top
        ]

        for i, corner in enumerate(corners):
            vertices.append(corner)
            normals.append(face_normals[i])

        # Faces
        # Bottom
        indices.extend([start_idx + 0, start_idx + 2, start_idx + 1])
        indices.extend([start_idx + 0, start_idx + 3, start_idx + 2])
        # Top
        indices.extend([start_idx + 4, start_idx + 5, start_idx + 6])
        indices.extend([start_idx + 4, start_idx + 6, start_idx + 7])
        # Sides
        indices.extend([start_idx + 0, start_idx + 1, start_idx + 5])
        indices.extend([start_idx + 0, start_idx + 5, start_idx + 4])
        indices.extend([start_idx + 2, start_idx + 3, start_idx + 7])
        indices.extend([start_idx + 2, start_idx + 7, start_idx + 6])
        indices.extend([start_idx + 1, start_idx + 2, start_idx + 6])
        indices.extend([start_idx + 1, start_idx + 6, start_idx + 5])
        indices.extend([start_idx + 0, start_idx + 4, start_idx + 7])
        indices.extend([start_idx + 0, start_idx + 7, start_idx + 3])

    def _generate_drive_stem(self, vertices: List, indices: List, normals: List,
                              cx: float, cy: float, cz: float,
                              diameter: float, height: float):
        """Generate cylindrical drive stem connecting bracket to actuator body."""
        start_idx = len(vertices)

        n_segments = 12
        radius = diameter / 2

        # Generate cylinder vertices
        for i in range(2):  # Bottom and top rings
            y = cy + i * height
            for j in range(n_segments):
                theta = (j / n_segments) * TWO_PI
                x = cx + radius * np.cos(theta)
                z = cz + radius * np.sin(theta)
                vertices.append([x, y, z])
                normals.append([np.cos(theta), 0, np.sin(theta)])

        # Side triangles
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            v0 = start_idx + j
            v1 = start_idx + j_next
            v2 = start_idx + n_segments + j_next
            v3 = start_idx + n_segments + j

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

    def _generate_actuator_box(self, vertices: List, indices: List, normals: List,
                                cx: float, cy: float, cz: float,
                                width: float, depth: float, height: float):
        """Generate box-shaped actuator body (for electric/manual actuators)."""
        start_idx = len(vertices)

        hw = width / 2
        hd = depth / 2

        corners = [
            # Bottom face
            [cx - hw, cy, cz - hd],
            [cx + hw, cy, cz - hd],
            [cx + hw, cy, cz + hd],
            [cx - hw, cy, cz + hd],
            # Top face
            [cx - hw, cy + height, cz - hd],
            [cx + hw, cy + height, cz - hd],
            [cx + hw, cy + height, cz + hd],
            [cx - hw, cy + height, cz + hd],
        ]

        for corner in corners:
            vertices.append(corner)
            normals.append([0, 1, 0])  # Simplified normals

        # All 6 faces
        # Bottom
        indices.extend([start_idx + 0, start_idx + 2, start_idx + 1])
        indices.extend([start_idx + 0, start_idx + 3, start_idx + 2])
        # Top
        indices.extend([start_idx + 4, start_idx + 5, start_idx + 6])
        indices.extend([start_idx + 4, start_idx + 6, start_idx + 7])
        # Front (-Z)
        indices.extend([start_idx + 0, start_idx + 1, start_idx + 5])
        indices.extend([start_idx + 0, start_idx + 5, start_idx + 4])
        # Back (+Z)
        indices.extend([start_idx + 2, start_idx + 3, start_idx + 7])
        indices.extend([start_idx + 2, start_idx + 7, start_idx + 6])
        # Right (+X)
        indices.extend([start_idx + 1, start_idx + 2, start_idx + 6])
        indices.extend([start_idx + 1, start_idx + 6, start_idx + 5])
        # Left (-X)
        indices.extend([start_idx + 0, start_idx + 4, start_idx + 7])
        indices.extend([start_idx + 0, start_idx + 7, start_idx + 3])

    def _generate_actuator_cylinder(self, vertices: List, indices: List, normals: List,
                                     cx: float, cy: float, cz: float,
                                     radius: float, height: float):
        """Generate cylindrical actuator body (for pneumatic actuators)."""
        start_idx = len(vertices)

        n_segments = 16

        # Generate cylinder vertices
        for i in range(2):  # Bottom and top rings
            y = cy + i * height
            for j in range(n_segments):
                theta = (j / n_segments) * TWO_PI
                x = cx + radius * np.cos(theta)
                z = cz + radius * np.sin(theta)
                vertices.append([x, y, z])
                normals.append([np.cos(theta), 0, np.sin(theta)])

        # Add center points for caps
        vertices.append([cx, cy, cz])  # Bottom center
        normals.append([0, -1, 0])
        vertices.append([cx, cy + height, cz])  # Top center
        normals.append([0, 1, 0])

        bottom_center = start_idx + 2 * n_segments
        top_center = start_idx + 2 * n_segments + 1

        # Side triangles
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            v0 = start_idx + j
            v1 = start_idx + j_next
            v2 = start_idx + n_segments + j_next
            v3 = start_idx + n_segments + j

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

        # Bottom cap
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            indices.extend([bottom_center, start_idx + j_next, start_idx + j])

        # Top cap
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            indices.extend([top_center, start_idx + n_segments + j, start_idx + n_segments + j_next])

    def _generate_actuator_simple(self, vertices: List, indices: List, normals: List):
        """Fallback simple actuator for non-X axis orientations."""
        p = self.params
        start_idx = len(vertices)

        size = p.actuator_size
        hs = size / 2

        # Simple box above center
        y_base = p.center[1] + p.radius + 0.02

        corners = [
            [p.center[0] - hs, y_base, p.center[2] - hs],
            [p.center[0] + hs, y_base, p.center[2] - hs],
            [p.center[0] + hs, y_base, p.center[2] + hs],
            [p.center[0] - hs, y_base, p.center[2] + hs],
            [p.center[0] - hs, y_base + size, p.center[2] - hs],
            [p.center[0] + hs, y_base + size, p.center[2] - hs],
            [p.center[0] + hs, y_base + size, p.center[2] + hs],
            [p.center[0] - hs, y_base + size, p.center[2] + hs],
        ]

        for corner in corners:
            vertices.append(corner)
            normals.append([0, 1, 0])

        # Faces
        indices.extend([start_idx + 0, start_idx + 1, start_idx + 2])
        indices.extend([start_idx + 0, start_idx + 2, start_idx + 3])
        indices.extend([start_idx + 4, start_idx + 7, start_idx + 6])
        indices.extend([start_idx + 4, start_idx + 6, start_idx + 5])
        indices.extend([start_idx + 0, start_idx + 4, start_idx + 5])
        indices.extend([start_idx + 0, start_idx + 5, start_idx + 1])
        indices.extend([start_idx + 2, start_idx + 6, start_idx + 7])
        indices.extend([start_idx + 2, start_idx + 7, start_idx + 3])
        indices.extend([start_idx + 1, start_idx + 5, start_idx + 6])
        indices.extend([start_idx + 1, start_idx + 6, start_idx + 2])
        indices.extend([start_idx + 0, start_idx + 3, start_idx + 7])
        indices.extend([start_idx + 0, start_idx + 7, start_idx + 4])

    def _generate_electric_actuator(self, vertices: List, indices: List, normals: List,
                                     cx: float, cy: float, cz: float,
                                     width: float, depth: float, height: float):
        """
        Generate electric actuator with motor housing.

        Layout (side view):
                ┌───┐
                │MOT│  ← Motor cylinder
                │OR │
            ┌───┴───┴───┐
            │  GEARBOX  │  ← Main housing with internal gearing
            └───────────┘

        Components:
        1. Main gearbox housing (rectangular)
        2. Motor cylinder on top
        3. Position indicator on front
        """
        # 1. Gearbox housing (main box)
        self._generate_actuator_box(
            vertices, indices, normals,
            cx, cy, cz,
            width, depth, height
        )

        # 2. Motor cylinder on top of gearbox
        motor_radius = width * 0.25  # Motor diameter = 50% of box width
        motor_height = height * 0.8  # Motor height
        motor_cy = cy + height  # Motor sits on top of gearbox

        self._generate_motor_cylinder(
            vertices, indices, normals,
            cx, motor_cy, cz,
            motor_radius, motor_height
        )

        # 3. Position indicator plate on front (-Z face)
        indicator_width = width * 0.4
        indicator_height = height * 0.3
        indicator_depth = 0.005  # 5mm thick plate
        indicator_cy = cy + height * 0.6  # Centered vertically on front face
        indicator_cz = cz - depth / 2 - indicator_depth / 2

        self._generate_indicator_plate(
            vertices, indices, normals,
            cx, indicator_cy, indicator_cz,
            indicator_width, indicator_height, indicator_depth
        )

    def _generate_motor_cylinder(self, vertices: List, indices: List, normals: List,
                                  cx: float, cy: float, cz: float,
                                  radius: float, height: float):
        """Generate motor cylinder for electric actuator."""
        start_idx = len(vertices)
        n_segments = 16

        # Cylinder body
        for i in range(2):
            y = cy + i * height
            for j in range(n_segments):
                theta = (j / n_segments) * TWO_PI
                x = cx + radius * np.cos(theta)
                z = cz + radius * np.sin(theta)
                vertices.append([x, y, z])
                normals.append([np.cos(theta), 0, np.sin(theta)])

        # Center points for caps
        vertices.append([cx, cy, cz])  # Bottom center
        normals.append([0, -1, 0])
        vertices.append([cx, cy + height, cz])  # Top center
        normals.append([0, 1, 0])

        bottom_center = start_idx + 2 * n_segments
        top_center = start_idx + 2 * n_segments + 1

        # Side triangles
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            v0 = start_idx + j
            v1 = start_idx + j_next
            v2 = start_idx + n_segments + j_next
            v3 = start_idx + n_segments + j
            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

        # Bottom cap
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            indices.extend([bottom_center, start_idx + j_next, start_idx + j])

        # Top cap
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            indices.extend([top_center, start_idx + n_segments + j, start_idx + n_segments + j_next])

    def _generate_indicator_plate(self, vertices: List, indices: List, normals: List,
                                   cx: float, cy: float, cz: float,
                                   width: float, height: float, depth: float):
        """Generate position indicator plate on front of actuator."""
        start_idx = len(vertices)
        hw = width / 2
        hh = height / 2
        hd = depth / 2

        # Simple box for indicator
        corners = [
            [cx - hw, cy - hh, cz - hd],
            [cx + hw, cy - hh, cz - hd],
            [cx + hw, cy + hh, cz - hd],
            [cx - hw, cy + hh, cz - hd],
            [cx - hw, cy - hh, cz + hd],
            [cx + hw, cy - hh, cz + hd],
            [cx + hw, cy + hh, cz + hd],
            [cx - hw, cy + hh, cz + hd],
        ]

        for corner in corners:
            vertices.append(corner)
            normals.append([0, 0, -1])  # Front-facing

        # Front face (-Z)
        indices.extend([start_idx + 0, start_idx + 3, start_idx + 2])
        indices.extend([start_idx + 0, start_idx + 2, start_idx + 1])
        # Back face (+Z)
        indices.extend([start_idx + 4, start_idx + 5, start_idx + 6])
        indices.extend([start_idx + 4, start_idx + 6, start_idx + 7])
        # Top, bottom, sides
        indices.extend([start_idx + 2, start_idx + 3, start_idx + 7])
        indices.extend([start_idx + 2, start_idx + 7, start_idx + 6])
        indices.extend([start_idx + 0, start_idx + 1, start_idx + 5])
        indices.extend([start_idx + 0, start_idx + 5, start_idx + 4])
        indices.extend([start_idx + 1, start_idx + 2, start_idx + 6])
        indices.extend([start_idx + 1, start_idx + 6, start_idx + 5])
        indices.extend([start_idx + 0, start_idx + 4, start_idx + 7])
        indices.extend([start_idx + 0, start_idx + 7, start_idx + 3])

    def _generate_pneumatic_actuator(self, vertices: List, indices: List, normals: List,
                                      cx: float, cy: float, cz: float,
                                      radius: float, height: float):
        """
        Generate pneumatic actuator with scotch yoke design.

        Layout (top view):
            ┌───┬─────────────┬───┐
            │AIR│  CYLINDER   │AIR│
            │   │   BARREL    │   │
            └───┴─────────────┴───┘
                     ↑
                 Drive stem

        Components:
        1. Main cylinder barrel (horizontal, along X axis)
        2. End caps with air ports
        3. Yoke housing underneath connecting to stem
        """
        # Cylinder is horizontal along X axis (perpendicular to blade shaft)
        barrel_length = radius * 3  # Full stroke length
        barrel_radius = radius * 0.8

        # 1. Main cylinder barrel (horizontal along X)
        self._generate_horizontal_cylinder(
            vertices, indices, normals,
            cx, cy + height * 0.5, cz,  # Raised above base
            barrel_length, barrel_radius
        )

        # 2. End caps (larger diameter discs at each end)
        cap_radius = barrel_radius * 1.2
        cap_thickness = 0.015

        # Left end cap
        self._generate_end_cap(
            vertices, indices, normals,
            cx - barrel_length / 2 - cap_thickness / 2, cy + height * 0.5, cz,
            cap_radius, cap_thickness
        )

        # Right end cap
        self._generate_end_cap(
            vertices, indices, normals,
            cx + barrel_length / 2 + cap_thickness / 2, cy + height * 0.5, cz,
            cap_radius, cap_thickness
        )

        # 3. Yoke housing (box underneath connecting barrel to stem)
        yoke_width = barrel_length * 0.4
        yoke_depth = barrel_radius * 1.5
        yoke_height = height * 0.4

        self._generate_actuator_box(
            vertices, indices, normals,
            cx, cy, cz,
            yoke_width, yoke_depth, yoke_height
        )

    def _generate_horizontal_cylinder(self, vertices: List, indices: List, normals: List,
                                        cx: float, cy: float, cz: float,
                                        length: float, radius: float):
        """Generate horizontal cylinder (axis along X)."""
        start_idx = len(vertices)
        n_segments = 16

        # Cylinder along X axis
        for i in range(2):
            x = cx - length / 2 + i * length
            for j in range(n_segments):
                theta = (j / n_segments) * TWO_PI
                y = cy + radius * np.cos(theta)
                z = cz + radius * np.sin(theta)
                vertices.append([x, y, z])
                normals.append([0, np.cos(theta), np.sin(theta)])

        # Side triangles
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            v0 = start_idx + j
            v1 = start_idx + j_next
            v2 = start_idx + n_segments + j_next
            v3 = start_idx + n_segments + j
            indices.extend([v0, v2, v1])  # Reversed for correct winding
            indices.extend([v0, v3, v2])

    def _generate_end_cap(self, vertices: List, indices: List, normals: List,
                           cx: float, cy: float, cz: float,
                           radius: float, thickness: float):
        """Generate end cap disc for pneumatic cylinder."""
        start_idx = len(vertices)
        n_segments = 12

        # Two rings for disc thickness
        for i in range(2):
            x = cx - thickness / 2 + i * thickness
            nx = -1 if i == 0 else 1
            for j in range(n_segments):
                theta = (j / n_segments) * TWO_PI
                y = cy + radius * np.cos(theta)
                z = cz + radius * np.sin(theta)
                vertices.append([x, y, z])
                normals.append([nx, 0, 0])

        # Center points
        vertices.append([cx - thickness / 2, cy, cz])  # Left center
        normals.append([-1, 0, 0])
        vertices.append([cx + thickness / 2, cy, cz])  # Right center
        normals.append([1, 0, 0])

        left_center = start_idx + 2 * n_segments
        right_center = start_idx + 2 * n_segments + 1

        # Left face (fan triangles)
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            indices.extend([left_center, start_idx + j, start_idx + j_next])

        # Right face (fan triangles)
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            indices.extend([right_center, start_idx + n_segments + j_next, start_idx + n_segments + j])

        # Edge (connecting the two rings)
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            v0 = start_idx + j
            v1 = start_idx + j_next
            v2 = start_idx + n_segments + j_next
            v3 = start_idx + n_segments + j
            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

    def _generate_manual_actuator(self, vertices: List, indices: List, normals: List,
                                   cx: float, cy: float, cz: float,
                                   width: float, depth: float, height: float):
        """
        Generate manual actuator with handwheel.

        Layout (side view):
                  ╭───╮
                 ╱     ╲    ← Handwheel
                ╱   ●   ╲       (spoked wheel)
               ╲       ╱
                ╲     ╱
                 ╰───╯
                   │      ← Stem
            ┌──────┴──────┐
            │   GEARBOX   │  ← Small gear housing
            └─────────────┘

        Components:
        1. Small gear housing box
        2. Handwheel stem
        3. Handwheel (spoked disc)
        """
        # 1. Small gearbox housing
        gear_height = height * 0.6
        self._generate_actuator_box(
            vertices, indices, normals,
            cx, cy, cz,
            width * 0.8, depth * 0.8, gear_height
        )

        # 2. Handwheel stem
        stem_radius = 0.008  # 8mm radius
        stem_height = height * 0.5
        stem_cy = cy + gear_height

        self._generate_drive_stem(
            vertices, indices, normals,
            cx, stem_cy, cz,
            stem_radius * 2, stem_height
        )

        # 3. Handwheel (spoked wheel)
        wheel_radius = width * 0.6
        wheel_cy = stem_cy + stem_height

        self._generate_handwheel(
            vertices, indices, normals,
            cx, wheel_cy, cz,
            wheel_radius
        )

    def _generate_handwheel(self, vertices: List, indices: List, normals: List,
                             cx: float, cy: float, cz: float,
                             radius: float):
        """Generate handwheel with rim and spokes."""
        n_spokes = 5
        rim_thickness = 0.012  # 12mm thick rim
        rim_width = 0.015  # 15mm rim width (radial)
        spoke_radius = 0.006  # 6mm spoke radius
        hub_radius = 0.02  # 20mm hub

        # 1. Outer rim (torus-like, simplified as ring of boxes)
        self._generate_wheel_rim(
            vertices, indices, normals,
            cx, cy, cz,
            radius, rim_width, rim_thickness
        )

        # 2. Hub (small cylinder at center)
        hub_height = rim_thickness * 1.5
        hub_start = len(vertices)
        n_hub = 8

        for i in range(2):
            y = cy - hub_height / 2 + i * hub_height
            for j in range(n_hub):
                theta = (j / n_hub) * TWO_PI
                x = cx + hub_radius * np.cos(theta)
                z = cz + hub_radius * np.sin(theta)
                vertices.append([x, y, z])
                normals.append([np.cos(theta), 0, np.sin(theta)])

        # Hub side triangles
        for j in range(n_hub):
            j_next = (j + 1) % n_hub
            v0 = hub_start + j
            v1 = hub_start + j_next
            v2 = hub_start + n_hub + j_next
            v3 = hub_start + n_hub + j
            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

        # 3. Spokes (connecting hub to rim)
        for spoke_idx in range(n_spokes):
            spoke_angle = (spoke_idx / n_spokes) * TWO_PI
            self._generate_spoke(
                vertices, indices, normals,
                cx, cy, cz,
                spoke_angle, hub_radius, radius - rim_width, spoke_radius
            )

    def _generate_wheel_rim(self, vertices: List, indices: List, normals: List,
                             cx: float, cy: float, cz: float,
                             radius: float, width: float, thickness: float):
        """Generate wheel rim as a segmented ring."""
        n_segments = 24
        inner_r = radius - width
        outer_r = radius

        start_idx = len(vertices)

        # Four rings: inner-bottom, inner-top, outer-bottom, outer-top
        for is_outer in [False, True]:
            r = outer_r if is_outer else inner_r
            for is_top in [False, True]:
                y = cy - thickness / 2 + (thickness if is_top else 0)
                for j in range(n_segments):
                    theta = (j / n_segments) * TWO_PI
                    x = cx + r * np.cos(theta)
                    z = cz + r * np.sin(theta)
                    vertices.append([x, y, z])
                    # Normal points outward for outer, inward for inner
                    if is_outer:
                        normals.append([np.cos(theta), 0, np.sin(theta)])
                    else:
                        normals.append([-np.cos(theta), 0, -np.sin(theta)])

        # Indices for each ring
        inner_bottom = start_idx
        inner_top = start_idx + n_segments
        outer_bottom = start_idx + 2 * n_segments
        outer_top = start_idx + 3 * n_segments

        # Outer surface
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            v0 = outer_bottom + j
            v1 = outer_bottom + j_next
            v2 = outer_top + j_next
            v3 = outer_top + j
            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

        # Inner surface (reversed winding)
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            v0 = inner_bottom + j
            v1 = inner_bottom + j_next
            v2 = inner_top + j_next
            v3 = inner_top + j
            indices.extend([v0, v2, v1])
            indices.extend([v0, v3, v2])

        # Top face (annular ring)
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            v0 = inner_top + j
            v1 = inner_top + j_next
            v2 = outer_top + j_next
            v3 = outer_top + j
            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

        # Bottom face (annular ring, reversed)
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            v0 = inner_bottom + j
            v1 = inner_bottom + j_next
            v2 = outer_bottom + j_next
            v3 = outer_bottom + j
            indices.extend([v0, v2, v1])
            indices.extend([v0, v3, v2])

    def _generate_spoke(self, vertices: List, indices: List, normals: List,
                         cx: float, cy: float, cz: float,
                         angle: float, inner_r: float, outer_r: float, spoke_r: float):
        """Generate a single spoke of the handwheel."""
        start_idx = len(vertices)
        n_segments = 6

        # Spoke runs from inner_r to outer_r at given angle
        spoke_start_x = cx + inner_r * np.cos(angle)
        spoke_start_z = cz + inner_r * np.sin(angle)
        spoke_end_x = cx + outer_r * np.cos(angle)
        spoke_end_z = cz + outer_r * np.sin(angle)

        # Generate cylinder along spoke direction
        spoke_dir_x = np.cos(angle)
        spoke_dir_z = np.sin(angle)
        spoke_length = outer_r - inner_r

        # Two rings of vertices
        for i in range(2):
            t = i  # 0 at start, 1 at end
            sx = spoke_start_x + t * (spoke_end_x - spoke_start_x)
            sz = spoke_start_z + t * (spoke_end_z - spoke_start_z)

            for j in range(n_segments):
                theta = (j / n_segments) * TWO_PI
                # Perpendicular to spoke direction
                # For Y-axis aligned wheel, spoke in XZ plane
                # Normal perpendicular: (-sin(angle), 0, cos(angle)) and (0, 1, 0)
                perp_x = -np.sin(angle) * np.cos(theta)
                perp_y = np.sin(theta)
                perp_z = np.cos(angle) * np.cos(theta)

                x = sx + spoke_r * perp_x
                y = cy + spoke_r * perp_y
                z = sz + spoke_r * perp_z

                vertices.append([x, y, z])
                normals.append([perp_x, perp_y, perp_z])

        # Side triangles
        for j in range(n_segments):
            j_next = (j + 1) % n_segments
            v0 = start_idx + j
            v1 = start_idx + j_next
            v2 = start_idx + n_segments + j_next
            v3 = start_idx + n_segments + j
            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

    # =========================================================================
    # ANIMATION METHODS
    # =========================================================================
    
    def update_animation(self, dt: float, target_position: float = None, 
                         transition_time: float = 0.5):
        """
        Update damper blade animation with smooth transition.
        
        The blade rotates from closed (0°) to open (90°).
        
        Args:
            dt: Time step [seconds]
            target_position: Target position (0=closed, 1=open). 
                           Uses current target if None.
            transition_time: Time for full 0→1 transition [seconds]
        """
        if target_position is not None:
            self._target_position = max(0.0, min(1.0, target_position))
        
        # Move current position toward target
        if transition_time > 0:
            rate = 1.0 / transition_time  # Position change per second
            delta = self._target_position - self._current_position
            
            if abs(delta) > rate * dt:
                # Haven't reached target yet
                self._current_position += np.sign(delta) * rate * dt
            else:
                # Reached target
                self._current_position = self._target_position
        else:
            self._current_position = self._target_position
        
        # Keep in valid range
        self._current_position = max(0.0, min(1.0, self._current_position))
    
    def get_blade_position(self) -> float:
        """Get current blade position (0=closed, 1=open)."""
        return self._current_position
    
    def get_blade_angle(self) -> float:
        """
        Get current blade rotation angle [radians].
        
        Position 0 (closed) = 0° = blade perpendicular to flow
        Position 1 (open) = 90° = blade parallel to flow
        """
        return self._current_position * PI / 2
    
    def set_blade_position(self, position: float):
        """
        Set blade position directly (no animation).
        
        Args:
            position: Position 0=closed, 1=fully open
        """
        self._current_position = max(0.0, min(1.0, position))
        self._target_position = self._current_position
    
    def get_blade_transform(self, position: float = None) -> np.ndarray:
        """
        Get 4x4 transformation matrix for blade rotation.
        
        For butterfly damper (axis='x'):
        - Blade rotates around Y-axis (shaft axis)
        - Centered at damper center
        
        Args:
            position: Damper position (0-1). Uses current if None.
            
        Returns:
            4x4 homogeneous transformation matrix
        """
        if position is None:
            position = self._current_position
        
        p = self.params
        cx, cy, cz = p.center
        
        # Blade angle: 0 at closed, 90° at open
        angle = position * PI / 2
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        if p.axis == "x":
            # Blade rotates around Y-axis (perpendicular to flow and blade face)
            # Rotation is in XZ plane
            transform = np.array([
                [cos_a,  0, sin_a, cx - cx*cos_a - cz*sin_a],
                [0,      1, 0,     0],
                [-sin_a, 0, cos_a, cz + cx*sin_a - cz*cos_a],
                [0,      0, 0,     1]
            ], dtype=np.float32)
        elif p.axis == "y":
            # Blade rotates around X-axis
            transform = np.array([
                [1, 0,      0,     0],
                [0, cos_a, -sin_a, cy - cy*cos_a + cz*sin_a],
                [0, sin_a,  cos_a, cz - cy*sin_a - cz*cos_a],
                [0, 0,      0,     1]
            ], dtype=np.float32)
        else:  # z axis
            # Blade rotates around X-axis
            transform = np.array([
                [1, 0,      0,     0],
                [0, cos_a, -sin_a, cy - cy*cos_a + cz*sin_a],
                [0, sin_a,  cos_a, cz - cy*sin_a - cz*cos_a],
                [0, 0,      0,     1]
            ], dtype=np.float32)
        
        return transform
    
    def get_static_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get mesh for static (non-moving) parts of the damper.
        
        Static parts:
        - Cylindrical housing
        - Flanges
        - Actuator housing
        
        Returns:
            Tuple of (vertices, indices, normals)
        """
        if self._static_vertices is None:
            self._generate_separated_meshes()
        
        return self._static_vertices, self._static_indices, self._static_normals
    
    def get_blade_mesh(self, position: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get mesh for the damper blade at specified position.
        
        Uses cached base mesh and applies rotation transform for efficiency.
        
        Args:
            position: Damper position (0=closed, 1=open). 
                     Uses current position if None.
            
        Returns:
            Tuple of (vertices, indices, normals) with rotation applied
        """
        if position is None:
            position = self._current_position
        
        p = self.params
        
        # Generate base blade mesh at position=0 if not cached
        if self._blade_base_vertices is None:
            blade_verts = []
            blade_indices = []
            blade_normals = []
            
            # Generate at position=0 (closed)
            original_position = p.position
            p.position = 0.0
            
            if p.damper_type == "butterfly":
                self._generate_butterfly_blade(blade_verts, blade_indices, blade_normals)
            elif p.damper_type == "louver":
                self._generate_louver_blades(blade_verts, blade_indices, blade_normals)
            else:
                self._generate_iris_blades(blade_verts, blade_indices, blade_normals)
            
            p.position = original_position
            
            self._blade_base_vertices = np.array(blade_verts, dtype=np.float32)
            self._blade_indices = np.array(blade_indices, dtype=np.int32)
            self._blade_base_normals = np.array(blade_normals, dtype=np.float32)
        
        # Apply rotation based on position
        # Position 0 = 0° (blade perpendicular to flow - closed)
        # Position 1 = 90° (blade parallel to flow - open)
        angle = position * PI / 2
        
        if abs(angle) < 1e-6:
            # No rotation needed
            return (self._blade_base_vertices.copy(), 
                    self._blade_indices.copy(), 
                    self._blade_base_normals.copy())
        
        # Apply rotation transform
        rotated_verts = self._blade_base_vertices.copy()
        rotated_normals = self._blade_base_normals.copy()
        
        cx, cy, cz = p.center
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        if p.axis == "x":
            # Blade rotates in XZ plane (around Y axis through blade center)
            for i in range(len(rotated_verts)):
                # Translate to origin relative to blade pivot
                x = rotated_verts[i, 0] - cx
                z = rotated_verts[i, 2] - cz
                
                # Rotate around Y
                new_x = x * cos_a + z * sin_a
                new_z = -x * sin_a + z * cos_a
                
                rotated_verts[i, 0] = new_x + cx
                rotated_verts[i, 2] = new_z + cz
                
                # Rotate normals
                nx = rotated_normals[i, 0]
                nz = rotated_normals[i, 2]
                rotated_normals[i, 0] = nx * cos_a + nz * sin_a
                rotated_normals[i, 2] = -nx * sin_a + nz * cos_a
        else:
            # For other axes, similar rotation logic
            pass
        
        return rotated_verts, self._blade_indices.copy(), rotated_normals
    
    def _generate_separated_meshes(self):
        """Generate separate meshes for static and animated parts."""
        p = self.params
        
        # Generate static parts
        static_verts = []
        static_indices = []
        static_normals = []
        
        self._generate_housing(static_verts, static_indices, static_normals)
        self._generate_flanges(static_verts, static_indices, static_normals)
        self._generate_actuator(static_verts, static_indices, static_normals)
        
        self._static_vertices = np.array(static_verts, dtype=np.float32)
        self._static_indices = np.array(static_indices, dtype=np.int32)
        self._static_normals = np.array(static_normals, dtype=np.float32)
    
    def set_position(self, position: float):
        """
        Set damper position.

        Args:
            position: Position 0=closed, 1=fully open
        """
        self.params.position = max(0, min(1, position))
        self._current_position = self.params.position
        self._target_position = self.params.position
        # Invalidate mesh to regenerate
        self._vertices = None
        self._indices = None
        self._normals = None
        self._static_vertices = None  # Also invalidate separated meshes

    def get_pressure_drop(self, flow_rate: float) -> float:
        """
        Estimate pressure drop at given flow rate.

        Args:
            flow_rate: Air flow rate [m³/h]

        Returns:
            Pressure drop [Pa]
        """
        # dP = 0.5 * rho * v^2 * K
        # K depends on position
        Q = flow_rate / 3600  # m³/s
        A_eff = self.params.flow_area()

        if A_eff < 1e-6:
            return float('inf')

        v = Q / A_eff
        rho = 1.2

        # Loss coefficient varies with position
        # Fully open ~ 0.2, throttled ~ 5-20
        K = 0.2 + (1 - self.params.position) ** 2 * 20

        return 0.5 * rho * v ** 2 * K

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the damper geometry."""
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
        Get connection ports for the damper.
        
        Flow direction depends on axis parameter:
        - axis='x': flow along X (-X inlet, +X outlet)
        - axis='y': flow along Y
        - axis='z': flow along Z
        
        Ports:
        - 'inlet': Upstream connection
        - 'outlet': Downstream connection
        
        Returns:
            Dictionary of port name to ConnectionPort
        """
        from ..connection_ports import ConnectionPort, PortType
        
        p = self.params
        half_length = p.housing_length / 2
        
        # Direction vectors based on axis
        if p.axis == "x":
            inlet_dir = (-1.0, 0.0, 0.0)
            outlet_dir = (1.0, 0.0, 0.0)
            inlet_pos = (p.center[0] - half_length, p.center[1], p.center[2])
            outlet_pos = (p.center[0] + half_length, p.center[1], p.center[2])
        elif p.axis == "y":
            inlet_dir = (0.0, -1.0, 0.0)
            outlet_dir = (0.0, 1.0, 0.0)
            inlet_pos = (p.center[0], p.center[1] - half_length, p.center[2])
            outlet_pos = (p.center[0], p.center[1] + half_length, p.center[2])
        else:  # z
            inlet_dir = (0.0, 0.0, -1.0)
            outlet_dir = (0.0, 0.0, 1.0)
            inlet_pos = (p.center[0], p.center[1], p.center[2] - half_length)
            outlet_pos = (p.center[0], p.center[1], p.center[2] + half_length)
        
        return {
            'inlet': ConnectionPort(
                position=inlet_pos,
                direction=inlet_dir,
                diameter=p.diameter,
                port_type=PortType.FLANGED,
                name="damper_inlet",
                compatible_types=[PortType.CIRCULAR, PortType.FLANGED],
            ),
            'outlet': ConnectionPort(
                position=outlet_pos,
                direction=outlet_dir,
                diameter=p.diameter,
                port_type=PortType.FLANGED,
                name="damper_outlet",
                compatible_types=[PortType.CIRCULAR, PortType.FLANGED],
            ),
        }


def create_standard_damper(
    diameter: float = 0.30,
    damper_type: str = "butterfly",
    position: float = 1.0,
    actuator_type: str = "manual"
) -> FlowDamper:
    """
    Create a standard flow control damper.

    Args:
        diameter: Duct diameter [m]
        damper_type: Type ("butterfly", "louver", "iris")
        position: Initial position (0=closed, 1=open)
        actuator_type: Actuator type ("manual", "pneumatic", "electric")

    Returns:
        FlowDamper instance
    """
    params = DamperParams(
        diameter=diameter,
        damper_type=damper_type,
        position=position,
        actuator_type=actuator_type,
    )

    return FlowDamper(params)
