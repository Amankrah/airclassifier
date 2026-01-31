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
    - Cylindrical housing
    - Damper blade(s)
    - Actuator housing
    - Flanges

    Coordinate system:
    - Origin at center of damper
    - Flow along specified axis
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
        """Generate butterfly damper blade."""
        p = self.params

        start_idx = len(vertices)

        # Blade is a circular disk that rotates
        # At position 0 (closed), blade is perpendicular to flow
        # At position 1 (open), blade is parallel to flow

        blade_angle = p.position * PI / 2  # 0 to 90 degrees
        r = p.radius - 0.002  # Slightly smaller than housing
        half_thick = p.blade_thickness / 2

        n_radial = p.resolution // 2

        # Generate blade as a disk
        # Rotation is around the Y axis (perpendicular to flow X and blade Z)

        for side in [-1, 1]:
            blade_side_start = len(vertices)

            # Center vertex
            if p.axis == "x":
                cx, cy, cz = p.center[0], p.center[1], p.center[2]
                vertices.append([cx + side * half_thick * np.cos(blade_angle),
                               cy, cz + side * half_thick * np.sin(blade_angle)])
                normals.append([np.cos(blade_angle) * side, 0, np.sin(blade_angle) * side])

                # Edge vertices
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    # Disk in Y-Z plane, rotated by blade_angle around Y
                    y = cy + r * np.sin(theta)
                    z_local = r * np.cos(theta)
                    x_offset = z_local * np.sin(blade_angle) + side * half_thick * np.cos(blade_angle)
                    z_offset = z_local * np.cos(blade_angle) + side * half_thick * np.sin(blade_angle)

                    vertices.append([cx + x_offset, y, cz + z_offset])
                    normals.append([np.cos(blade_angle) * side, 0, np.sin(blade_angle) * side])
            else:
                # Simplified for other axes
                vertices.append([p.center[0], p.center[1], p.center[2]])
                normals.append([0, 0, side])
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    vertices.append([p.center[0] + r * np.cos(theta),
                                   p.center[1] + r * np.sin(theta), p.center[2]])
                    normals.append([0, 0, side])

            # Triangles for this side of blade
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                if side > 0:
                    indices.extend([blade_side_start, blade_side_start + 1 + j,
                                  blade_side_start + 1 + j_next])
                else:
                    indices.extend([blade_side_start, blade_side_start + 1 + j_next,
                                  blade_side_start + 1 + j])

        # Blade shaft (pivot rod)
        self._add_blade_shaft(vertices, indices, normals)

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
            # Cylindrical body for pneumatic actuator
            self._generate_actuator_cylinder(
                vertices, indices, normals,
                x_center, body_base_y, z_center,
                actuator_width * 0.4, actuator_height
            )
        else:
            # Box body for electric/manual actuator
            self._generate_actuator_box(
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

    def set_position(self, position: float):
        """
        Set damper position.

        Args:
            position: Position 0=closed, 1=fully open
        """
        self.params.position = max(0, min(1, position))
        # Invalidate mesh to regenerate
        self._vertices = None
        self._indices = None
        self._normals = None

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
    position: float = 1.0
) -> FlowDamper:
    """
    Create a standard flow control damper.

    Args:
        diameter: Duct diameter [m]
        damper_type: Type ("butterfly", "louver", "iris")
        position: Initial position (0=closed, 1=open)

    Returns:
        FlowDamper instance
    """
    params = DamperParams(
        diameter=diameter,
        damper_type=damper_type,
        position=position,
    )

    return FlowDamper(params)
