"""
Bag filter (baghouse) component for fine particle collection.

Bag filters provide high-efficiency collection of fine particles
that escape cyclone separators. Essential for capturing the
protein-rich fine fraction in legume processing.

Principle:
- Dust-laden air passes through fabric filter bags
- Particles deposit on bag surface forming filter cake
- Periodic pulse-jet cleaning removes accumulated dust
- Collection efficiency >99.9% for particles >1 μm
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import numpy as np
import warp as wp

from ...utils.constants import PI, TWO_PI
from ..connection_ports import ConnectionPort, PortType


@dataclass
class BagFilterParams:
    """Parameters for bag filter/baghouse."""

    # Housing dimensions
    housing_width: float         # [m] Housing width (X direction)
    housing_depth: float         # [m] Housing depth (Z direction)
    housing_height: float        # [m] Total housing height (Y direction)

    # Bag specifications
    num_bags_x: int              # Number of bags in X direction
    num_bags_z: int              # Number of bags in Z direction
    bag_diameter: float          # [m] Individual bag diameter
    bag_length: float            # [m] Bag length (hanging down)
    bag_spacing_x: float         # [m] Center-to-center spacing in X
    bag_spacing_z: float         # [m] Center-to-center spacing in Z

    # Sections
    clean_air_plenum_height: float   # [m] Height of clean air plenum (above tube sheet)
    tube_sheet_thickness: float      # [m] Tube sheet thickness
    dirty_air_section_height: float  # [m] Height for dirty air entry

    # Hopper
    hopper_height: float         # [m] Hopper height
    hopper_outlet_width: float   # [m] Hopper outlet width
    hopper_outlet_depth: float   # [m] Hopper outlet depth
    hopper_angle: float          # [rad] Hopper wall angle from vertical

    # Inlet/Outlet
    dirty_air_inlet_diameter: float   # [m] Dirty air inlet diameter
    clean_air_outlet_diameter: float  # [m] Clean air outlet diameter

    # Pulse-jet cleaning system
    include_pulse_jet: bool = True           # Include pulse-jet system geometry
    pulse_header_diameter: float = 0.05      # [m] Main air header pipe diameter (50mm)
    blow_tube_diameter: float = 0.025        # [m] Blow tube diameter (25mm)
    blow_tube_length: float = 0.15           # [m] Blow tube extension into bag
    nozzle_diameter: float = 0.012           # [m] Nozzle tip diameter (12mm)
    air_tank_diameter: float = 0.20          # [m] Compressed air tank diameter
    air_tank_length: float = 0.40            # [m] Compressed air tank length

    # Position
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Bottom center of hopper

    # Mesh resolution
    resolution: int = 16

    @property
    def num_bags(self) -> int:
        """Total number of bags."""
        return self.num_bags_x * self.num_bags_z

    @property
    def total_filter_area(self) -> float:
        """Total filter area [m²]."""
        single_bag_area = PI * self.bag_diameter * self.bag_length
        return self.num_bags * single_bag_area

    @property
    def tube_sheet_height(self) -> float:
        """Height of tube sheet from housing bottom."""
        return self.hopper_height + self.dirty_air_section_height

    @property
    def air_to_cloth_ratio(self) -> float:
        """Design air-to-cloth ratio [m³/min/m²] for 1 m³/s flow."""
        # Typical design: 1.5-3.0 m³/min/m²
        return 60.0 / self.total_filter_area  # For 1 m³/s

    def get_air_to_cloth(self, flow_rate_m3s: float) -> float:
        """Calculate air-to-cloth ratio for given flow rate."""
        return (flow_rate_m3s * 60) / self.total_filter_area


class BagFilter:
    """
    Bag filter (baghouse) for fine particle collection.

    Components:
    - Hopper: Collection of fallen dust
    - Dirty air section: Inlet and distribution
    - Filter bags: Hanging from tube sheet
    - Tube sheet: Supports bags, separates dirty/clean air
    - Clean air plenum: Collects filtered air
    - Pulse-jet cleaning system:
      * Compressed air tank (reservoir) on top
      * Main feed pipe from tank
      * Header pipes across each bag row
      * Blow tubes above each bag
      * Conical nozzles directing pulses into bags

    Coordinate system:
    - Origin at bottom center of hopper outlet
    - Y-axis pointing upward
    - X-axis along housing width
    - Z-axis along housing depth
    """

    def __init__(self, params: BagFilterParams):
        """
        Initialize bag filter.

        Args:
            params: BagFilterParams defining the geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

        # Calculate bag positions
        self._calculate_bag_positions()

    def _calculate_bag_positions(self):
        """Calculate center positions of all bags."""
        p = self.params
        self.bag_positions = []

        # Starting position (centered in housing)
        x_start = p.center[0] - (p.num_bags_x - 1) * p.bag_spacing_x / 2
        z_start = p.center[2] - (p.num_bags_z - 1) * p.bag_spacing_z / 2
        y_top = p.center[1] + p.tube_sheet_height + p.tube_sheet_thickness

        for i in range(p.num_bags_x):
            for j in range(p.num_bags_z):
                x = x_start + i * p.bag_spacing_x
                z = z_start + j * p.bag_spacing_z
                self.bag_positions.append((x, y_top, z))

    def generate_mesh(self, include_bags: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for the bag filter.

        Args:
            include_bags: Whether to include individual bag geometry

        Returns:
            Tuple of (vertices, indices, normals)
        """
        p = self.params
        vertices = []
        indices = []
        normals = []

        # Generate housing
        self._generate_housing(vertices, indices, normals)

        # Generate hopper
        self._generate_hopper(vertices, indices, normals)

        # Generate tube sheet (simplified as plate with holes)
        self._generate_tube_sheet(vertices, indices, normals)

        # Generate bags if requested
        if include_bags:
            self._generate_bags(vertices, indices, normals)

        # Generate inlet and outlet
        self._generate_inlet_outlet(vertices, indices, normals)

        # Generate pulse-jet cleaning system
        if p.include_pulse_jet:
            self._generate_pulse_jet_system(vertices, indices, normals)

        self._vertices = np.array(vertices, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.int32)
        self._normals = np.array(normals, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def _generate_housing(self, vertices: List, indices: List, normals: List):
        """Generate housing box geometry."""
        p = self.params

        hw = p.housing_width / 2
        hd = p.housing_depth / 2
        y_bottom = p.center[1] + p.hopper_height
        y_top = p.center[1] + p.housing_height

        start_idx = len(vertices)

        # 8 corners of housing (excluding hopper)
        housing_verts = [
            [p.center[0] - hw, y_bottom, p.center[2] - hd],  # 0
            [p.center[0] + hw, y_bottom, p.center[2] - hd],  # 1
            [p.center[0] + hw, y_bottom, p.center[2] + hd],  # 2
            [p.center[0] - hw, y_bottom, p.center[2] + hd],  # 3
            [p.center[0] - hw, y_top, p.center[2] - hd],     # 4
            [p.center[0] + hw, y_top, p.center[2] - hd],     # 5
            [p.center[0] + hw, y_top, p.center[2] + hd],     # 6
            [p.center[0] - hw, y_top, p.center[2] + hd],     # 7
        ]

        for v in housing_verts:
            vertices.append(v)
            normals.append([0.0, 0.0, 0.0])

        # Housing faces (4 walls + top)
        # Front wall (-Z)
        indices.extend([start_idx + 0, start_idx + 4, start_idx + 5])
        indices.extend([start_idx + 0, start_idx + 5, start_idx + 1])
        # Back wall (+Z)
        indices.extend([start_idx + 2, start_idx + 6, start_idx + 7])
        indices.extend([start_idx + 2, start_idx + 7, start_idx + 3])
        # Left wall (-X)
        indices.extend([start_idx + 3, start_idx + 7, start_idx + 4])
        indices.extend([start_idx + 3, start_idx + 4, start_idx + 0])
        # Right wall (+X)
        indices.extend([start_idx + 1, start_idx + 5, start_idx + 6])
        indices.extend([start_idx + 1, start_idx + 6, start_idx + 2])
        # Top
        indices.extend([start_idx + 4, start_idx + 7, start_idx + 6])
        indices.extend([start_idx + 4, start_idx + 6, start_idx + 5])

    def _generate_hopper(self, vertices: List, indices: List, normals: List):
        """Generate hopper geometry."""
        p = self.params

        hw_top = p.housing_width / 2
        hd_top = p.housing_depth / 2
        hw_bot = p.hopper_outlet_width / 2
        hd_bot = p.hopper_outlet_depth / 2

        y_top = p.center[1] + p.hopper_height
        y_bottom = p.center[1]

        start_idx = len(vertices)

        # Hopper vertices (frustum)
        hopper_verts = [
            # Bottom (outlet)
            [p.center[0] - hw_bot, y_bottom, p.center[2] - hd_bot],  # 0
            [p.center[0] + hw_bot, y_bottom, p.center[2] - hd_bot],  # 1
            [p.center[0] + hw_bot, y_bottom, p.center[2] + hd_bot],  # 2
            [p.center[0] - hw_bot, y_bottom, p.center[2] + hd_bot],  # 3
            # Top (connects to housing)
            [p.center[0] - hw_top, y_top, p.center[2] - hd_top],     # 4
            [p.center[0] + hw_top, y_top, p.center[2] - hd_top],     # 5
            [p.center[0] + hw_top, y_top, p.center[2] + hd_top],     # 6
            [p.center[0] - hw_top, y_top, p.center[2] + hd_top],     # 7
        ]

        for v in hopper_verts:
            vertices.append(v)
            normals.append([0.0, 0.0, 0.0])

        # Hopper faces (4 sloped walls)
        # Front wall
        indices.extend([start_idx + 0, start_idx + 4, start_idx + 5])
        indices.extend([start_idx + 0, start_idx + 5, start_idx + 1])
        # Back wall
        indices.extend([start_idx + 2, start_idx + 6, start_idx + 7])
        indices.extend([start_idx + 2, start_idx + 7, start_idx + 3])
        # Left wall
        indices.extend([start_idx + 3, start_idx + 7, start_idx + 4])
        indices.extend([start_idx + 3, start_idx + 4, start_idx + 0])
        # Right wall
        indices.extend([start_idx + 1, start_idx + 5, start_idx + 6])
        indices.extend([start_idx + 1, start_idx + 6, start_idx + 2])

    def _generate_tube_sheet(self, vertices: List, indices: List, normals: List):
        """Generate tube sheet (simplified as solid plate)."""
        p = self.params

        hw = p.housing_width / 2
        hd = p.housing_depth / 2
        y_bottom = p.center[1] + p.tube_sheet_height
        y_top = y_bottom + p.tube_sheet_thickness

        start_idx = len(vertices)

        # Tube sheet plate vertices
        plate_verts = [
            [p.center[0] - hw, y_bottom, p.center[2] - hd],
            [p.center[0] + hw, y_bottom, p.center[2] - hd],
            [p.center[0] + hw, y_bottom, p.center[2] + hd],
            [p.center[0] - hw, y_bottom, p.center[2] + hd],
            [p.center[0] - hw, y_top, p.center[2] - hd],
            [p.center[0] + hw, y_top, p.center[2] - hd],
            [p.center[0] + hw, y_top, p.center[2] + hd],
            [p.center[0] - hw, y_top, p.center[2] + hd],
        ]

        for v in plate_verts:
            vertices.append(v)
            normals.append([0.0, 1.0, 0.0])

        # Bottom face (visible from dirty air section)
        indices.extend([start_idx + 0, start_idx + 1, start_idx + 2])
        indices.extend([start_idx + 0, start_idx + 2, start_idx + 3])

    def _generate_bags(self, vertices: List, indices: List, normals: List):
        """Generate filter bag geometry (simplified as cylinders)."""
        p = self.params
        n_radial = max(8, p.resolution // 2)

        for bag_x, bag_y, bag_z in self.bag_positions:
            start_idx = len(vertices)
            r = p.bag_diameter / 2

            # Generate cylinder for each bag
            for i in range(2):  # Top and bottom rings
                y = bag_y - i * p.bag_length

                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    vx = bag_x + r * np.cos(theta)
                    vz = bag_z + r * np.sin(theta)

                    vertices.append([vx, y, vz])
                    normals.append([np.cos(theta), 0.0, np.sin(theta)])

            # Generate triangles
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = start_idx + j
                v1 = start_idx + j_next
                v2 = start_idx + n_radial + j_next
                v3 = start_idx + n_radial + j

                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

            # Bottom cap
            cap_start = len(vertices)
            y_bottom = bag_y - p.bag_length
            vertices.append([bag_x, y_bottom, bag_z])
            normals.append([0.0, -1.0, 0.0])

            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                vx = bag_x + r * np.cos(theta)
                vz = bag_z + r * np.sin(theta)
                vertices.append([vx, y_bottom, vz])
                normals.append([0.0, -1.0, 0.0])

            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                indices.extend([cap_start, cap_start + 1 + j_next, cap_start + 1 + j])

    def _generate_inlet_outlet(self, vertices: List, indices: List, normals: List):
        """Generate inlet and outlet pipes."""
        p = self.params
        n_radial = p.resolution

        # Dirty air inlet (on -X side of housing, in dirty air section)
        # Pipe extends from housing wall outward in -X direction
        inlet_y = p.center[1] + p.hopper_height + p.dirty_air_section_height / 2
        inlet_length = p.housing_width * 0.3
        # Start at outer end of pipe (most -X position), extend toward housing
        inlet_x_start = p.center[0] - p.housing_width / 2 - inlet_length

        self._add_pipe(vertices, indices, normals,
                      center=(inlet_x_start, inlet_y, p.center[2]),
                      diameter=p.dirty_air_inlet_diameter,
                      length=inlet_length,
                      axis='x', n_radial=n_radial)

        # Clean air outlet (on top)
        outlet_y = p.center[1] + p.housing_height
        outlet_length = p.housing_height * 0.1

        self._add_pipe(vertices, indices, normals,
                      center=(p.center[0], outlet_y, p.center[2]),
                      diameter=p.clean_air_outlet_diameter,
                      length=outlet_length,
                      axis='y', n_radial=n_radial)

    def _add_pipe(self, vertices: List, indices: List, normals: List,
                  center: Tuple, diameter: float, length: float,
                  axis: str, n_radial: int):
        """Add a cylindrical pipe to the mesh."""
        start_idx = len(vertices)
        r = diameter / 2

        for i in range(2):
            if axis == 'x':
                x = center[0] + i * length
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    y = center[1] + r * np.cos(theta)
                    z = center[2] + r * np.sin(theta)
                    vertices.append([x, y, z])
                    normals.append([0.0, np.cos(theta), np.sin(theta)])
            elif axis == 'y':
                y = center[1] + i * length
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x = center[0] + r * np.cos(theta)
                    z = center[2] + r * np.sin(theta)
                    vertices.append([x, y, z])
                    normals.append([np.cos(theta), 0.0, np.sin(theta)])

        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = start_idx + j
            v1 = start_idx + j_next
            v2 = start_idx + n_radial + j_next
            v3 = start_idx + n_radial + j

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

    def _generate_pulse_jet_system(self, vertices: List, indices: List, normals: List):
        """
        Generate pulse-jet cleaning system geometry.
        
        Components:
        - Compressed air tank (reservoir) mounted on top
        - Main header pipes running across bag rows
        - Blow tubes extending down above each bag
        - Nozzle tips for directing air pulses
        """
        p = self.params
        n_radial = max(8, p.resolution // 2)
        
        # Position in clean air plenum (above tube sheet)
        tube_sheet_top = p.center[1] + p.tube_sheet_height + p.tube_sheet_thickness
        plenum_height = p.clean_air_plenum_height
        
        # Header height: in upper part of plenum
        header_y = tube_sheet_top + plenum_height * 0.7
        
        # ============================================================
        # 1. COMPRESSED AIR TANK (mounted on top of housing, +Z side)
        # ============================================================
        tank_y = p.center[1] + p.housing_height + p.air_tank_diameter / 2 + 0.05
        tank_z = p.center[2] + p.housing_depth / 2 - p.air_tank_diameter
        tank_x = p.center[0]
        
        self._add_horizontal_cylinder(
            vertices, indices, normals,
            center=(tank_x, tank_y, tank_z),
            diameter=p.air_tank_diameter,
            length=p.air_tank_length,
            axis='x',
            n_radial=n_radial,
            caps=True
        )
        
        # Tank end caps (hemispherical look - simplified as flat)
        # Already included with caps=True
        
        # ============================================================
        # 2. MAIN FEED PIPE from tank down into housing
        # ============================================================
        feed_pipe_length = tank_y - header_y - p.pulse_header_diameter
        if feed_pipe_length > 0:
            self._add_pipe(
                vertices, indices, normals,
                center=(tank_x, header_y + p.pulse_header_diameter/2, tank_z),
                diameter=p.pulse_header_diameter,
                length=feed_pipe_length,
                axis='y',
                n_radial=n_radial
            )
        
        # ============================================================
        # 3. HEADER PIPES (one for each row of bags in X direction)
        # ============================================================
        # Headers run along Z direction
        x_start = p.center[0] - (p.num_bags_x - 1) * p.bag_spacing_x / 2
        z_start = p.center[2] - (p.num_bags_z - 1) * p.bag_spacing_z / 2
        z_end = p.center[2] + (p.num_bags_z - 1) * p.bag_spacing_z / 2
        header_length = z_end - z_start + p.bag_spacing_z * 0.5
        
        for i in range(p.num_bags_x):
            header_x = x_start + i * p.bag_spacing_x
            header_z_start = z_start - p.bag_spacing_z * 0.25
            
            self._add_horizontal_cylinder(
                vertices, indices, normals,
                center=(header_x, header_y, header_z_start),
                diameter=p.pulse_header_diameter,
                length=header_length,
                axis='z',
                n_radial=n_radial,
                caps=True
            )
        
        # ============================================================
        # 4. BLOW TUBES with NOZZLES (one above each bag)
        # ============================================================
        for bag_x, bag_y_top, bag_z in self.bag_positions:
            # Blow tube extends from header down toward bag opening
            tube_top = header_y - p.pulse_header_diameter / 2
            tube_bottom = tube_sheet_top + 0.02  # Just above tube sheet
            tube_length = tube_top - tube_bottom
            
            if tube_length > 0:
                # Main blow tube
                self._add_pipe(
                    vertices, indices, normals,
                    center=(bag_x, tube_bottom, bag_z),
                    diameter=p.blow_tube_diameter,
                    length=tube_length,
                    axis='y',
                    n_radial=n_radial
                )
                
                # Nozzle tip (extends into bag opening)
                nozzle_top = tube_bottom
                nozzle_bottom = nozzle_top - p.blow_tube_length
                
                self._add_nozzle(
                    vertices, indices, normals,
                    center=(bag_x, nozzle_bottom, bag_z),
                    top_diameter=p.blow_tube_diameter,
                    bottom_diameter=p.nozzle_diameter,
                    length=p.blow_tube_length,
                    n_radial=n_radial
                )

    def _add_horizontal_cylinder(self, vertices: List, indices: List, normals: List,
                                  center: Tuple, diameter: float, length: float,
                                  axis: str, n_radial: int, caps: bool = False):
        """Add a horizontal cylinder (for tanks and headers)."""
        start_idx = len(vertices)
        r = diameter / 2
        
        # Generate cylinder body
        for i in range(2):  # Two end rings
            if axis == 'x':
                x = center[0] - length/2 + i * length
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    y = center[1] + r * np.cos(theta)
                    z = center[2] + r * np.sin(theta)
                    vertices.append([x, y, z])
                    normals.append([0.0, np.cos(theta), np.sin(theta)])
            elif axis == 'z':
                z = center[2] + i * length
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x = center[0] + r * np.cos(theta)
                    y = center[1] + r * np.sin(theta)
                    vertices.append([x, y, z])
                    normals.append([np.cos(theta), np.sin(theta), 0.0])
        
        # Connect rings
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = start_idx + j
            v1 = start_idx + j_next
            v2 = start_idx + n_radial + j_next
            v3 = start_idx + n_radial + j
            
            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])
        
        # End caps
        if caps:
            for end in range(2):
                cap_idx = len(vertices)
                if axis == 'x':
                    cap_x = center[0] - length/2 + end * length
                    cap_normal = [-1.0, 0.0, 0.0] if end == 0 else [1.0, 0.0, 0.0]
                    # Center vertex
                    vertices.append([cap_x, center[1], center[2]])
                    normals.append(cap_normal)
                    # Rim vertices
                    for j in range(n_radial):
                        theta = (j / n_radial) * TWO_PI
                        y = center[1] + r * np.cos(theta)
                        z = center[2] + r * np.sin(theta)
                        vertices.append([cap_x, y, z])
                        normals.append(cap_normal)
                elif axis == 'z':
                    cap_z = center[2] + end * length
                    cap_normal = [0.0, 0.0, -1.0] if end == 0 else [0.0, 0.0, 1.0]
                    # Center vertex
                    vertices.append([center[0], center[1], cap_z])
                    normals.append(cap_normal)
                    # Rim vertices
                    for j in range(n_radial):
                        theta = (j / n_radial) * TWO_PI
                        x = center[0] + r * np.cos(theta)
                        y = center[1] + r * np.sin(theta)
                        vertices.append([x, y, cap_z])
                        normals.append(cap_normal)
                
                # Cap triangles
                for j in range(n_radial):
                    j_next = (j + 1) % n_radial
                    if end == 0:
                        indices.extend([cap_idx, cap_idx + 1 + j_next, cap_idx + 1 + j])
                    else:
                        indices.extend([cap_idx, cap_idx + 1 + j, cap_idx + 1 + j_next])

    def _add_nozzle(self, vertices: List, indices: List, normals: List,
                    center: Tuple, top_diameter: float, bottom_diameter: float,
                    length: float, n_radial: int):
        """Add a conical nozzle (tapers from blow tube to nozzle tip)."""
        start_idx = len(vertices)
        r_top = top_diameter / 2
        r_bottom = bottom_diameter / 2
        
        # Bottom ring (nozzle tip)
        for j in range(n_radial):
            theta = (j / n_radial) * TWO_PI
            x = center[0] + r_bottom * np.cos(theta)
            z = center[2] + r_bottom * np.sin(theta)
            y = center[1]
            vertices.append([x, y, z])
            # Approximate outward normal for cone
            slope = (r_top - r_bottom) / length
            ny = slope / np.sqrt(1 + slope*slope)
            nr = 1 / np.sqrt(1 + slope*slope)
            normals.append([nr * np.cos(theta), ny, nr * np.sin(theta)])
        
        # Top ring (connects to blow tube)
        for j in range(n_radial):
            theta = (j / n_radial) * TWO_PI
            x = center[0] + r_top * np.cos(theta)
            z = center[2] + r_top * np.sin(theta)
            y = center[1] + length
            vertices.append([x, y, z])
            slope = (r_top - r_bottom) / length
            ny = slope / np.sqrt(1 + slope*slope)
            nr = 1 / np.sqrt(1 + slope*slope)
            normals.append([nr * np.cos(theta), ny, nr * np.sin(theta)])
        
        # Connect rings
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = start_idx + j
            v1 = start_idx + j_next
            v2 = start_idx + n_radial + j_next
            v3 = start_idx + n_radial + j
            
            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])
        
        # Bottom cap (nozzle tip opening)
        cap_idx = len(vertices)
        vertices.append([center[0], center[1], center[2]])
        normals.append([0.0, -1.0, 0.0])
        
        for j in range(n_radial):
            theta = (j / n_radial) * TWO_PI
            x = center[0] + r_bottom * np.cos(theta)
            z = center[2] + r_bottom * np.sin(theta)
            vertices.append([x, center[1], z])
            normals.append([0.0, -1.0, 0.0])
        
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            indices.extend([cap_idx, cap_idx + 1 + j_next, cap_idx + 1 + j])

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the bag filter geometry."""
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
        Get connection ports for the bag filter.
        
        Ports:
        - dirty_air_inlet: Inlet for dust-laden air (circular, on side)
        - clean_air_outlet: Outlet for filtered air (circular, on top)
        - dust_outlet: Outlet for collected dust at hopper bottom (rectangular)
        """
        p = self.params
        
        # Dirty air inlet (on -X side of housing, in dirty air section)
        # This allows the bag filter to receive flow from cyclones on its left
        inlet_y = p.center[1] + p.hopper_height + p.dirty_air_section_height / 2
        inlet_x = p.center[0] - p.housing_width / 2 - p.housing_width * 0.3  # End of inlet pipe on -X side

        dirty_air_inlet = ConnectionPort(
            position=(inlet_x, inlet_y, p.center[2]),
            direction=(-1.0, 0.0, 0.0),  # Faces -X (receives flow from -X direction)
            diameter=p.dirty_air_inlet_diameter,
            port_type=PortType.CIRCULAR,
            name="dirty_air_inlet"
        )
        
        # Clean air outlet (on top)
        outlet_y = p.center[1] + p.housing_height + p.housing_height * 0.1  # End of outlet pipe
        
        clean_air_outlet = ConnectionPort(
            position=(p.center[0], outlet_y, p.center[2]),
            direction=(0.0, 1.0, 0.0),  # Faces up
            diameter=p.clean_air_outlet_diameter,
            port_type=PortType.CIRCULAR,
            name="clean_air_outlet"
        )
        
        # Dust outlet at hopper bottom
        dust_outlet = ConnectionPort(
            position=p.center,  # Bottom of hopper
            direction=(0.0, -1.0, 0.0),  # Faces down
            width=p.hopper_outlet_width,
            height=p.hopper_outlet_depth,
            port_type=PortType.RECTANGULAR,
            name="dust_outlet"
        )
        
        return {
            'dirty_air_inlet': dirty_air_inlet,
            'clean_air_outlet': clean_air_outlet,
            'dust_outlet': dust_outlet
        }


def create_standard_bag_filter(
    flow_rate_m3s: float = 1.0,
    air_to_cloth: float = 2.0
) -> BagFilter:
    """
    Create a standard bag filter sized for given flow rate.

    Args:
        flow_rate_m3s: Volumetric flow rate [m³/s]
        air_to_cloth: Target air-to-cloth ratio [m³/min/m²]

    Returns:
        BagFilter instance
    """
    # Calculate required filter area
    required_area = (flow_rate_m3s * 60) / air_to_cloth

    # Standard bag size
    bag_diameter = 0.15  # 150mm typical
    bag_length = 2.0     # 2m typical

    single_bag_area = PI * bag_diameter * bag_length

    # Number of bags needed
    num_bags = int(np.ceil(required_area / single_bag_area))

    # Arrange in grid
    num_bags_x = int(np.ceil(np.sqrt(num_bags)))
    num_bags_z = int(np.ceil(num_bags / num_bags_x))

    # Spacing
    bag_spacing = bag_diameter * 2.5  # 2.5x diameter spacing

    # Housing size
    housing_width = num_bags_x * bag_spacing + 0.3
    housing_depth = num_bags_z * bag_spacing + 0.3

    params = BagFilterParams(
        housing_width=housing_width,
        housing_depth=housing_depth,
        housing_height=bag_length + 1.5,  # Bags + plenums
        num_bags_x=num_bags_x,
        num_bags_z=num_bags_z,
        bag_diameter=bag_diameter,
        bag_length=bag_length,
        bag_spacing_x=bag_spacing,
        bag_spacing_z=bag_spacing,
        clean_air_plenum_height=0.5,
        tube_sheet_thickness=0.02,
        dirty_air_section_height=0.5,
        hopper_height=0.8,
        hopper_outlet_width=0.2,
        hopper_outlet_depth=0.2,
        hopper_angle=np.radians(60),
        dirty_air_inlet_diameter=0.3,
        clean_air_outlet_diameter=0.4,
    )

    return BagFilter(params)
