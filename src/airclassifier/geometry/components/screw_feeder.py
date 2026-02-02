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
from typing import Tuple, List, Optional, Dict
import numpy as np
import warp as wp

from ...utils.constants import PI, TWO_PI
from ..connection_ports import ConnectionPort, PortType


@dataclass
class ScrewFeederParams:
    """
    Parameters for screw feeder / auger conveyor.
    
    DYNAMIC PROPORTIONAL SIZING:
    The inlet and outlet diameters are direct parameters that should be set
    to match the connected components (airlock outlet and de-agglomerator inlet).
    This ensures proper geometric fit in the assembly.
    
    Connection points:
    - inlet_diameter: Set to match upstream component (e.g., airlock outlet)
    - outlet_diameter: Set to match downstream component (e.g., deagg inlet)
    
    The neck geometry is sized proportionally to create visible, properly
    aligned connection points for material flow.
    """

    # Screw geometry
    screw_diameter: float        # [m] Screw/auger outer diameter
    shaft_diameter: float        # [m] Central shaft diameter
    screw_pitch: float           # [m] Pitch of screw flights (axial distance per revolution)
    flight_thickness: float      # [m] Thickness of helical flight

    # Trough geometry
    trough_length: float         # [m] Length of trough
    trough_clearance: float      # [m] Gap between screw OD and trough

    # Inlet/outlet - DIRECT DIAMETERS for proper connection sizing
    inlet_diameter: float        # [m] Inlet port diameter (match to airlock outlet)
    outlet_diameter: float       # [m] Outlet port diameter (match to deagg inlet)

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
    
    ENCLOSED TUBE DESIGN for industrial powder handling:
    - Fully sealed cylindrical housing prevents particle escape
    - Suitable for food/pharma hygiene requirements
    - Allows pressurized or vacuum operation
    - Dust-tight construction

    Components:
    - Fully enclosed cylindrical tube housing
    - Helical screw with central shaft
    - End caps with shaft bearing holes
    - Flanged inlet neck (top, receives from airlock)
    - Flanged outlet neck (bottom, discharges to deagglomerator)

    Material Flow:
        AIRLOCK → [INLET FLANGE] → ENCLOSED TUBE → [OUTLET FLANGE] → DEAGGLOMERATOR
                         ↓                                ↓
                   Material enters              Screw conveys →
                   from top                     Material exits bottom

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
        """
        Generate FULLY ENCLOSED cylindrical tube housing for the screw.
        
        Real industrial screw feeders use a sealed tube design to:
        - Prevent particle escape/contamination
        - Contain dust
        - Allow pressurized/vacuum operation
        - Meet food/pharma hygiene standards
        
        The tube has openings only at:
        - Inlet (top, near start) - receives material from airlock
        - Outlet (bottom, at end) - discharges to deagglomerator
        """
        p = self.params
        n_radial = p.resolution_radial
        n_axial = p.resolution_axial

        start_idx = len(vertices)
        r = p.trough_radius
        wall_thickness = 0.003  # 3mm walls
        outer_r = r + wall_thickness

        # Generate FULL cylindrical tube (360 degrees, fully enclosed)
        for i in range(n_axial + 1):
            t_axial = i / n_axial
            axial_pos = t_axial * p.trough_length

            for j in range(n_radial):
                # Full circle (0 to 360 degrees)
                theta = (j / n_radial) * TWO_PI

                if p.axis == "x":
                    x = p.center[0] + axial_pos
                    y = p.center[1] + outer_r * np.sin(theta)
                    z = p.center[2] + outer_r * np.cos(theta)
                    nx, ny, nz = 0.0, np.sin(theta), np.cos(theta)
                elif p.axis == "y":
                    x = p.center[0] + outer_r * np.cos(theta)
                    y = p.center[1] + axial_pos
                    z = p.center[2] + outer_r * np.sin(theta)
                    nx, ny, nz = np.cos(theta), 0.0, np.sin(theta)
                else:  # z-axis
                    x = p.center[0] + outer_r * np.cos(theta)
                    y = p.center[1] + outer_r * np.sin(theta)
                    z = p.center[2] + axial_pos
                    nx, ny, nz = np.cos(theta), np.sin(theta), 0.0

                vertices.append([x, y, z])
                normals.append([nx, ny, nz])

        # Generate triangles for tube
        for i in range(n_axial):
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = start_idx + i * n_radial + j
                v1 = start_idx + i * n_radial + j_next
                v2 = start_idx + (i + 1) * n_radial + j_next
                v3 = start_idx + (i + 1) * n_radial + j

                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

        # Add end caps (with holes for shaft bearings)
        self._add_end_caps(vertices, indices, normals, outer_r)

    def _add_end_caps(self, vertices: List, indices: List, normals: List,
                      outer_r: float):
        """
        Add end caps to the tube with bearing holes for the shaft.
        
        Each end has an annular cap (ring) with a center hole for the shaft
        bearing/seal assembly.
        """
        p = self.params
        n_radial = p.resolution_radial
        bearing_r = p.shaft_radius * 1.5  # Bearing housing radius
        
        # Front cap (at x=0 for x-axis)
        front_cap_start = len(vertices)
        
        if p.axis == "x":
            x_front = p.center[0]
            x_back = p.center[0] + p.trough_length
            
            # Front cap - outer ring
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                y = p.center[1] + outer_r * np.sin(theta)
                z = p.center[2] + outer_r * np.cos(theta)
                vertices.append([x_front, y, z])
                normals.append([-1.0, 0.0, 0.0])
            
            # Front cap - inner ring (bearing hole)
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                y = p.center[1] + bearing_r * np.sin(theta)
                z = p.center[2] + bearing_r * np.cos(theta)
                vertices.append([x_front, y, z])
                normals.append([-1.0, 0.0, 0.0])
            
            # Front cap triangles (annular ring)
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v_outer = front_cap_start + j
                v_outer_next = front_cap_start + j_next
                v_inner = front_cap_start + n_radial + j
                v_inner_next = front_cap_start + n_radial + j_next
                
                indices.extend([v_outer, v_inner, v_outer_next])
                indices.extend([v_outer_next, v_inner, v_inner_next])
            
            # Back cap
            back_cap_start = len(vertices)
            
            # Back cap - outer ring
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                y = p.center[1] + outer_r * np.sin(theta)
                z = p.center[2] + outer_r * np.cos(theta)
                vertices.append([x_back, y, z])
                normals.append([1.0, 0.0, 0.0])
            
            # Back cap - inner ring (bearing hole)
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                y = p.center[1] + bearing_r * np.sin(theta)
                z = p.center[2] + bearing_r * np.cos(theta)
                vertices.append([x_back, y, z])
                normals.append([1.0, 0.0, 0.0])
            
            # Back cap triangles
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v_outer = back_cap_start + j
                v_outer_next = back_cap_start + j_next
                v_inner = back_cap_start + n_radial + j
                v_inner_next = back_cap_start + n_radial + j_next
                
                indices.extend([v_outer, v_outer_next, v_inner])
                indices.extend([v_outer_next, v_inner_next, v_inner])

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
        """
        Generate flanged inlet neck on TOP of enclosed tube with saddle joint.

        The inlet uses a saddle joint that curves around the cylindrical tube:
        - Saddle base follows the tube curvature
        - Transitions to circular neck
        - Ends with flange for connection to rotary airlock

        DYNAMIC PROPORTIONAL SIZING:
        The inlet diameter is set directly from params.inlet_diameter to
        match the upstream component (airlock outlet).
        """
        p = self.params
        n_radial = max(20, p.resolution_radial)
        n_transition = 4  # Segments for saddle-to-neck transition

        # Use direct inlet diameter from params (matches airlock outlet)
        neck_radius = p.inlet_diameter / 2
        r_tube = p.trough_radius + 0.003  # Tube outer radius

        # Saddle and neck dimensions
        saddle_height = 0.010  # Height of saddle transition
        neck_length = p.inlet_diameter * 0.4  # Neck length
        flange_thickness = 0.008  # 8mm flange

        if p.axis == "x":
            # Inlet positioned at 15% of trough length from start
            x_center = p.center[0] + p.trough_length * 0.15
            z_center = p.center[2]

            # === 1. SADDLE BASE (sits on tube curve) ===
            saddle_start = len(vertices)
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                # Position in XZ plane (neck cross-section)
                x_local = neck_radius * np.cos(theta)
                z_local = neck_radius * np.sin(theta)

                # Saddle base follows the tube curve
                y_offset = np.sqrt(max(0, r_tube**2 - z_local**2))
                y = p.center[1] + y_offset

                x = x_center + x_local
                z = z_center + z_local

                vertices.append([x, y, z])
                # Normal points along saddle surface
                norm_len = np.sqrt(x_local**2 + y_offset**2)
                if norm_len > 0.001:
                    normals.append([x_local/norm_len * 0.3, y_offset/norm_len, 0.0])
                else:
                    normals.append([0.0, 1.0, 0.0])

            # === 2. TRANSITION RINGS (saddle to circular) ===
            for t in range(1, n_transition + 1):
                blend = t / n_transition

                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x_local = neck_radius * np.cos(theta)
                    z_local = neck_radius * np.sin(theta)

                    # Blend from saddle curve to flat circle
                    y_saddle = np.sqrt(max(0, r_tube**2 - z_local**2))
                    y_flat = r_tube + saddle_height * blend
                    y = p.center[1] + (y_saddle * (1 - blend) + y_flat * blend)

                    x = x_center + x_local
                    z = z_center + z_local

                    vertices.append([x, y, z])
                    normals.append([np.cos(theta), 0.0, np.sin(theta)])

            # === 3. CYLINDRICAL NECK ===
            y_neck_start = p.center[1] + r_tube + saddle_height
            y_neck_end = y_neck_start + neck_length

            neck_start = len(vertices)
            # Start ring
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = x_center + neck_radius * np.cos(theta)
                z = z_center + neck_radius * np.sin(theta)
                vertices.append([x, y_neck_start, z])
                normals.append([np.cos(theta), 0.0, np.sin(theta)])

            # End ring
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = x_center + neck_radius * np.cos(theta)
                z = z_center + neck_radius * np.sin(theta)
                vertices.append([x, y_neck_end, z])
                normals.append([np.cos(theta), 0.0, np.sin(theta)])

            # === 4. FLANGE ===
            flange_start = len(vertices)
            flange_radius = neck_radius * 1.3
            y_flange = y_neck_end

            # Flange inner ring
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = x_center + neck_radius * np.cos(theta)
                z = z_center + neck_radius * np.sin(theta)
                vertices.append([x, y_flange, z])
                normals.append([0.0, 1.0, 0.0])

            # Flange outer ring
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = x_center + flange_radius * np.cos(theta)
                z = z_center + flange_radius * np.sin(theta)
                vertices.append([x, y_flange, z])
                normals.append([0.0, 1.0, 0.0])

            # === GENERATE TRIANGLES ===
            # Saddle to first transition ring
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = saddle_start + j
                v1 = saddle_start + j_next
                v2 = saddle_start + n_radial + j_next
                v3 = saddle_start + n_radial + j
                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

            # Between transition rings
            for t in range(1, n_transition):
                for j in range(n_radial):
                    j_next = (j + 1) % n_radial
                    base = saddle_start + t * n_radial
                    v0 = base + j
                    v1 = base + j_next
                    v2 = base + n_radial + j_next
                    v3 = base + n_radial + j
                    indices.extend([v0, v1, v2])
                    indices.extend([v0, v2, v3])

            # Last transition to neck start
            trans_last = saddle_start + n_transition * n_radial
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = trans_last + j
                v1 = trans_last + j_next
                v2 = neck_start + j_next
                v3 = neck_start + j
                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

            # Neck cylinder
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = neck_start + j
                v1 = neck_start + j_next
                v2 = neck_start + n_radial + j_next
                v3 = neck_start + n_radial + j
                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

            # Flange face (annular ring)
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v_inner = flange_start + j
                v_inner_next = flange_start + j_next
                v_outer = flange_start + n_radial + j
                v_outer_next = flange_start + n_radial + j_next
                indices.extend([v_inner, v_inner_next, v_outer_next])
                indices.extend([v_inner, v_outer_next, v_outer])

    def _generate_outlet(self, vertices: List, indices: List, normals: List):
        """
        Generate flanged outlet neck on BOTTOM of enclosed tube with saddle joint.

        The outlet uses a saddle joint that curves around the cylindrical tube:
        - Saddle base follows the tube curvature (bottom)
        - Transitions to circular neck
        - Ends with flange for connection to deagglomerator

        DYNAMIC PROPORTIONAL SIZING:
        The outlet diameter is set directly from params.outlet_diameter to
        match the downstream component (de-agglomerator inlet).
        """
        p = self.params
        n_radial = max(20, p.resolution_radial)
        n_transition = 4  # Segments for saddle-to-neck transition

        neck_radius = p.outlet_diameter / 2
        r_tube = p.trough_radius + 0.003  # Tube outer radius

        # Saddle and neck dimensions
        saddle_height = 0.010  # Height of saddle transition
        neck_length = p.outlet_diameter * 0.4  # Neck length
        flange_thickness = 0.008  # 8mm flange

        # Outlet at 85% of trough length (near end where material accumulates)
        if p.axis == "x":
            x_center = p.center[0] + p.trough_length * 0.85
            z_center = p.center[2]

            # === 1. SADDLE BASE (sits on tube curve, bottom) ===
            saddle_start = len(vertices)
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                # Position in XZ plane (neck cross-section)
                x_local = neck_radius * np.cos(theta)
                z_local = neck_radius * np.sin(theta)

                # Saddle base follows the tube curve (bottom side)
                y_offset = np.sqrt(max(0, r_tube**2 - z_local**2))
                y = p.center[1] - y_offset  # Negative for bottom

                x = x_center + x_local
                z = z_center + z_local

                vertices.append([x, y, z])
                # Normal points along saddle surface
                norm_len = np.sqrt(x_local**2 + y_offset**2)
                if norm_len > 0.001:
                    normals.append([x_local/norm_len * 0.3, -y_offset/norm_len, 0.0])
                else:
                    normals.append([0.0, -1.0, 0.0])

            # === 2. TRANSITION RINGS (saddle to circular) ===
            for t in range(1, n_transition + 1):
                blend = t / n_transition

                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x_local = neck_radius * np.cos(theta)
                    z_local = neck_radius * np.sin(theta)

                    # Blend from saddle curve to flat circle
                    y_saddle = np.sqrt(max(0, r_tube**2 - z_local**2))
                    y_flat = r_tube + saddle_height * blend
                    y = p.center[1] - (y_saddle * (1 - blend) + y_flat * blend)

                    x = x_center + x_local
                    z = z_center + z_local

                    vertices.append([x, y, z])
                    normals.append([np.cos(theta), 0.0, np.sin(theta)])

            # === 3. CYLINDRICAL NECK ===
            y_neck_start = p.center[1] - r_tube - saddle_height
            y_neck_end = y_neck_start - neck_length

            neck_start = len(vertices)
            # Start ring
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = x_center + neck_radius * np.cos(theta)
                z = z_center + neck_radius * np.sin(theta)
                vertices.append([x, y_neck_start, z])
                normals.append([np.cos(theta), 0.0, np.sin(theta)])

            # End ring
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = x_center + neck_radius * np.cos(theta)
                z = z_center + neck_radius * np.sin(theta)
                vertices.append([x, y_neck_end, z])
                normals.append([np.cos(theta), 0.0, np.sin(theta)])

            # === 4. FLANGE ===
            flange_start = len(vertices)
            flange_radius = neck_radius * 1.3
            y_flange = y_neck_end

            # Flange inner ring
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = x_center + neck_radius * np.cos(theta)
                z = z_center + neck_radius * np.sin(theta)
                vertices.append([x, y_flange, z])
                normals.append([0.0, -1.0, 0.0])

            # Flange outer ring
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = x_center + flange_radius * np.cos(theta)
                z = z_center + flange_radius * np.sin(theta)
                vertices.append([x, y_flange, z])
                normals.append([0.0, -1.0, 0.0])

            # === GENERATE TRIANGLES ===
            # Saddle to first transition ring (reversed winding for bottom)
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = saddle_start + j
                v1 = saddle_start + j_next
                v2 = saddle_start + n_radial + j_next
                v3 = saddle_start + n_radial + j
                indices.extend([v0, v2, v1])
                indices.extend([v0, v3, v2])

            # Between transition rings
            for t in range(1, n_transition):
                for j in range(n_radial):
                    j_next = (j + 1) % n_radial
                    base = saddle_start + t * n_radial
                    v0 = base + j
                    v1 = base + j_next
                    v2 = base + n_radial + j_next
                    v3 = base + n_radial + j
                    indices.extend([v0, v2, v1])
                    indices.extend([v0, v3, v2])

            # Last transition to neck start
            trans_last = saddle_start + n_transition * n_radial
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = trans_last + j
                v1 = trans_last + j_next
                v2 = neck_start + j_next
                v3 = neck_start + j
                indices.extend([v0, v2, v1])
                indices.extend([v0, v3, v2])

            # Neck cylinder
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = neck_start + j
                v1 = neck_start + j_next
                v2 = neck_start + n_radial + j_next
                v3 = neck_start + n_radial + j
                indices.extend([v0, v2, v1])
                indices.extend([v0, v3, v2])

            # Flange face (annular ring, facing down)
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v_inner = flange_start + j
                v_inner_next = flange_start + j_next
                v_outer = flange_start + n_radial + j
                v_outer_next = flange_start + n_radial + j_next
                indices.extend([v_inner, v_outer_next, v_inner_next])
                indices.extend([v_inner, v_outer, v_outer_next])

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

    @property
    def ports(self) -> Dict[str, ConnectionPort]:
        """
        Get connection ports for this component.

        The port positions represent the ACTUAL CONNECTION SURFACES where
        components physically meet (at flange faces).

        ENCLOSED TUBE CONNECTION POINTS with SADDLE JOINTS:
        - Material enters through TOP (inlet) from rotary airlock
        - Material exits through BOTTOM (outlet) to deagglomerator
        - Both connections use saddle joints that curve around the tube
        - Flanged ends for dust-tight sealing

        DYNAMIC PROPORTIONAL SIZING:
        - inlet_diameter: Direct from params, matches airlock outlet
        - outlet_diameter: Direct from params, matches deagg inlet
        - Port positions match the geometry in _generate_inlet/_generate_outlet

        Returns:
            Dictionary of port name to ConnectionPort:
            - 'inlet': Top inlet flange (from airlock)
            - 'outlet': Bottom outlet flange (to deagglomerator)
        """
        p = self.params

        # Dimensions must match _generate_inlet/_generate_outlet
        r_tube = p.trough_radius + 0.003
        saddle_height = 0.010
        inlet_neck_length = p.inlet_diameter * 0.4
        outlet_neck_length = p.outlet_diameter * 0.4

        # Inlet position: saddle + neck from tube surface
        inlet_top_y = r_tube + saddle_height + inlet_neck_length

        # Outlet position: saddle + neck from tube surface (bottom)
        outlet_bottom_y = -(r_tube + saddle_height + outlet_neck_length)

        if p.axis == "x":
            # Inlet at 15% of trough length (matches geometry)
            inlet_x = p.trough_length * 0.15
            inlet_pos = (inlet_x, inlet_top_y, 0.0)
            inlet_dir = (0.0, 1.0, 0.0)  # Points up

            # Outlet at 85% of trough length (matches geometry)
            outlet_x = p.trough_length * 0.85
            outlet_pos = (outlet_x, outlet_bottom_y, 0.0)
            outlet_dir = (0.0, -1.0, 0.0)  # Points down
        elif p.axis == "y":
            inlet_pos = (inlet_top_y, p.trough_length * 0.15, 0.0)
            inlet_dir = (1.0, 0.0, 0.0)
            outlet_pos = (outlet_bottom_y, p.trough_length * 0.85, 0.0)
            outlet_dir = (-1.0, 0.0, 0.0)
        else:  # z-axis
            inlet_pos = (0.0, inlet_top_y, p.trough_length * 0.15)
            inlet_dir = (0.0, 1.0, 0.0)
            outlet_pos = (0.0, outlet_bottom_y, p.trough_length * 0.85)
            outlet_dir = (0.0, -1.0, 0.0)

        return {
            'inlet': ConnectionPort(
                position=inlet_pos,
                direction=inlet_dir,
                diameter=p.inlet_diameter,
                port_type=PortType.FLANGED,
                name="feeder_inlet",
                flange_diameter=p.inlet_diameter * 1.3,
                compatible_types=[PortType.CIRCULAR, PortType.GRAVITY, PortType.FLANGED],
            ),
            'outlet': ConnectionPort(
                position=outlet_pos,
                direction=outlet_dir,
                diameter=p.outlet_diameter,
                port_type=PortType.FLANGED,
                name="feeder_outlet",
                flange_diameter=p.outlet_diameter * 1.3,
                compatible_types=[PortType.CIRCULAR, PortType.GRAVITY, PortType.FLANGED],
            ),
        }


def create_standard_screw_feeder(
    screw_diameter: float = 0.10,
    feed_rate_kg_h: float = 500,
    bulk_density: float = 500,
    inlet_diameter: float = None,
    outlet_diameter: float = None,
) -> ScrewFeeder:
    """
    Create a standard screw feeder sized for given feed rate.
    
    DYNAMIC PROPORTIONAL SIZING:
    The inlet and outlet diameters can be specified directly to match
    the connected components (airlock outlet and de-agglomerator inlet).
    If not specified, defaults are calculated proportional to screw diameter.

    Args:
        screw_diameter: Screw diameter [m]
        feed_rate_kg_h: Target feed rate [kg/h]
        bulk_density: Material bulk density [kg/m^3]
        inlet_diameter: Inlet port diameter [m] (default: screw_diameter * 1.2)
        outlet_diameter: Outlet port diameter [m] (default: screw_diameter * 0.8)

    Returns:
        ScrewFeeder instance
    """
    # Standard proportions
    shaft_diameter = screw_diameter * 0.3
    screw_pitch = screw_diameter * 0.8  # Standard pitch

    # Calculate length for reasonable fill level and RPM
    rpm = 30
    fill_level = 0.30

    # Length sized for ~3 pitches (standard)
    trough_length = screw_pitch * 3
    
    # Default inlet/outlet diameters if not specified
    if inlet_diameter is None:
        inlet_diameter = screw_diameter * 1.2
    if outlet_diameter is None:
        outlet_diameter = screw_diameter * 0.8

    params = ScrewFeederParams(
        screw_diameter=screw_diameter,
        shaft_diameter=shaft_diameter,
        screw_pitch=screw_pitch,
        flight_thickness=0.003,
        trough_length=trough_length,
        trough_clearance=0.003,
        inlet_diameter=inlet_diameter,
        outlet_diameter=outlet_diameter,
        rpm=rpm,
        fill_level=fill_level,
    )

    return ScrewFeeder(params)
