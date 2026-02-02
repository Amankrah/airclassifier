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
from typing import Tuple, List, Dict
import numpy as np
import warp as wp

from ...utils.constants import PI, TWO_PI
from ..connection_ports import ConnectionPort, PortType


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

    # Connections - circular neck diameters for flanged connections
    inlet_diameter: float = None   # [m] Inlet neck diameter (matches hopper discharge)
    outlet_diameter: float = None  # [m] Outlet neck diameter (matches feeder inlet)
    
    # Legacy rectangular dimensions (used for sizing if diameters not specified)
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
        
        # Set circular neck diameters to match rotor pocket opening
        # This should connect properly to hopper discharge and feeder inlet
        if self.inlet_diameter is None:
            # Inlet diameter sized for rotor opening area
            self.inlet_diameter = min(self.inlet_width, self.inlet_length) * 1.0
        if self.outlet_diameter is None:
            self.outlet_diameter = min(self.outlet_width, self.outlet_length) * 1.0

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
        """
        Generate cylindrical housing with openings for inlet and outlet.
        
        Real rotary airlocks have:
        - Horizontal cylindrical housing (rotor axis along Z)
        - Inlet opening on TOP (+Y direction) 
        - Outlet opening on BOTTOM (-Y direction)
        - The openings are where material enters/exits
        """
        p = self.params
        n_radial = p.resolution_radial
        n_axial = p.resolution_axial

        start_idx = len(vertices)
        r = p.housing_outer_radius
        half_length = p.rotor_length / 2 + p.housing_thickness
        
        # Calculate the angular span of inlet/outlet openings
        # The opening is a circular hole on top/bottom of the cylinder
        inlet_radius = p.inlet_diameter / 2
        outlet_radius = p.outlet_diameter / 2
        
        # Angular span of the inlet opening (at top, Y+)
        # sin(theta) = opening_radius / housing_radius
        inlet_half_angle = np.arcsin(min(0.9, inlet_radius / r))
        outlet_half_angle = np.arcsin(min(0.9, outlet_radius / r))

        # Generate housing cylinder with cutouts for inlet/outlet
        # We'll generate the housing in sections, skipping the inlet/outlet regions
        for i in range(n_axial + 1):
            t = i / n_axial
            if p.axis == "z":
                z = p.center[2] - half_length + t * 2 * half_length
                # Check if we're in the inlet/outlet region (middle of housing)
                z_from_center = abs(z - p.center[2])
                in_opening_region = z_from_center < min(inlet_radius, outlet_radius, half_length * 0.8)
                
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    
                    # Skip vertices in the inlet region (top, around PI/2)
                    # Skip vertices in the outlet region (bottom, around 3*PI/2 = -PI/2)
                    skip_inlet = in_opening_region and abs(theta - PI/2) < inlet_half_angle
                    skip_outlet = in_opening_region and abs(theta - 3*PI/2) < outlet_half_angle
                    
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

        # Generate triangles (excluding opening regions handled by saddle joints)
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
        """
        Generate inlet and outlet with saddle joints connecting to housing.

        Real rotary airlock design:
        - Inlet on TOP (+Y) where material falls in
        - Outlet on BOTTOM (-Y) where material exits
        - Saddle joint connects circular pipe to cylindrical housing
        - Flanged necks for bolted connections
        """
        p = self.params

        # Neck length from housing surface to flange face
        neck_length = p.housing_thickness * 4.0
        
        # Inlet (top, +Y for z-axis rotation)
        self._add_saddle_neck(vertices, indices, normals,
                             is_inlet=True, 
                             diameter=p.inlet_diameter,
                             length=neck_length)

        # Outlet (bottom, -Y for z-axis rotation)
        self._add_saddle_neck(vertices, indices, normals,
                             is_inlet=False,
                             diameter=p.outlet_diameter,
                             length=neck_length)

    def _add_saddle_neck(self, vertices: List, indices: List, normals: List,
                        is_inlet: bool, diameter: float, length: float):
        """
        Add inlet/outlet neck with saddle joint connecting to curved housing.
        
        This creates a realistic connection where:
        1. Saddle base sits on the curved housing surface
        2. Transition piece connects saddle to circular neck
        3. Circular neck extends outward
        4. Flange at the outer end
        """
        p = self.params
        n_radial = max(20, p.resolution_radial)
        n_transition = 4  # Segments for saddle-to-neck transition
        
        neck_radius = diameter / 2
        r_housing = p.housing_outer_radius
        sign = 1 if is_inlet else -1

        # Saddle dimensions - base sits on housing curve
        saddle_height = p.housing_thickness * 1.5  # Height of saddle transition
        
        if p.axis == "z":
            # For Z-axis rotation: inlet is +Y, outlet is -Y
            # The saddle base follows the housing curvature
            
            # === 1. SADDLE BASE (sits on housing curve) ===
            saddle_start = len(vertices)
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                # Position in XZ plane (neck cross-section)
                x_local = neck_radius * np.cos(theta)
                z_local = neck_radius * np.sin(theta)
                
                # The saddle base follows the housing curve
                # y position varies based on x position (curving over the cylinder)
                y_offset = np.sqrt(max(0, r_housing**2 - x_local**2))
                y = p.center[1] + sign * y_offset
                
                x = p.center[0] + x_local
                z = p.center[2] + z_local
                
                vertices.append([x, y, z])
                # Normal points along saddle surface (mostly radial on housing)
                norm_len = np.sqrt(x_local**2 + y_offset**2)
                if norm_len > 0.001:
                    normals.append([x_local/norm_len * 0.3, sign * y_offset/norm_len, 0.0])
                else:
                    normals.append([0.0, sign, 0.0])
            
            # === 2. TRANSITION RINGS (saddle to circular) ===
            for t in range(1, n_transition + 1):
                blend = t / n_transition
                ring_start = len(vertices)
                
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x_local = neck_radius * np.cos(theta)
                    z_local = neck_radius * np.sin(theta)
                    
                    # Blend from saddle curve to flat circle
                    y_saddle = np.sqrt(max(0, r_housing**2 - x_local**2))
                    y_flat = r_housing + saddle_height * blend
                    y = p.center[1] + sign * (y_saddle * (1 - blend) + y_flat * blend)
                    
                    x = p.center[0] + x_local
                    z = p.center[2] + z_local
                    
                    vertices.append([x, y, z])
                    normals.append([np.cos(theta), 0.0, np.sin(theta)])
            
            # === 3. CYLINDRICAL NECK ===
            y_neck_start = p.center[1] + sign * (r_housing + saddle_height)
            y_neck_end = y_neck_start + sign * length
            
            neck_start = len(vertices)
            # Start ring (connects to transition)
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + neck_radius * np.cos(theta)
                z = p.center[2] + neck_radius * np.sin(theta)
                vertices.append([x, y_neck_start, z])
                normals.append([np.cos(theta), 0.0, np.sin(theta)])
            
            # End ring (where flange attaches)
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + neck_radius * np.cos(theta)
                z = p.center[2] + neck_radius * np.sin(theta)
                vertices.append([x, y_neck_end, z])
                normals.append([np.cos(theta), 0.0, np.sin(theta)])
            
            # === GENERATE TRIANGLES ===
            # Saddle to first transition ring
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = saddle_start + j
                v1 = saddle_start + j_next
                v2 = saddle_start + n_radial + j_next
                v3 = saddle_start + n_radial + j
                if sign > 0:
                    indices.extend([v0, v1, v2])
                    indices.extend([v0, v2, v3])
                else:
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
                    if sign > 0:
                        indices.extend([v0, v1, v2])
                        indices.extend([v0, v2, v3])
                    else:
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
                if sign > 0:
                    indices.extend([v0, v1, v2])
                    indices.extend([v0, v2, v3])
                else:
                    indices.extend([v0, v2, v1])
                    indices.extend([v0, v3, v2])
            
            # Neck cylinder
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = neck_start + j
                v1 = neck_start + j_next
                v2 = neck_start + n_radial + j_next
                v3 = neck_start + n_radial + j
                if sign > 0:
                    indices.extend([v0, v1, v2])
                    indices.extend([v0, v2, v3])
                else:
                    indices.extend([v0, v2, v1])
                    indices.extend([v0, v3, v2])
            
            # === 4. FLANGE ===
            self._add_flange_ring(vertices, indices, normals, 
                                 is_inlet, neck_radius, length + saddle_height)
        
        else:
            # For other axes - use simpler straight neck (can be expanded later)
            # This maintains backward compatibility
            start_idx = len(vertices)
            
            if p.axis == "y":
                z_base = p.center[2] + sign * r_housing
                z_outer = z_base + sign * length
                
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x = p.center[0] + neck_radius * np.cos(theta)
                    y = p.center[1] + neck_radius * np.sin(theta)
                    vertices.append([x, y, z_base])
                    normals.append([np.cos(theta), np.sin(theta), 0.0])
                
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x = p.center[0] + neck_radius * np.cos(theta)
                    y = p.center[1] + neck_radius * np.sin(theta)
                    vertices.append([x, y, z_outer])
                    normals.append([np.cos(theta), np.sin(theta), 0.0])
            else:  # x-axis
                y_base = p.center[1] + sign * r_housing
                y_outer = y_base + sign * length
                
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x = p.center[0] + neck_radius * np.cos(theta)
                    z = p.center[2] + neck_radius * np.sin(theta)
                    vertices.append([x, y_base, z])
                    normals.append([np.cos(theta), 0.0, np.sin(theta)])
                
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x = p.center[0] + neck_radius * np.cos(theta)
                    z = p.center[2] + neck_radius * np.sin(theta)
                    vertices.append([x, y_outer, z])
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
            
            self._add_flange_ring(vertices, indices, normals, 
                                 is_inlet, neck_radius, length)

    def _add_flange_ring(self, vertices: List, indices: List, normals: List,
                        is_inlet: bool, neck_radius: float, neck_length: float):
        """Add an annular flange ring at the end of the inlet/outlet neck.
        
        The flange is a donut shape with:
        - Inner diameter = neck diameter (the actual flow opening)
        - Outer diameter = flange diameter (for mounting)
        """
        p = self.params
        n_radial = max(16, p.resolution_radial // 2)
        
        r_housing = p.housing_outer_radius
        sign = 1 if is_inlet else -1
        
        # Inner radius is the neck (flow opening)
        inner_radius = neck_radius
        # Outer radius is the flange (mounting surface)
        outer_radius = neck_radius * 1.3
        flange_thickness = p.housing_thickness
        
        start_idx = len(vertices)

        if p.axis == "z":
            # Flange sits at the end of the neck (no gap)
            y_flange = p.center[1] + sign * (r_housing + neck_length)

            # Inner ring (at neck diameter - this is the visible opening)
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + inner_radius * np.cos(theta)
                z = p.center[2] + inner_radius * np.sin(theta)
                vertices.append([x, y_flange, z])
                normals.append([0.0, sign, 0.0])  # Face outward

            # Outer ring (flange edge)
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + outer_radius * np.cos(theta)
                z = p.center[2] + outer_radius * np.sin(theta)
                vertices.append([x, y_flange, z])
                normals.append([0.0, sign, 0.0])  # Face outward

        elif p.axis == "y":
            # Flange sits at the end of the neck (no gap)
            z_flange = p.center[2] + sign * (r_housing + neck_length)

            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + inner_radius * np.cos(theta)
                y = p.center[1] + inner_radius * np.sin(theta)
                vertices.append([x, y, z_flange])
                normals.append([0.0, 0.0, sign])

            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + outer_radius * np.cos(theta)
                y = p.center[1] + outer_radius * np.sin(theta)
                vertices.append([x, y, z_flange])
                normals.append([0.0, 0.0, sign])

        else:  # x-axis
            # Flange sits at the end of the neck (no gap)
            y_flange = p.center[1] + sign * (r_housing + neck_length)

            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + inner_radius * np.cos(theta)
                z = p.center[2] + inner_radius * np.sin(theta)
                vertices.append([x, y_flange, z])
                normals.append([0.0, sign, 0.0])

            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + outer_radius * np.cos(theta)
                z = p.center[2] + outer_radius * np.sin(theta)
                vertices.append([x, y_flange, z])
                normals.append([0.0, sign, 0.0])
        
        # Generate triangles for flange face (annular ring between inner and outer)
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v_inner = start_idx + j
            v_inner_next = start_idx + j_next
            v_outer = start_idx + n_radial + j
            v_outer_next = start_idx + n_radial + j_next
            
            if sign > 0:
                # Top face (looking down at inlet)
                indices.extend([v_inner, v_inner_next, v_outer_next])
                indices.extend([v_inner, v_outer_next, v_outer])
            else:
                # Bottom face (looking up at outlet)
                indices.extend([v_inner, v_outer_next, v_inner_next])
                indices.extend([v_inner, v_outer, v_outer_next])

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

    @property
    def ports(self) -> Dict[str, ConnectionPort]:
        """
        Get connection ports for this component.
        
        The port positions represent the ACTUAL CONNECTION SURFACES where
        components physically meet (end of inlet/outlet neck flanges).
        
        Returns:
            Dictionary of port name to ConnectionPort:
            - 'inlet': Top inlet for material from hopper (circular)
            - 'outlet': Bottom outlet to feeder/conveyor (circular)
        """
        p = self.params
        r = p.housing_outer_radius

        # Dimensions must match _add_saddle_neck
        saddle_height = p.housing_thickness * 1.5  # Height of saddle transition
        neck_length = p.housing_thickness * 4.0    # Length of cylindrical neck

        # Total extension from housing center to flange face (flange sits at neck end)
        total_extension = r + saddle_height + neck_length
        
        if p.axis == "z":
            # For z-axis rotation: inlet at +Y, outlet at -Y
            inlet_pos = (0.0, total_extension, 0.0)
            inlet_dir = (0.0, 1.0, 0.0)  # Points up
            outlet_pos = (0.0, -total_extension, 0.0)
            outlet_dir = (0.0, -1.0, 0.0)  # Points down
        elif p.axis == "y":
            # For y-axis rotation: inlet at +Z, outlet at -Z
            inlet_pos = (0.0, 0.0, total_extension)
            inlet_dir = (0.0, 0.0, 1.0)
            outlet_pos = (0.0, 0.0, -total_extension)
            outlet_dir = (0.0, 0.0, -1.0)
        else:  # x-axis
            inlet_pos = (0.0, total_extension, 0.0)
            inlet_dir = (0.0, 1.0, 0.0)
            outlet_pos = (0.0, -total_extension, 0.0)
            outlet_dir = (0.0, -1.0, 0.0)
        
        # Flange diameter is 1.3x neck diameter
        inlet_flange_dia = p.inlet_diameter * 1.3
        outlet_flange_dia = p.outlet_diameter * 1.3
        
        return {
            'inlet': ConnectionPort(
                position=inlet_pos,
                direction=inlet_dir,
                diameter=p.inlet_diameter,
                port_type=PortType.FLANGED,
                name="airlock_inlet",
                flange_diameter=inlet_flange_dia,
                compatible_types=[PortType.CIRCULAR, PortType.GRAVITY, PortType.FLANGED],
            ),
            'outlet': ConnectionPort(
                position=outlet_pos,
                direction=outlet_dir,
                diameter=p.outlet_diameter,
                port_type=PortType.FLANGED,
                name="airlock_outlet",
                flange_diameter=outlet_flange_dia,
                compatible_types=[PortType.CIRCULAR, PortType.GRAVITY, PortType.FLANGED],
            ),
        }


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
