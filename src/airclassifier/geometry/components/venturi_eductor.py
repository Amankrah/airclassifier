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
    solids_inlet_angle: float    # [rad] Angle of solids entry from radial (tilt angle)
    solids_inlet_position: float # [m] Distance from throat start to solids entry
    
    # Angular position of solids inlet around the venturi axis
    # For axis='y': 0=+X side, π/2=+Z side, π=-X side, 3π/2=-Z side
    solids_inlet_angular_position: float = 0.0  # [rad] Default: +X side

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
        """
        Add solids inlet tube geometry with realistic industrial features.
        
        Creates a tangential inlet tube with:
        - Proper wall thickness (schedule 40 pipe proportions)
        - Weld fillet at junction with main body
        - Flange collar at outer end for connection
        - Correct orientation for each axis configuration
        
        For axis='y' (vertical venturi):
        - Air enters from -Y (bottom), exits +Y (top)
        - Solids inlet comes from +X direction, angled upward for gravity feed
        """
        p = self.params

        # Position of solids inlet along main axis
        axial_pos = p.throat_start_position + p.solids_inlet_position
        r_main = p.throat_diameter / 2
        r_inlet_outer = p.solids_inlet_diameter / 2
        
        # Wall thickness (typical schedule 40 proportions)
        wall_thickness = max(r_inlet_outer * 0.15, 0.002)  # Min 2mm wall
        r_inlet_inner = r_inlet_outer - wall_thickness

        # Inlet tube length - extends outward for connection
        tube_length = p.throat_diameter * 2.5
        
        # Weld fillet dimensions
        weld_height = wall_thickness * 2.0
        weld_radius = wall_thickness * 1.5
        
        # Flange collar at outer end
        flange_thickness = wall_thickness * 2
        flange_radius = r_inlet_outer * 1.3

        n_radial = max(p.resolution_radial // 2, 12)
        n_axial = 5  # Sections along tube length
        
        # Entry angle from the radial direction
        entry_angle = p.solids_inlet_angle

        # ================================================================
        # Generate inlet tube geometry based on venturi axis
        # ================================================================
        
        if p.axis == "x":
            # Horizontal venturi - flow along X
            # Solids inlet extends in Y-Z plane at angle
            self._generate_inlet_tube_x(
                vertices, indices, normals,
                axial_pos, r_main, r_inlet_outer, r_inlet_inner,
                tube_length, entry_angle, weld_height, weld_radius,
                flange_thickness, flange_radius, n_radial, n_axial
            )
            
        elif p.axis == "y":
            # Vertical venturi - flow along Y (main air from -Y)
            # Solids inlet from +X side, angled for gravity feed from +Z
            self._generate_inlet_tube_y(
                vertices, indices, normals,
                axial_pos, r_main, r_inlet_outer, r_inlet_inner,
                tube_length, entry_angle, weld_height, weld_radius,
                flange_thickness, flange_radius, n_radial, n_axial
            )
            
        elif p.axis == "z":
            # Flow along Z
            self._generate_inlet_tube_z(
                vertices, indices, normals,
                axial_pos, r_main, r_inlet_outer, r_inlet_inner,
                tube_length, entry_angle, weld_height, weld_radius,
                flange_thickness, flange_radius, n_radial, n_axial
            )

    def _generate_inlet_tube_y(self, vertices, indices, normals,
                                axial_pos, r_main, r_outer, r_inner,
                                tube_length, entry_angle, weld_h, weld_r,
                                flange_t, flange_r, n_radial, n_axial):
        """
        Generate solids inlet tube for vertical venturi (axis='y').
        
        COORDINATE SYSTEM (for axis='y'):
        - X+: Horizontal (from air filter toward deagglomerator)
        - Y+: Vertical (upward, venturi axis direction)
        - Z+: Horizontal (away from classification system, toward feed)
        
        The inlet tube extends radially outward from the venturi at the
        angular position specified by solids_inlet_angular_position:
        - angular_pos=0 → inlet on +X side
        - angular_pos=π/2 → inlet on +Z side (toward feed system)
        - angular_pos=π → inlet on -X side
        - angular_pos=3π/2 → inlet on -Z side
        
        Entry angle tilts the tube upward/downward for gravity feed alignment.
        """
        p = self.params
        start_idx = len(vertices)
        
        # Y position along venturi
        y_center = p.center[1] + axial_pos
        
        # Angular position around venturi (where inlet is located)
        angular_pos = getattr(p, 'solids_inlet_angular_position', 0.0)
        cos_ang = np.cos(angular_pos)  # X component
        sin_ang = np.sin(angular_pos)  # Z component
        
        # Entry angle tilts the tube upward (+Y) for receiving angled feed
        cos_a = np.cos(entry_angle)
        sin_a = np.sin(entry_angle)
        
        # Tube direction: extends radially outward at angular_pos, 
        # with upward tilt based on entry_angle
        # Radial direction (in X-Z plane): (cos_ang, 0, sin_ang)
        # Tilt: reduce radial component by cos_a, add Y component sin_a
        tube_dir = np.array([cos_ang * cos_a, sin_a, sin_ang * cos_a])
        tube_dir = tube_dir / np.linalg.norm(tube_dir)
        
        # Starting point at main body surface (at angular position)
        junction_x = p.center[0] + r_main * cos_ang
        junction_z = p.center[2] + r_main * sin_ang
        
        # Create local coordinate system for tube cross-section
        # tube_dir is axial, need two perpendicular vectors
        up = np.array([0.0, 1.0, 0.0])  # Y is up
        tangent = np.cross(tube_dir, up)
        if np.linalg.norm(tangent) < 0.01:
            # tube_dir is nearly vertical, use different reference
            tangent = np.array([-sin_ang, 0.0, cos_ang])  # perpendicular in X-Z plane
        tangent = tangent / np.linalg.norm(tangent)
        binormal = np.cross(tangent, tube_dir)
        binormal = binormal / np.linalg.norm(binormal)
        
        # ============================================================
        # Generate tube sections from junction to outer end
        # ============================================================
        
        # Axial positions along tube
        sections = []
        
        # Section 0: At junction with weld fillet (inner edge)
        sections.append({
            't': 0.0,
            'r_out': r_outer + weld_h,
            'r_in': r_inner,
        })
        
        # Section 1: Just past weld fillet
        sections.append({
            't': weld_r * 2,
            'r_out': r_outer,
            'r_in': r_inner,
        })
        
        # Section 2-3: Main tube body
        sections.append({
            't': tube_length * 0.3,
            'r_out': r_outer,
            'r_in': r_inner,
        })
        sections.append({
            't': tube_length * 0.7,
            'r_out': r_outer,
            'r_in': r_inner,
        })
        
        # Section 4: Flange start
        sections.append({
            't': tube_length - flange_t,
            'r_out': r_outer,
            'r_in': r_inner,
        })
        
        # Section 5: Flange outer edge
        sections.append({
            't': tube_length,
            'r_out': flange_r,
            'r_in': r_inner,
        })
        
        # Generate vertices for each section
        for sec in sections:
            t = sec['t']
            r_out = sec['r_out']
            r_in = sec['r_in']
            
            # Center point of this section
            cx = junction_x + t * tube_dir[0]
            cy = y_center + t * tube_dir[1]
            cz = junction_z + t * tube_dir[2]
            
            # Outer ring
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                # Position in tube cross-section plane
                offset = r_out * (np.cos(theta) * tangent + np.sin(theta) * binormal)
                vx = cx + offset[0]
                vy = cy + offset[1]
                vz = cz + offset[2]
                vertices.append([vx, vy, vz])
                
                # Normal pointing outward
                n_vec = np.cos(theta) * tangent + np.sin(theta) * binormal
                normals.append(n_vec.tolist())
            
            # Inner ring (for wall thickness)
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                offset = r_in * (np.cos(theta) * tangent + np.sin(theta) * binormal)
                vx = cx + offset[0]
                vy = cy + offset[1]
                vz = cz + offset[2]
                vertices.append([vx, vy, vz])
                
                # Normal pointing inward
                n_vec = -(np.cos(theta) * tangent + np.sin(theta) * binormal)
                normals.append(n_vec.tolist())
        
        # ============================================================
        # Generate triangles
        # ============================================================
        n_sections = len(sections)
        verts_per_section = n_radial * 2  # outer + inner rings
        
        # Outer surface triangles (between sections)
        for s in range(n_sections - 1):
            base0 = start_idx + s * verts_per_section
            base1 = start_idx + (s + 1) * verts_per_section
            
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                
                # Outer surface quad
                v0 = base0 + j
                v1 = base0 + j_next
                v2 = base1 + j_next
                v3 = base1 + j
                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])
                
                # Inner surface quad (reversed winding)
                vi0 = base0 + n_radial + j
                vi1 = base0 + n_radial + j_next
                vi2 = base1 + n_radial + j_next
                vi3 = base1 + n_radial + j
                indices.extend([vi0, vi2, vi1])
                indices.extend([vi0, vi3, vi2])
        
        # End cap at flange (outer ring to inner ring)
        last_section_base = start_idx + (n_sections - 1) * verts_per_section
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            vo0 = last_section_base + j
            vo1 = last_section_base + j_next
            vi0 = last_section_base + n_radial + j
            vi1 = last_section_base + n_radial + j_next
            indices.extend([vo0, vo1, vi1])
            indices.extend([vo0, vi1, vi0])

    def _generate_inlet_tube_x(self, vertices, indices, normals,
                                axial_pos, r_main, r_outer, r_inner,
                                tube_length, entry_angle, weld_h, weld_r,
                                flange_t, flange_r, n_radial, n_axial):
        """Generate solids inlet tube for horizontal venturi (axis='x')."""
        p = self.params
        start_idx = len(vertices)
        
        x_center = p.center[0] + axial_pos
        cos_a = np.cos(entry_angle)
        sin_a = np.sin(entry_angle)
        
        # Tube extends in Y-Z plane
        tube_dir = np.array([0.0, cos_a, sin_a])
        tube_dir = tube_dir / np.linalg.norm(tube_dir)
        
        junction_y = p.center[1] + r_main
        junction_z = p.center[2]
        
        # Local coordinate system
        tangent = np.array([1.0, 0.0, 0.0])
        binormal = np.cross(tangent, tube_dir)
        binormal = binormal / np.linalg.norm(binormal)
        
        sections = [
            {'t': 0.0, 'r_out': r_outer + weld_h, 'r_in': r_inner},
            {'t': weld_r * 2, 'r_out': r_outer, 'r_in': r_inner},
            {'t': tube_length * 0.5, 'r_out': r_outer, 'r_in': r_inner},
            {'t': tube_length - flange_t, 'r_out': r_outer, 'r_in': r_inner},
            {'t': tube_length, 'r_out': flange_r, 'r_in': r_inner},
        ]
        
        for sec in sections:
            t = sec['t']
            r_out, r_in = sec['r_out'], sec['r_in']
            cx = x_center
            cy = junction_y + t * tube_dir[1]
            cz = junction_z + t * tube_dir[2]
            
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                offset = r_out * (np.cos(theta) * tangent + np.sin(theta) * binormal)
                vertices.append([cx + offset[0], cy + offset[1], cz + offset[2]])
                n_vec = np.cos(theta) * tangent + np.sin(theta) * binormal
                normals.append(n_vec.tolist())
            
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                offset = r_in * (np.cos(theta) * tangent + np.sin(theta) * binormal)
                vertices.append([cx + offset[0], cy + offset[1], cz + offset[2]])
                n_vec = -(np.cos(theta) * tangent + np.sin(theta) * binormal)
                normals.append(n_vec.tolist())
        
        self._generate_tube_triangles(indices, start_idx, len(sections), n_radial)

    def _generate_inlet_tube_z(self, vertices, indices, normals,
                                axial_pos, r_main, r_outer, r_inner,
                                tube_length, entry_angle, weld_h, weld_r,
                                flange_t, flange_r, n_radial, n_axial):
        """Generate solids inlet tube for venturi with flow along Z-axis."""
        p = self.params
        start_idx = len(vertices)
        
        z_center = p.center[2] + axial_pos
        cos_a = np.cos(entry_angle)
        sin_a = np.sin(entry_angle)
        
        # Tube extends in X-Y plane
        tube_dir = np.array([cos_a, sin_a, 0.0])
        tube_dir = tube_dir / np.linalg.norm(tube_dir)
        
        junction_x = p.center[0] + r_main
        junction_y = p.center[1]
        
        # Local coordinate system
        tangent = np.array([0.0, 0.0, 1.0])
        binormal = np.cross(tangent, tube_dir)
        binormal = binormal / np.linalg.norm(binormal)
        
        sections = [
            {'t': 0.0, 'r_out': r_outer + weld_h, 'r_in': r_inner},
            {'t': weld_r * 2, 'r_out': r_outer, 'r_in': r_inner},
            {'t': tube_length * 0.5, 'r_out': r_outer, 'r_in': r_inner},
            {'t': tube_length - flange_t, 'r_out': r_outer, 'r_in': r_inner},
            {'t': tube_length, 'r_out': flange_r, 'r_in': r_inner},
        ]
        
        for sec in sections:
            t = sec['t']
            r_out, r_in = sec['r_out'], sec['r_in']
            cx = junction_x + t * tube_dir[0]
            cy = junction_y + t * tube_dir[1]
            cz = z_center
            
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                offset = r_out * (np.cos(theta) * tangent + np.sin(theta) * binormal)
                vertices.append([cx + offset[0], cy + offset[1], cz + offset[2]])
                n_vec = np.cos(theta) * tangent + np.sin(theta) * binormal
                normals.append(n_vec.tolist())
            
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                offset = r_in * (np.cos(theta) * tangent + np.sin(theta) * binormal)
                vertices.append([cx + offset[0], cy + offset[1], cz + offset[2]])
                n_vec = -(np.cos(theta) * tangent + np.sin(theta) * binormal)
                normals.append(n_vec.tolist())
        
        self._generate_tube_triangles(indices, start_idx, len(sections), n_radial)

    def _generate_tube_triangles(self, indices, start_idx, n_sections, n_radial):
        """Generate triangles for tube geometry."""
        verts_per_section = n_radial * 2
        
        for s in range(n_sections - 1):
            base0 = start_idx + s * verts_per_section
            base1 = start_idx + (s + 1) * verts_per_section
            
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                
                # Outer surface
                v0, v1 = base0 + j, base0 + j_next
                v2, v3 = base1 + j_next, base1 + j
                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])
                
                # Inner surface (reversed winding)
                vi0, vi1 = base0 + n_radial + j, base0 + n_radial + j_next
                vi2, vi3 = base1 + n_radial + j_next, base1 + n_radial + j
                indices.extend([vi0, vi2, vi1])
                indices.extend([vi0, vi3, vi2])
        
        # End cap
        last_base = start_idx + (n_sections - 1) * verts_per_section
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            vo0, vo1 = last_base + j, last_base + j_next
            vi0, vi1 = last_base + n_radial + j, last_base + n_radial + j_next
            indices.extend([vo0, vo1, vi1])
            indices.extend([vo0, vi1, vi0])

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
        
        # Solids inlet at throat - positioned at outer end of inlet tube
        # The inlet tube extends outward with an angle for gravity feed
        axial_pos = p.throat_start_position + p.solids_inlet_position
        r_main = p.throat_diameter / 2
        tube_length = p.throat_diameter * 2.5  # Match geometry generation
        entry_angle = p.solids_inlet_angle
        cos_a = np.cos(entry_angle)
        sin_a = np.sin(entry_angle)
        
        # Angular position around venturi axis
        angular_pos = getattr(p, 'solids_inlet_angular_position', 0.0)
        cos_ang = np.cos(angular_pos)
        sin_ang = np.sin(angular_pos)
        
        if p.axis == "x":
            # Horizontal venturi - solids inlet extends in Y-Z plane
            solids_pos = (
                p.center[0] + axial_pos,
                p.center[1] + r_main + tube_length * cos_a,
                p.center[2] + tube_length * sin_a
            )
            # Direction points outward from tube (toward feed source)
            norm = np.sqrt(cos_a**2 + sin_a**2)
            solids_dir = (0.0, cos_a / norm, sin_a / norm)
            
        elif p.axis == "y":
            # Vertical venturi - air from -Y, outlet to +Y
            # Solids inlet extends radially at angular_pos with entry_angle tilt
            # 
            # COORDINATE SYSTEM:
            # - X+: Horizontal (toward deagglomerator)
            # - Y+: Vertical (upward, venturi axis)
            # - Z+: Horizontal (away from classifier, toward feed)
            # 
            # Tube direction: radial outward at angular_pos, tilted up by entry_angle
            tube_dir = np.array([cos_ang * cos_a, sin_a, sin_ang * cos_a])
            tube_dir = tube_dir / np.linalg.norm(tube_dir)
            
            # Position: starts at venturi surface, extends along tube_dir
            solids_pos = (
                p.center[0] + r_main * cos_ang + tube_length * tube_dir[0],
                p.center[1] + axial_pos + tube_length * tube_dir[1],
                p.center[2] + r_main * sin_ang + tube_length * tube_dir[2]
            )
            # Direction points outward along tube axis (toward feed source)
            solids_dir = tuple(tube_dir)
            
        else:  # axis == "z"
            # Flow along Z - solids inlet extends in X-Y plane
            solids_pos = (
                p.center[0] + r_main + tube_length * cos_a,
                p.center[1] + tube_length * sin_a,
                p.center[2] + axial_pos
            )
            norm = np.sqrt(cos_a**2 + sin_a**2)
            solids_dir = (cos_a / norm, sin_a / norm, 0.0)
        
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
    throat_ratio: float = 0.5,
    solids_inlet_angular_position: float = PI / 2,  # Default: +Z side for feed from +Z
    solids_inlet_angle: float = None,  # Tilt angle (default: 15 degrees)
) -> VenturiEducator:
    """
    Create a standard venturi eductor with typical proportions.

    Args:
        inlet_diameter: Inlet diameter [m]
        throat_ratio: Throat to inlet diameter ratio (0.3-0.6 typical)
        solids_inlet_angular_position: Angular position around venturi axis [rad]
            - 0 = +X side
            - π/2 = +Z side (default, toward feed system)
            - π = -X side
            - 3π/2 = -Z side
        solids_inlet_angle: Tilt angle of inlet tube [rad] (default: 15 degrees)

    Returns:
        VenturiEducator instance
    """
    throat_d = inlet_diameter * throat_ratio
    
    if solids_inlet_angle is None:
        solids_inlet_angle = np.radians(15)  # 15° default for angled feed shaft

    params = VenturiEducatorParams(
        inlet_diameter=inlet_diameter,
        throat_diameter=throat_d,
        outlet_diameter=inlet_diameter * 0.9,  # Slightly smaller outlet
        convergent_angle=np.radians(12),       # 12° half-angle
        divergent_angle=np.radians(5),         # 5° half-angle for good recovery
        solids_inlet_diameter=throat_d * 0.8,
        solids_inlet_angle=solids_inlet_angle,
        solids_inlet_position=throat_d * 0.3,  # Just into throat
        solids_inlet_angular_position=solids_inlet_angular_position,
    )

    return VenturiEducator(params)
