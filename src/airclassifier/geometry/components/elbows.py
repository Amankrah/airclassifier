"""
Elbow and bend components for air classification systems.

This module provides elbow/bend geometries for ductwork,
supporting round and rectangular cross-sections with various
bend radii and optional turning vanes.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class ElbowParams:
    """
    Parameters for duct elbow/bend.
    
    Attributes:
        elbow_type: Type of elbow ("round", "rectangular", "mitered")
        diameter: Duct diameter for round elbows [m]
        width: Width for rectangular elbows [m]
        height: Height for rectangular elbows [m]
        bend_radius: Centerline bend radius [m]
        bend_angle: Bend angle [rad] (default π/2 = 90°)
        wall_thickness: Wall thickness [m]
        num_gores: Number of gores for mitered elbows
        turning_vanes: Whether to include turning vanes
        num_vanes: Number of turning vanes (if enabled)
        center: Center of the bend arc (x, y, z) [m]
        inlet_direction: Direction of incoming flow (dx, dy, dz)
        bend_axis: Axis of rotation for the bend (ax, ay, az)
    """
    elbow_type: str = "round"
    diameter: float = 0.2
    width: float = 0.3
    height: float = 0.2
    bend_radius: float = 0.3
    bend_angle: float = np.pi / 2  # 90 degrees
    wall_thickness: float = 0.002
    num_gores: int = 5
    turning_vanes: bool = False
    num_vanes: int = 3
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    inlet_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    bend_axis: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    
    @property
    def r_d_ratio(self) -> float:
        """Bend radius to diameter ratio (R/D)."""
        if self.elbow_type == "round":
            return self.bend_radius / self.diameter
        else:
            # Use hydraulic diameter
            dh = 2 * self.width * self.height / (self.width + self.height)
            return self.bend_radius / dh
    
    @property
    def is_tight_radius(self) -> bool:
        """Check if this is a tight radius bend (R/D < 1.5)."""
        return self.r_d_ratio < 1.5
    
    @property
    def arc_length(self) -> float:
        """Centerline arc length of the bend [m]."""
        return self.bend_radius * self.bend_angle
    
    @property
    def inlet_direction_normalized(self) -> Tuple[float, float, float]:
        """Normalized inlet direction."""
        d = np.array(self.inlet_direction)
        return tuple(d / np.linalg.norm(d))
    
    @property
    def bend_axis_normalized(self) -> Tuple[float, float, float]:
        """Normalized bend axis."""
        a = np.array(self.bend_axis)
        return tuple(a / np.linalg.norm(a))
    
    @property
    def outlet_direction(self) -> Tuple[float, float, float]:
        """Calculate outlet direction based on bend angle."""
        # Rotate inlet direction around bend axis by bend_angle
        inlet = np.array(self.inlet_direction_normalized)
        axis = np.array(self.bend_axis_normalized)
        angle = self.bend_angle
        
        # Rodrigues' rotation formula
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        outlet = (inlet * cos_a + 
                  np.cross(axis, inlet) * sin_a + 
                  axis * np.dot(axis, inlet) * (1 - cos_a))
        return tuple(outlet)
    
    def get_pressure_loss_coefficient(self) -> float:
        """
        Estimate pressure loss coefficient K for the elbow.
        
        Uses ASHRAE duct fitting database correlations.
        
        Returns:
            Loss coefficient K (ΔP = K * 0.5 * ρ * V²)
        """
        r_d = self.r_d_ratio
        angle_factor = self.bend_angle / (np.pi / 2)  # Relative to 90°
        
        if self.elbow_type == "round":
            if r_d >= 2.0:
                K_base = 0.11
            elif r_d >= 1.5:
                K_base = 0.14
            elif r_d >= 1.0:
                K_base = 0.22
            else:
                K_base = 0.50
        elif self.elbow_type == "mitered":
            # Mitered elbow depends on number of pieces
            if self.num_gores >= 5:
                K_base = 0.25
            elif self.num_gores >= 4:
                K_base = 0.30
            elif self.num_gores >= 3:
                K_base = 0.40
            else:
                K_base = 1.2  # Single miter
        else:  # rectangular
            aspect = self.width / self.height
            if r_d >= 1.5:
                K_base = 0.15 * aspect ** 0.25
            else:
                K_base = 0.30 * aspect ** 0.25
        
        # Adjust for turning vanes
        if self.turning_vanes:
            K_base *= 0.5
        
        return K_base * angle_factor


class Elbow:
    """
    Duct elbow/bend geometry for air classification systems.
    
    Generates mesh for elbows with round, rectangular, or mitered profiles.
    """
    
    def __init__(self, params: ElbowParams):
        """
        Initialize elbow.
        
        Args:
            params: Elbow parameters
        """
        self.params = params
        self._vertices: Optional[np.ndarray] = None
        self._indices: Optional[np.ndarray] = None
        self._normals: Optional[np.ndarray] = None
    
    @property
    def vertices(self) -> np.ndarray:
        """Get mesh vertices, generating if needed."""
        if self._vertices is None:
            self.generate_mesh()
        return self._vertices
    
    @property
    def indices(self) -> np.ndarray:
        """Get mesh indices, generating if needed."""
        if self._indices is None:
            self.generate_mesh()
        return self._indices
    
    @property
    def normals(self) -> np.ndarray:
        """Get mesh normals, generating if needed."""
        if self._normals is None:
            self.generate_mesh()
        return self._normals
    
    def generate_mesh(self, num_segments: int = 24,
                      num_arc_divisions: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triangular mesh for the elbow.
        
        Args:
            num_segments: Circumferential segments for round elbows
            num_arc_divisions: Divisions along the bend arc
            
        Returns:
            Tuple of (vertices, indices, normals)
        """
        p = self.params
        
        if p.elbow_type == "round":
            return self._generate_round_elbow(num_segments, num_arc_divisions)
        elif p.elbow_type == "mitered":
            return self._generate_mitered_elbow(num_segments)
        else:  # rectangular
            return self._generate_rectangular_elbow(num_arc_divisions)
    
    def _get_rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Get 3x3 rotation matrix for rotating around axis by angle."""
        axis = axis / np.linalg.norm(axis)
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        x, y, z = axis
        
        return np.array([
            [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
            [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
            [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
        ])
    
    def _generate_round_elbow(self, num_segments: int,
                               num_arc_divisions: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate mesh for round elbow (torus section)."""
        all_vertices = []
        all_indices = []
        all_normals = []
        
        p = self.params
        center = np.array(p.center)
        inlet_dir = np.array(p.inlet_direction_normalized)
        bend_axis = np.array(p.bend_axis_normalized)
        
        r_inner = p.diameter / 2
        r_outer = r_inner + p.wall_thickness
        R = p.bend_radius
        
        # Calculate the center of the bend arc
        # The arc center is offset from the elbow center perpendicular to inlet direction
        perp = np.cross(inlet_dir, bend_axis)
        perp = perp / np.linalg.norm(perp)
        arc_center = center + R * perp
        
        def add_torus_surface(radius: float, outward_normal: bool):
            """Add a torus-section surface."""
            nonlocal all_vertices, all_indices, all_normals
            base_idx = len(all_vertices)
            
            for i in range(num_arc_divisions + 1):
                # Angle along the bend arc
                arc_angle = p.bend_angle * i / num_arc_divisions
                
                # Get rotation matrix for this position along arc
                rot = self._get_rotation_matrix(bend_axis, arc_angle)
                
                # Direction from arc center to duct center at this arc position
                radial = -rot @ perp  # Points from arc center toward duct center
                
                # Duct center at this arc position
                duct_center = arc_center + R * (-radial)
                
                # Local coordinate system for the circular cross-section
                # tangent = direction along the duct at this point
                tangent = rot @ inlet_dir
                local_y = np.cross(tangent, radial)
                local_y = local_y / np.linalg.norm(local_y)
                
                for j in range(num_segments):
                    theta = 2 * np.pi * j / num_segments
                    
                    # Point on circle
                    local_offset = radius * (np.cos(theta) * radial + np.sin(theta) * local_y)
                    pt = duct_center + local_offset
                    all_vertices.append(list(pt))
                    
                    # Normal
                    norm = np.cos(theta) * radial + np.sin(theta) * local_y
                    if not outward_normal:
                        norm = -norm
                    all_normals.append(list(norm))
            
            # Triangles
            for i in range(num_arc_divisions):
                for j in range(num_segments):
                    i0 = base_idx + i * num_segments + j
                    i1 = base_idx + i * num_segments + (j + 1) % num_segments
                    i2 = base_idx + (i + 1) * num_segments + j
                    i3 = base_idx + (i + 1) * num_segments + (j + 1) % num_segments
                    
                    if outward_normal:
                        all_indices.extend([i0, i2, i1])
                        all_indices.extend([i1, i2, i3])
                    else:
                        all_indices.extend([i0, i1, i2])
                        all_indices.extend([i1, i3, i2])
        
        # Outer surface
        add_torus_surface(r_outer, True)
        
        # Inner surface
        add_torus_surface(r_inner, False)
        
        # End caps (annular rings at inlet and outlet)
        def add_end_cap(arc_angle: float, normal_positive: bool):
            nonlocal all_vertices, all_indices, all_normals
            base_idx = len(all_vertices)
            
            rot = self._get_rotation_matrix(bend_axis, arc_angle)
            radial = -rot @ perp
            duct_center = arc_center + R * (-radial)
            tangent = rot @ inlet_dir
            local_y = np.cross(tangent, radial)
            local_y = local_y / np.linalg.norm(local_y)
            
            norm = list(tangent) if normal_positive else list(-tangent)
            
            for radius in [r_inner, r_outer]:
                for j in range(num_segments):
                    theta = 2 * np.pi * j / num_segments
                    local_offset = radius * (np.cos(theta) * radial + np.sin(theta) * local_y)
                    pt = duct_center + local_offset
                    all_vertices.append(list(pt))
                    all_normals.append(norm)
            
            for j in range(num_segments):
                i0 = base_idx + j
                i1 = base_idx + (j + 1) % num_segments
                i2 = base_idx + num_segments + j
                i3 = base_idx + num_segments + (j + 1) % num_segments
                
                if normal_positive:
                    all_indices.extend([i0, i1, i2])
                    all_indices.extend([i1, i3, i2])
                else:
                    all_indices.extend([i0, i2, i1])
                    all_indices.extend([i1, i2, i3])
        
        # Inlet cap (at arc_angle = 0)
        add_end_cap(0.0, False)
        # Outlet cap (at arc_angle = bend_angle)
        add_end_cap(p.bend_angle, True)
        
        # Add turning vanes if specified
        if p.turning_vanes and p.num_vanes > 0:
            self._add_turning_vanes(all_vertices, all_indices, all_normals,
                                    arc_center, perp, num_arc_divisions)
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def _add_turning_vanes(self, all_vertices: list, all_indices: list, 
                           all_normals: list, arc_center: np.ndarray,
                           perp: np.ndarray, num_arc_divisions: int):
        """Add turning vanes to the elbow."""
        p = self.params
        bend_axis = np.array(p.bend_axis_normalized)
        inlet_dir = np.array(p.inlet_direction_normalized)
        
        r_inner = p.diameter / 2
        R = p.bend_radius
        vane_thickness = 0.002
        
        for v in range(p.num_vanes):
            # Position vane at fractional radius
            frac = (v + 1) / (p.num_vanes + 1)
            vane_radius = r_inner * frac
            
            base_idx = len(all_vertices)
            
            # Create vane surface along arc
            for i in range(num_arc_divisions + 1):
                arc_angle = p.bend_angle * i / num_arc_divisions
                rot = self._get_rotation_matrix(bend_axis, arc_angle)
                
                radial = -rot @ perp
                duct_center = arc_center + R * (-radial)
                
                # Vane follows the radial direction
                pt_inner = duct_center + vane_radius * radial
                pt_outer = duct_center + (vane_radius + vane_thickness) * radial
                
                # Normal perpendicular to vane surface
                tangent = rot @ inlet_dir
                local_y = np.cross(tangent, radial)
                local_y = local_y / np.linalg.norm(local_y)
                
                all_vertices.append(list(pt_inner))
                all_normals.append(list(local_y))
                all_vertices.append(list(pt_outer))
                all_normals.append(list(local_y))
            
            # Triangles for vane
            for i in range(num_arc_divisions):
                i0 = base_idx + i * 2
                i1 = base_idx + i * 2 + 1
                i2 = base_idx + (i + 1) * 2
                i3 = base_idx + (i + 1) * 2 + 1
                
                all_indices.extend([i0, i2, i1])
                all_indices.extend([i1, i2, i3])
    
    def _generate_mitered_elbow(self, num_segments: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate mesh for mitered elbow (segmented straight sections)."""
        all_vertices = []
        all_indices = []
        all_normals = []
        
        p = self.params
        center = np.array(p.center)
        inlet_dir = np.array(p.inlet_direction_normalized)
        bend_axis = np.array(p.bend_axis_normalized)
        
        r_inner = p.diameter / 2
        r_outer = r_inner + p.wall_thickness
        
        # Calculate miter angles
        num_miters = p.num_gores
        angle_per_section = p.bend_angle / num_miters
        
        # Generate each straight section
        current_dir = inlet_dir.copy()
        current_pos = center.copy()
        
        for section in range(num_miters):
            base_idx = len(all_vertices)
            
            # Length of this section (shorter sections for mitered)
            section_length = p.bend_radius * np.tan(angle_per_section / 2) * 2
            if section == 0 or section == num_miters - 1:
                section_length /= 2
            
            # End position of this section
            end_pos = current_pos + section_length * current_dir
            
            # Local coordinate system
            if abs(current_dir[2]) < 0.9:
                local_y = np.cross(current_dir, [0, 0, 1])
            else:
                local_y = np.cross(current_dir, [1, 0, 0])
            local_y = local_y / np.linalg.norm(local_y)
            local_z = np.cross(current_dir, local_y)
            
            # Add cylinder section
            for t in [0.0, 1.0]:
                pos = current_pos + t * section_length * current_dir
                for i in range(num_segments):
                    theta = 2 * np.pi * i / num_segments
                    
                    for radius, outward in [(r_outer, True), (r_inner, False)]:
                        offset = radius * (np.cos(theta) * local_y + np.sin(theta) * local_z)
                        pt = pos + offset
                        all_vertices.append(list(pt))
                        
                        norm = np.cos(theta) * local_y + np.sin(theta) * local_z
                        if not outward:
                            norm = -norm
                        all_normals.append(list(norm))
            
            # Triangles for outer surface
            n = num_segments * 2  # vertices per ring (inner + outer)
            for i in range(num_segments):
                # Outer surface
                i0 = base_idx + i * 2
                i1 = base_idx + ((i + 1) % num_segments) * 2
                i2 = base_idx + n + i * 2
                i3 = base_idx + n + ((i + 1) % num_segments) * 2
                
                all_indices.extend([i0, i2, i1])
                all_indices.extend([i1, i2, i3])
                
                # Inner surface
                i0 = base_idx + i * 2 + 1
                i1 = base_idx + ((i + 1) % num_segments) * 2 + 1
                i2 = base_idx + n + i * 2 + 1
                i3 = base_idx + n + ((i + 1) % num_segments) * 2 + 1
                
                all_indices.extend([i0, i1, i2])
                all_indices.extend([i1, i3, i2])
            
            # Update for next section
            current_pos = end_pos
            rot = self._get_rotation_matrix(bend_axis, angle_per_section)
            current_dir = rot @ current_dir
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def _generate_rectangular_elbow(self, num_arc_divisions: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate mesh for rectangular elbow."""
        all_vertices = []
        all_indices = []
        all_normals = []
        
        p = self.params
        center = np.array(p.center)
        inlet_dir = np.array(p.inlet_direction_normalized)
        bend_axis = np.array(p.bend_axis_normalized)
        
        hw = p.width / 2
        hh = p.height / 2
        t = p.wall_thickness
        R = p.bend_radius
        
        # Calculate arc center
        perp = np.cross(inlet_dir, bend_axis)
        perp = perp / np.linalg.norm(perp)
        arc_center = center + R * perp
        
        def get_rect_corners(half_w: float, half_h: float, duct_center: np.ndarray,
                            radial: np.ndarray, local_y: np.ndarray) -> List[np.ndarray]:
            """Get 4 corners of rectangular cross-section."""
            return [
                duct_center + half_w * radial - half_h * local_y,
                duct_center + half_w * radial + half_h * local_y,
                duct_center - half_w * radial + half_h * local_y,
                duct_center - half_w * radial - half_h * local_y,
            ]
        
        # Generate outer and inner surfaces along arc
        for surface_type in ['outer', 'inner']:
            if surface_type == 'outer':
                half_w, half_h = hw + t, hh + t
                outward = True
            else:
                half_w, half_h = hw, hh
                outward = False
            
            # Store corners for each arc position
            corner_rings = []
            
            for i in range(num_arc_divisions + 1):
                arc_angle = p.bend_angle * i / num_arc_divisions
                rot = self._get_rotation_matrix(bend_axis, arc_angle)
                
                radial = -rot @ perp
                duct_center = arc_center + R * (-radial)
                tangent = rot @ inlet_dir
                local_y = np.cross(tangent, radial)
                local_y = local_y / np.linalg.norm(local_y)
                
                corners = get_rect_corners(half_w, half_h, duct_center, radial, local_y)
                corner_rings.append((corners, radial, local_y, tangent))
            
            # Create faces
            for i in range(num_arc_divisions):
                corners1, rad1, ly1, tan1 = corner_rings[i]
                corners2, rad2, ly2, tan2 = corner_rings[i + 1]
                
                # 4 faces connecting the rings
                face_normals = [rad1, ly1, -rad1, -ly1] if outward else [-rad1, -ly1, rad1, ly1]
                
                for f in range(4):
                    base_idx = len(all_vertices)
                    
                    c1_a = corners1[f]
                    c1_b = corners1[(f + 1) % 4]
                    c2_a = corners2[f]
                    c2_b = corners2[(f + 1) % 4]
                    
                    norm = list(face_normals[f])
                    
                    all_vertices.append(list(c1_a))
                    all_normals.append(norm)
                    all_vertices.append(list(c1_b))
                    all_normals.append(norm)
                    all_vertices.append(list(c2_a))
                    all_normals.append(norm)
                    all_vertices.append(list(c2_b))
                    all_normals.append(norm)
                    
                    if outward:
                        all_indices.extend([base_idx, base_idx + 2, base_idx + 1])
                        all_indices.extend([base_idx + 1, base_idx + 2, base_idx + 3])
                    else:
                        all_indices.extend([base_idx, base_idx + 1, base_idx + 2])
                        all_indices.extend([base_idx + 1, base_idx + 3, base_idx + 2])
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def get_inlet_position(self) -> Tuple[float, float, float]:
        """Get center position of inlet."""
        return self.params.center
    
    def get_outlet_position(self) -> Tuple[float, float, float]:
        """Get center position of outlet."""
        p = self.params
        center = np.array(p.center)
        inlet_dir = np.array(p.inlet_direction_normalized)
        bend_axis = np.array(p.bend_axis_normalized)
        R = p.bend_radius
        
        # Calculate arc center
        perp = np.cross(inlet_dir, bend_axis)
        perp = perp / np.linalg.norm(perp)
        arc_center = center + R * perp
        
        # Rotate to outlet position
        rot = self._get_rotation_matrix(bend_axis, p.bend_angle)
        radial = -rot @ perp
        outlet_center = arc_center + R * (-radial)
        
        return tuple(outlet_center)
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box (min, max) corners."""
        verts = self.vertices
        return verts.min(axis=0), verts.max(axis=0)
    
    def to_warp_mesh(self) -> Any:
        """Create a Warp mesh object from the geometry."""
        if wp is None:
            raise ImportError("NVIDIA Warp is required for mesh creation")
        
        return wp.Mesh(
            points=wp.array(self.vertices, dtype=wp.vec3),
            indices=wp.array(self.indices.flatten(), dtype=wp.int32)
        )


# Factory functions

def create_90_degree_elbow(diameter: float = 0.2,
                           bend_radius: float = None,
                           **kwargs) -> Elbow:
    """
    Create a standard 90-degree round elbow.
    
    Args:
        diameter: Duct diameter [m]
        bend_radius: Centerline bend radius [m] (default: 1.5 * diameter)
        **kwargs: Additional parameters
        
    Returns:
        Elbow instance
    """
    if bend_radius is None:
        bend_radius = 1.5 * diameter  # Standard R/D = 1.5
    
    params = ElbowParams(
        elbow_type="round",
        diameter=diameter,
        bend_radius=bend_radius,
        bend_angle=np.pi / 2,
        **kwargs
    )
    return Elbow(params)


def create_45_degree_elbow(diameter: float = 0.2,
                           bend_radius: float = None,
                           **kwargs) -> Elbow:
    """
    Create a standard 45-degree round elbow.
    
    Args:
        diameter: Duct diameter [m]
        bend_radius: Centerline bend radius [m]
        **kwargs: Additional parameters
        
    Returns:
        Elbow instance
    """
    if bend_radius is None:
        bend_radius = 1.5 * diameter
    
    params = ElbowParams(
        elbow_type="round",
        diameter=diameter,
        bend_radius=bend_radius,
        bend_angle=np.pi / 4,
        **kwargs
    )
    return Elbow(params)


def create_mitered_elbow(diameter: float = 0.2,
                         num_gores: int = 5,
                         bend_angle: float = np.pi / 2,
                         **kwargs) -> Elbow:
    """
    Create a mitered (segmented) elbow.
    
    Args:
        diameter: Duct diameter [m]
        num_gores: Number of gore sections
        bend_angle: Total bend angle [rad]
        **kwargs: Additional parameters
        
    Returns:
        Elbow instance
    """
    # Mitered elbows have effective radius based on geometry
    bend_radius = diameter * 0.5 / np.sin(bend_angle / (2 * num_gores))
    
    params = ElbowParams(
        elbow_type="mitered",
        diameter=diameter,
        bend_radius=bend_radius,
        bend_angle=bend_angle,
        num_gores=num_gores,
        **kwargs
    )
    return Elbow(params)


def create_elbow_with_vanes(diameter: float = 0.2,
                            bend_radius: float = None,
                            num_vanes: int = 3,
                            **kwargs) -> Elbow:
    """
    Create an elbow with turning vanes for reduced pressure loss.
    
    Args:
        diameter: Duct diameter [m]
        bend_radius: Centerline bend radius [m]
        num_vanes: Number of turning vanes
        **kwargs: Additional parameters
        
    Returns:
        Elbow instance
    """
    if bend_radius is None:
        bend_radius = diameter  # Tighter radius OK with vanes
    
    params = ElbowParams(
        elbow_type="round",
        diameter=diameter,
        bend_radius=bend_radius,
        bend_angle=np.pi / 2,
        turning_vanes=True,
        num_vanes=num_vanes,
        **kwargs
    )
    return Elbow(params)
