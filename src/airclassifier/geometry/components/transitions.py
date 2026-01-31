"""
Duct transition components for air classification systems.

This module provides transition geometries for connecting ducts
of different sizes and shapes (round-to-round, round-to-rectangular,
rectangular-to-rectangular).
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class TransitionParams:
    """
    Parameters for duct transitions and reducers.
    
    Attributes:
        transition_type: Type of transition 
            ("round_to_round", "round_to_rect", "rect_to_rect", "rect_to_round")
        inlet_dimensions: Inlet dimensions (diameter,) or (width, height)
        outlet_dimensions: Outlet dimensions (diameter,) or (width, height)
        length: Transition length [m]
        concentric: True for concentric, False for eccentric
        eccentric_offset: Offset for eccentric transitions (x, y) [m]
        wall_thickness: Wall thickness [m]
        max_angle: Maximum expansion/contraction angle [rad]
        center: Center position of transition inlet (x, y, z) [m]
        direction: Direction vector (dx, dy, dz)
    """
    transition_type: str
    inlet_dimensions: Tuple[float, ...]
    outlet_dimensions: Tuple[float, ...]
    length: float
    concentric: bool = True
    eccentric_offset: Tuple[float, float] = (0.0, 0.0)
    wall_thickness: float = 0.002
    max_angle: float = 0.2618  # 15 degrees
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    
    @property
    def is_expansion(self) -> bool:
        """Check if this is an expansion (outlet larger than inlet)."""
        inlet_area = self.inlet_area
        outlet_area = self.outlet_area
        return outlet_area > inlet_area
    
    @property
    def is_contraction(self) -> bool:
        """Check if this is a contraction (outlet smaller than inlet)."""
        return not self.is_expansion
    
    @property
    def inlet_area(self) -> float:
        """Inlet cross-sectional area [m²]."""
        if self.transition_type.startswith("round"):
            d = self.inlet_dimensions[0]
            return np.pi * (d / 2) ** 2
        else:
            w, h = self.inlet_dimensions[:2]
            return w * h
    
    @property
    def outlet_area(self) -> float:
        """Outlet cross-sectional area [m²]."""
        if self.transition_type.endswith("round"):
            d = self.outlet_dimensions[0]
            return np.pi * (d / 2) ** 2
        else:
            w, h = self.outlet_dimensions[:2]
            return w * h
    
    @property
    def area_ratio(self) -> float:
        """Ratio of outlet area to inlet area."""
        return self.outlet_area / self.inlet_area
    
    @property
    def direction_normalized(self) -> Tuple[float, float, float]:
        """Normalized direction vector."""
        d = np.array(self.direction)
        norm = np.linalg.norm(d)
        if norm > 0:
            d = d / norm
        return tuple(d)
    
    @property
    def expansion_angle(self) -> float:
        """
        Calculate the expansion/contraction half-angle [rad].
        
        Returns:
            Angle in radians
        """
        if self.transition_type == "round_to_round":
            r_in = self.inlet_dimensions[0] / 2
            r_out = self.outlet_dimensions[0] / 2
            return np.arctan(abs(r_out - r_in) / self.length)
        else:
            # Use hydraulic diameter approximation
            if self.transition_type.startswith("round"):
                d_in = self.inlet_dimensions[0]
            else:
                w, h = self.inlet_dimensions[:2]
                d_in = 2 * w * h / (w + h)
            
            if self.transition_type.endswith("round"):
                d_out = self.outlet_dimensions[0]
            else:
                w, h = self.outlet_dimensions[:2]
                d_out = 2 * w * h / (w + h)
            
            return np.arctan(abs(d_out - d_in) / (2 * self.length))
    
    def get_pressure_loss_coefficient(self) -> float:
        """
        Estimate pressure loss coefficient K.
        
        For expansions, uses Borda-Carnot formula.
        For contractions, uses standard contraction coefficients.
        
        Returns:
            Loss coefficient K (ΔP = K * 0.5 * ρ * V²)
        """
        AR = self.area_ratio
        
        if self.is_expansion:
            # Borda-Carnot for sudden expansion, modified for gradual
            angle_deg = np.degrees(self.expansion_angle)
            if angle_deg <= 15:
                # Well-designed gradual expansion
                K = 0.1 * (1 - 1/AR) ** 2
            else:
                # Approaching sudden expansion
                K = (1 - 1/AR) ** 2
        else:
            # Contraction
            angle_deg = np.degrees(self.expansion_angle)
            if angle_deg <= 30:
                K = 0.04 * (1 - AR) ** 2
            else:
                K = 0.5 * (1 - AR)
        
        return K


class Transition:
    """
    Duct transition geometry for air classification systems.
    
    Generates mesh for transitions between different duct sizes/shapes.
    """
    
    def __init__(self, params: TransitionParams):
        """
        Initialize transition.
        
        Args:
            params: Transition parameters
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
    
    def generate_mesh(self, num_segments: int = 32, 
                      num_length_divisions: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triangular mesh for the transition.
        
        Args:
            num_segments: Circumferential segments for round sections
            num_length_divisions: Divisions along length
            
        Returns:
            Tuple of (vertices, indices, normals)
        """
        p = self.params
        
        if p.transition_type == "round_to_round":
            return self._generate_round_to_round(num_segments, num_length_divisions)
        elif p.transition_type == "round_to_rect":
            return self._generate_round_to_rect(num_segments, num_length_divisions)
        elif p.transition_type == "rect_to_round":
            return self._generate_rect_to_round(num_segments, num_length_divisions)
        else:  # rect_to_rect
            return self._generate_rect_to_rect(num_length_divisions)
    
    def _get_coordinate_system(self):
        """Get local coordinate system basis vectors."""
        dx, dy, dz = self.params.direction_normalized
        
        if abs(dz) < 0.9:
            perp1 = np.cross([dx, dy, dz], [0, 0, 1])
        else:
            perp1 = np.cross([dx, dy, dz], [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross([dx, dy, dz], perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        
        return np.array([dx, dy, dz]), perp1, perp2
    
    def _transform_point(self, local_x: float, local_y: float, z_offset: float,
                        direction: np.ndarray, perp1: np.ndarray, perp2: np.ndarray) -> list:
        """Transform local coordinates to world coordinates."""
        cx, cy, cz = self.params.center
        x = cx + local_x * perp1[0] + local_y * perp2[0] + z_offset * direction[0]
        y = cy + local_x * perp1[1] + local_y * perp2[1] + z_offset * direction[1]
        z = cz + local_x * perp1[2] + local_y * perp2[2] + z_offset * direction[2]
        return [x, y, z]
    
    def _generate_round_to_round(self, num_segments: int, 
                                  num_divisions: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate mesh for round-to-round transition (cone frustum)."""
        all_vertices = []
        all_indices = []
        all_normals = []
        
        p = self.params
        direction, perp1, perp2 = self._get_coordinate_system()
        
        r_in = p.inlet_dimensions[0] / 2
        r_out = p.outlet_dimensions[0] / 2
        t = p.wall_thickness
        
        # Calculate eccentric offset at outlet
        if p.concentric:
            offset_x, offset_y = 0.0, 0.0
        else:
            offset_x, offset_y = p.eccentric_offset
        
        def add_conical_surface(r_start: float, r_end: float, 
                               offset_start: Tuple[float, float],
                               offset_end: Tuple[float, float],
                               outward: bool = True):
            """Add a conical/frustum surface."""
            nonlocal all_vertices, all_indices, all_normals
            
            # Generate rings along length
            for i in range(num_divisions + 1):
                t_param = i / num_divisions
                z_pos = t_param * p.length
                radius = r_start + t_param * (r_end - r_start)
                ox = offset_start[0] + t_param * (offset_end[0] - offset_start[0])
                oy = offset_start[1] + t_param * (offset_end[1] - offset_start[1])
                
                for j in range(num_segments):
                    theta = 2 * np.pi * j / num_segments
                    local_x = ox + radius * np.cos(theta)
                    local_y = oy + radius * np.sin(theta)
                    
                    pt = self._transform_point(local_x, local_y, z_pos, 
                                               direction, perp1, perp2)
                    all_vertices.append(pt)
                    
                    # Normal (approximate for cone)
                    slope = (r_end - r_start) / p.length
                    nx = np.cos(theta) * perp1[0] + np.sin(theta) * perp2[0]
                    ny = np.cos(theta) * perp1[1] + np.sin(theta) * perp2[1]
                    nz = np.cos(theta) * perp1[2] + np.sin(theta) * perp2[2]
                    
                    # Adjust for slope
                    factor = 1.0 / np.sqrt(1 + slope ** 2)
                    ax_component = -slope * factor
                    radial = factor
                    
                    norm = [
                        radial * nx + ax_component * direction[0],
                        radial * ny + ax_component * direction[1],
                        radial * nz + ax_component * direction[2]
                    ]
                    if not outward:
                        norm = [-n for n in norm]
                    all_normals.append(norm)
            
            # Generate triangles
            base_idx = len(all_vertices) - (num_divisions + 1) * num_segments
            for i in range(num_divisions):
                for j in range(num_segments):
                    i0 = base_idx + i * num_segments + j
                    i1 = base_idx + i * num_segments + (j + 1) % num_segments
                    i2 = base_idx + (i + 1) * num_segments + j
                    i3 = base_idx + (i + 1) * num_segments + (j + 1) % num_segments
                    
                    if outward:
                        all_indices.extend([i0, i2, i1])
                        all_indices.extend([i1, i2, i3])
                    else:
                        all_indices.extend([i0, i1, i2])
                        all_indices.extend([i1, i3, i2])
        
        # Outer surface
        add_conical_surface(r_in + t, r_out + t, (0, 0), (offset_x, offset_y), True)
        
        # Inner surface
        add_conical_surface(r_in, r_out, (0, 0), (offset_x, offset_y), False)
        
        # End caps (annular rings)
        def add_annular_cap(r_inner: float, r_outer: float, z_off: float,
                           offset: Tuple[float, float], normal_positive: bool):
            nonlocal all_vertices, all_indices, all_normals
            base_idx = len(all_vertices)
            
            norm = list(direction) if normal_positive else [-d for d in direction]
            
            for radius in [r_inner, r_outer]:
                for j in range(num_segments):
                    theta = 2 * np.pi * j / num_segments
                    local_x = offset[0] + radius * np.cos(theta)
                    local_y = offset[1] + radius * np.sin(theta)
                    pt = self._transform_point(local_x, local_y, z_off,
                                               direction, perp1, perp2)
                    all_vertices.append(pt)
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
        
        # Inlet cap
        add_annular_cap(r_in, r_in + t, 0.0, (0, 0), False)
        # Outlet cap
        add_annular_cap(r_out, r_out + t, p.length, (offset_x, offset_y), True)
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def _generate_round_to_rect(self, num_segments: int,
                                 num_divisions: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate mesh for round-to-rectangular transition."""
        all_vertices = []
        all_indices = []
        all_normals = []
        
        p = self.params
        direction, perp1, perp2 = self._get_coordinate_system()
        
        r_in = p.inlet_dimensions[0] / 2
        w_out = p.outlet_dimensions[0] / 2
        h_out = p.outlet_dimensions[1] / 2
        t = p.wall_thickness
        
        # Interpolate between circle and rectangle
        for layer in range(num_divisions + 1):
            t_param = layer / num_divisions
            z_pos = t_param * p.length
            
            base_idx = len(all_vertices)
            
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                
                # Circle point
                cx = r_in * np.cos(theta)
                cy = r_in * np.sin(theta)
                
                # Rectangle point (superellipse for smooth corners)
                n = 2 + 6 * t_param  # Goes from circle (n=2) toward rectangle
                rx = w_out * np.sign(np.cos(theta)) * abs(np.cos(theta)) ** (2/n)
                ry = h_out * np.sign(np.sin(theta)) * abs(np.sin(theta)) ** (2/n)
                
                # Interpolate
                local_x = cx + t_param * (rx - cx)
                local_y = cy + t_param * (ry - cy)
                
                # Outer surface
                scale = 1 + t / max(abs(local_x), abs(local_y), r_in)
                outer_x = local_x * scale
                outer_y = local_y * scale
                
                pt_outer = self._transform_point(outer_x, outer_y, z_pos,
                                                  direction, perp1, perp2)
                all_vertices.append(pt_outer)
                
                # Approximate normal
                nx = np.cos(theta) * perp1[0] + np.sin(theta) * perp2[0]
                ny = np.cos(theta) * perp1[1] + np.sin(theta) * perp2[1]
                nz = np.cos(theta) * perp1[2] + np.sin(theta) * perp2[2]
                all_normals.append([nx, ny, nz])
            
            # Connect to previous layer
            if layer > 0:
                prev_base = base_idx - num_segments
                for i in range(num_segments):
                    i0 = prev_base + i
                    i1 = prev_base + (i + 1) % num_segments
                    i2 = base_idx + i
                    i3 = base_idx + (i + 1) % num_segments
                    
                    all_indices.extend([i0, i2, i1])
                    all_indices.extend([i1, i2, i3])
        
        # Inner surface (similar but smaller)
        outer_vertex_count = len(all_vertices)
        
        for layer in range(num_divisions + 1):
            t_param = layer / num_divisions
            z_pos = t_param * p.length
            
            base_idx = len(all_vertices)
            
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                
                cx = r_in * np.cos(theta)
                cy = r_in * np.sin(theta)
                
                n = 2 + 6 * t_param
                rx = w_out * np.sign(np.cos(theta)) * abs(np.cos(theta)) ** (2/n)
                ry = h_out * np.sign(np.sin(theta)) * abs(np.sin(theta)) ** (2/n)
                
                local_x = cx + t_param * (rx - cx)
                local_y = cy + t_param * (ry - cy)
                
                pt_inner = self._transform_point(local_x, local_y, z_pos,
                                                  direction, perp1, perp2)
                all_vertices.append(pt_inner)
                
                nx = np.cos(theta) * perp1[0] + np.sin(theta) * perp2[0]
                ny = np.cos(theta) * perp1[1] + np.sin(theta) * perp2[1]
                nz = np.cos(theta) * perp1[2] + np.sin(theta) * perp2[2]
                all_normals.append([-nx, -ny, -nz])  # Inward normal
            
            if layer > 0:
                prev_base = base_idx - num_segments
                for i in range(num_segments):
                    i0 = prev_base + i
                    i1 = prev_base + (i + 1) % num_segments
                    i2 = base_idx + i
                    i3 = base_idx + (i + 1) % num_segments
                    
                    all_indices.extend([i0, i1, i2])
                    all_indices.extend([i1, i3, i2])
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def _generate_rect_to_round(self, num_segments: int,
                                 num_divisions: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate mesh for rectangular-to-round transition."""
        # Similar to round_to_rect but reversed
        all_vertices = []
        all_indices = []
        all_normals = []
        
        p = self.params
        direction, perp1, perp2 = self._get_coordinate_system()
        
        w_in = p.inlet_dimensions[0] / 2
        h_in = p.inlet_dimensions[1] / 2
        r_out = p.outlet_dimensions[0] / 2
        t = p.wall_thickness
        
        for layer in range(num_divisions + 1):
            t_param = layer / num_divisions
            z_pos = t_param * p.length
            
            base_idx = len(all_vertices)
            
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                
                # Rectangle point
                n = 8 - 6 * t_param  # Goes from rectangle toward circle
                n = max(n, 2.0)
                rx = w_in * np.sign(np.cos(theta)) * abs(np.cos(theta)) ** (2/n)
                ry = h_in * np.sign(np.sin(theta)) * abs(np.sin(theta)) ** (2/n)
                
                # Circle point
                cx = r_out * np.cos(theta)
                cy = r_out * np.sin(theta)
                
                # Interpolate
                local_x = rx + t_param * (cx - rx)
                local_y = ry + t_param * (cy - ry)
                
                # Outer surface
                dist = np.sqrt(local_x**2 + local_y**2)
                if dist > 0:
                    outer_x = local_x * (1 + t / dist)
                    outer_y = local_y * (1 + t / dist)
                else:
                    outer_x, outer_y = local_x, local_y
                
                pt = self._transform_point(outer_x, outer_y, z_pos,
                                           direction, perp1, perp2)
                all_vertices.append(pt)
                
                nx = np.cos(theta) * perp1[0] + np.sin(theta) * perp2[0]
                ny = np.cos(theta) * perp1[1] + np.sin(theta) * perp2[1]
                nz = np.cos(theta) * perp1[2] + np.sin(theta) * perp2[2]
                all_normals.append([nx, ny, nz])
            
            if layer > 0:
                prev_base = base_idx - num_segments
                for i in range(num_segments):
                    i0 = prev_base + i
                    i1 = prev_base + (i + 1) % num_segments
                    i2 = base_idx + i
                    i3 = base_idx + (i + 1) % num_segments
                    
                    all_indices.extend([i0, i2, i1])
                    all_indices.extend([i1, i2, i3])
        
        # Inner surface
        for layer in range(num_divisions + 1):
            t_param = layer / num_divisions
            z_pos = t_param * p.length
            
            base_idx = len(all_vertices)
            
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                
                n = 8 - 6 * t_param
                n = max(n, 2.0)
                rx = w_in * np.sign(np.cos(theta)) * abs(np.cos(theta)) ** (2/n)
                ry = h_in * np.sign(np.sin(theta)) * abs(np.sin(theta)) ** (2/n)
                
                cx = r_out * np.cos(theta)
                cy = r_out * np.sin(theta)
                
                local_x = rx + t_param * (cx - rx)
                local_y = ry + t_param * (cy - ry)
                
                pt = self._transform_point(local_x, local_y, z_pos,
                                           direction, perp1, perp2)
                all_vertices.append(pt)
                
                nx = np.cos(theta) * perp1[0] + np.sin(theta) * perp2[0]
                ny = np.cos(theta) * perp1[1] + np.sin(theta) * perp2[1]
                nz = np.cos(theta) * perp1[2] + np.sin(theta) * perp2[2]
                all_normals.append([-nx, -ny, -nz])
            
            if layer > 0:
                prev_base = base_idx - num_segments
                for i in range(num_segments):
                    i0 = prev_base + i
                    i1 = prev_base + (i + 1) % num_segments
                    i2 = base_idx + i
                    i3 = base_idx + (i + 1) % num_segments
                    
                    all_indices.extend([i0, i1, i2])
                    all_indices.extend([i1, i3, i2])
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def _generate_rect_to_rect(self, num_divisions: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate mesh for rectangular-to-rectangular transition."""
        all_vertices = []
        all_indices = []
        all_normals = []
        
        p = self.params
        direction, perp1, perp2 = self._get_coordinate_system()
        
        w_in = p.inlet_dimensions[0] / 2
        h_in = p.inlet_dimensions[1] / 2
        w_out = p.outlet_dimensions[0] / 2
        h_out = p.outlet_dimensions[1] / 2
        t = p.wall_thickness
        
        # Eccentric offset
        if p.concentric:
            ox, oy = 0.0, 0.0
        else:
            ox, oy = p.eccentric_offset
        
        def add_quad_strip(corners_start: list, corners_end: list, normal_outward: bool):
            """Add a quad strip between two sets of 4 corners."""
            nonlocal all_vertices, all_indices, all_normals
            base_idx = len(all_vertices)
            
            # Calculate face normal from first quad
            v0 = np.array(corners_start[0])
            v1 = np.array(corners_start[1])
            v2 = np.array(corners_end[0])
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal = normal / norm_len
            if not normal_outward:
                normal = -normal
            normal = list(normal)
            
            for pt in corners_start:
                all_vertices.append(pt)
                all_normals.append(normal)
            for pt in corners_end:
                all_vertices.append(pt)
                all_normals.append(normal)
            
            # Two triangles for the quad
            if normal_outward:
                all_indices.extend([base_idx, base_idx + 1, base_idx + 4])
                all_indices.extend([base_idx + 1, base_idx + 5, base_idx + 4])
                all_indices.extend([base_idx + 1, base_idx + 2, base_idx + 5])
                all_indices.extend([base_idx + 2, base_idx + 6, base_idx + 5])
                all_indices.extend([base_idx + 2, base_idx + 3, base_idx + 6])
                all_indices.extend([base_idx + 3, base_idx + 7, base_idx + 6])
                all_indices.extend([base_idx + 3, base_idx + 0, base_idx + 7])
                all_indices.extend([base_idx + 0, base_idx + 4, base_idx + 7])
            else:
                all_indices.extend([base_idx, base_idx + 4, base_idx + 1])
                all_indices.extend([base_idx + 1, base_idx + 4, base_idx + 5])
                all_indices.extend([base_idx + 1, base_idx + 5, base_idx + 2])
                all_indices.extend([base_idx + 2, base_idx + 5, base_idx + 6])
                all_indices.extend([base_idx + 2, base_idx + 6, base_idx + 3])
                all_indices.extend([base_idx + 3, base_idx + 6, base_idx + 7])
                all_indices.extend([base_idx + 3, base_idx + 7, base_idx + 0])
                all_indices.extend([base_idx + 0, base_idx + 7, base_idx + 4])
        
        # Outer corners at inlet and outlet
        outer_in = [
            self._transform_point(-w_in - t, -h_in - t, 0, direction, perp1, perp2),
            self._transform_point(w_in + t, -h_in - t, 0, direction, perp1, perp2),
            self._transform_point(w_in + t, h_in + t, 0, direction, perp1, perp2),
            self._transform_point(-w_in - t, h_in + t, 0, direction, perp1, perp2),
        ]
        outer_out = [
            self._transform_point(ox - w_out - t, oy - h_out - t, p.length, direction, perp1, perp2),
            self._transform_point(ox + w_out + t, oy - h_out - t, p.length, direction, perp1, perp2),
            self._transform_point(ox + w_out + t, oy + h_out + t, p.length, direction, perp1, perp2),
            self._transform_point(ox - w_out - t, oy + h_out + t, p.length, direction, perp1, perp2),
        ]
        
        # Inner corners
        inner_in = [
            self._transform_point(-w_in, -h_in, 0, direction, perp1, perp2),
            self._transform_point(w_in, -h_in, 0, direction, perp1, perp2),
            self._transform_point(w_in, h_in, 0, direction, perp1, perp2),
            self._transform_point(-w_in, h_in, 0, direction, perp1, perp2),
        ]
        inner_out = [
            self._transform_point(ox - w_out, oy - h_out, p.length, direction, perp1, perp2),
            self._transform_point(ox + w_out, oy - h_out, p.length, direction, perp1, perp2),
            self._transform_point(ox + w_out, oy + h_out, p.length, direction, perp1, perp2),
            self._transform_point(ox - w_out, oy + h_out, p.length, direction, perp1, perp2),
        ]
        
        # Outer walls (4 faces)
        add_quad_strip(outer_in, outer_out, True)
        
        # Inner walls (4 faces, reversed normals)
        add_quad_strip(inner_in, inner_out, False)
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def get_inlet_position(self) -> Tuple[float, float, float]:
        """Get center position of inlet."""
        return self.params.center
    
    def get_outlet_position(self) -> Tuple[float, float, float]:
        """Get center position of outlet."""
        cx, cy, cz = self.params.center
        dx, dy, dz = self.params.direction_normalized
        L = self.params.length
        return (cx + L * dx, cy + L * dy, cz + L * dz)
    
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

def create_round_reducer(inlet_diameter: float,
                         outlet_diameter: float,
                         length: float = None,
                         **kwargs) -> Transition:
    """
    Create a round-to-round reducer/expander.
    
    Args:
        inlet_diameter: Inlet diameter [m]
        outlet_diameter: Outlet diameter [m]
        length: Transition length [m] (auto-calculated if None)
        **kwargs: Additional parameters
        
    Returns:
        Transition instance
    """
    if length is None:
        # Calculate length for 15° half-angle
        length = abs(outlet_diameter - inlet_diameter) / (2 * np.tan(np.radians(15)))
        length = max(length, 0.1)  # Minimum 100mm
    
    params = TransitionParams(
        transition_type="round_to_round",
        inlet_dimensions=(inlet_diameter,),
        outlet_dimensions=(outlet_diameter,),
        length=length,
        **kwargs
    )
    return Transition(params)


def create_round_to_rect_transition(inlet_diameter: float,
                                    outlet_width: float,
                                    outlet_height: float,
                                    length: float = None,
                                    **kwargs) -> Transition:
    """
    Create a round-to-rectangular transition.
    
    Args:
        inlet_diameter: Round inlet diameter [m]
        outlet_width: Rectangular outlet width [m]
        outlet_height: Rectangular outlet height [m]
        length: Transition length [m]
        **kwargs: Additional parameters
        
    Returns:
        Transition instance
    """
    if length is None:
        # Use diagonal difference for length calculation
        r = inlet_diameter / 2
        diag = np.sqrt(outlet_width**2 + outlet_height**2) / 2
        length = abs(diag - r) / np.tan(np.radians(15))
        length = max(length, 0.15)
    
    params = TransitionParams(
        transition_type="round_to_rect",
        inlet_dimensions=(inlet_diameter,),
        outlet_dimensions=(outlet_width, outlet_height),
        length=length,
        **kwargs
    )
    return Transition(params)


def create_rect_to_round_transition(inlet_width: float,
                                    inlet_height: float,
                                    outlet_diameter: float,
                                    length: float = None,
                                    **kwargs) -> Transition:
    """
    Create a rectangular-to-round transition.
    
    Args:
        inlet_width: Rectangular inlet width [m]
        inlet_height: Rectangular inlet height [m]
        outlet_diameter: Round outlet diameter [m]
        length: Transition length [m]
        **kwargs: Additional parameters
        
    Returns:
        Transition instance
    """
    if length is None:
        r = outlet_diameter / 2
        diag = np.sqrt(inlet_width**2 + inlet_height**2) / 2
        length = abs(diag - r) / np.tan(np.radians(15))
        length = max(length, 0.15)
    
    params = TransitionParams(
        transition_type="rect_to_round",
        inlet_dimensions=(inlet_width, inlet_height),
        outlet_dimensions=(outlet_diameter,),
        length=length,
        **kwargs
    )
    return Transition(params)
