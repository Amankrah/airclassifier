"""
Sight glass / inspection port components for air classification systems.

This module provides sight glass and inspection window geometries
for visual process monitoring.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class SightGlassParams:
    """
    Parameters for sight glass / inspection window.
    
    Attributes:
        glass_diameter: View diameter [m]
        glass_type: Glass material ("borosilicate", "tempered")
        flange_size: Standard flange size description
        light_port: Whether to include illumination port
        wiper: Whether to include internal wiper
        location: Position (x, y, z) [m]
        surface_normal: Normal direction of mounting surface
        glass_thickness: Glass thickness [m]
        flange_thickness: Mounting flange thickness [m]
        body_depth: Body depth into vessel [m]
    """
    glass_diameter: float = 0.100
    glass_type: str = "borosilicate"
    flange_size: str = "DN100"
    light_port: bool = False
    wiper: bool = False
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    surface_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    glass_thickness: float = 0.010
    flange_thickness: float = 0.020
    body_depth: float = 0.025
    
    @property
    def glass_radius(self) -> float:
        """Glass radius [m]."""
        return self.glass_diameter / 2
    
    @property
    def normal_normalized(self) -> Tuple[float, float, float]:
        """Normalized surface normal."""
        n = np.array(self.surface_normal)
        return tuple(n / np.linalg.norm(n))
    
    @property
    def flange_diameter(self) -> float:
        """Flange outer diameter [m]."""
        # Standard DIN flange sizes (approximate)
        flange_sizes = {
            "DN50": 0.120,
            "DN80": 0.160,
            "DN100": 0.190,
            "DN125": 0.220,
            "DN150": 0.250,
        }
        return flange_sizes.get(self.flange_size, self.glass_diameter * 1.8)


class SightGlass:
    """
    Sight glass / inspection window geometry.
    
    Generates mesh for visual inspection ports.
    """
    
    def __init__(self, params: SightGlassParams):
        """
        Initialize sight glass.
        
        Args:
            params: Sight glass parameters
        """
        self.params = params
        self._vertices: Optional[np.ndarray] = None
        self._indices: Optional[np.ndarray] = None
        self._normals: Optional[np.ndarray] = None
    
    @property
    def vertices(self) -> np.ndarray:
        """Get mesh vertices."""
        if self._vertices is None:
            self.generate_mesh()
        return self._vertices
    
    @property
    def indices(self) -> np.ndarray:
        """Get mesh indices."""
        if self._indices is None:
            self.generate_mesh()
        return self._indices
    
    @property
    def normals(self) -> np.ndarray:
        """Get mesh normals."""
        if self._normals is None:
            self.generate_mesh()
        return self._normals
    
    def generate_mesh(self, num_segments: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triangular mesh for the sight glass.
        
        Args:
            num_segments: Circumferential segments
            
        Returns:
            Tuple of (vertices, indices, normals)
        """
        all_vertices = []
        all_indices = []
        all_normals = []
        
        p = self.params
        center = np.array(p.location)
        normal = np.array(p.normal_normalized)
        
        # Create local coordinate system
        if abs(normal[2]) < 0.9:
            perp1 = np.cross(normal, [0, 0, 1])
        else:
            perp1 = np.cross(normal, [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(normal, perp1)
        
        # Mounting flange
        self._add_flange(all_vertices, all_indices, all_normals,
                        center, normal, perp1, perp2, num_segments)
        
        # Glass pane
        self._add_glass(all_vertices, all_indices, all_normals,
                       center, normal, perp1, perp2, num_segments)
        
        # Retaining ring / bezel
        self._add_bezel(all_vertices, all_indices, all_normals,
                       center, normal, perp1, perp2, num_segments)
        
        # Light port if specified
        if p.light_port:
            self._add_light_port(all_vertices, all_indices, all_normals,
                                center, normal, perp1, perp2, num_segments)
        
        # Wiper mechanism if specified
        if p.wiper:
            self._add_wiper(all_vertices, all_indices, all_normals,
                           center, normal, perp1, perp2, num_segments)
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def _add_flange(self, all_vertices, all_indices, all_normals,
                    center, normal, perp1, perp2, num_segments):
        """Add mounting flange."""
        p = self.params
        flange_r = p.flange_diameter / 2
        inner_r = p.glass_radius + 0.005
        
        base_idx = len(all_vertices)
        
        # Flange disc (annular)
        for t in [0, p.flange_thickness]:
            for radius in [inner_r, flange_r]:
                for i in range(num_segments):
                    theta = 2 * np.pi * i / num_segments
                    pt = center + radius * np.cos(theta) * perp1 + radius * np.sin(theta) * perp2 + t * normal
                    all_vertices.append(list(pt))
                    all_normals.append(list(normal if t > 0 else -normal))
        
        # Front face (inner to outer)
        for i in range(num_segments):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % num_segments
            i2 = base_idx + num_segments + i
            i3 = base_idx + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
        
        # Back face
        back_base = base_idx + 2 * num_segments
        for i in range(num_segments):
            i0 = back_base + i
            i1 = back_base + (i + 1) % num_segments
            i2 = back_base + num_segments + i
            i3 = back_base + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i1, i2])
            all_indices.extend([i1, i3, i2])
        
        # Outer edge
        edge_base = len(all_vertices)
        for t in [0, p.flange_thickness]:
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                pt = center + flange_r * np.cos(theta) * perp1 + flange_r * np.sin(theta) * perp2 + t * normal
                all_vertices.append(list(pt))
                n = np.cos(theta) * perp1 + np.sin(theta) * perp2
                all_normals.append(list(n))
        
        for i in range(num_segments):
            i0 = edge_base + i
            i1 = edge_base + (i + 1) % num_segments
            i2 = edge_base + num_segments + i
            i3 = edge_base + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
    
    def _add_glass(self, all_vertices, all_indices, all_normals,
                   center, normal, perp1, perp2, num_segments):
        """Add glass pane."""
        p = self.params
        glass_r = p.glass_radius
        glass_z = p.flange_thickness + 0.002  # Slightly recessed
        
        base_idx = len(all_vertices)
        
        # Glass disc (front and back faces)
        for face_offset, face_normal_dir in [(0, 1), (p.glass_thickness, 1)]:
            face_center = center + (glass_z + face_offset) * normal
            face_base = len(all_vertices)
            
            all_vertices.append(list(face_center))
            all_normals.append(list(face_normal_dir * normal))
            
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                pt = face_center + glass_r * np.cos(theta) * perp1 + glass_r * np.sin(theta) * perp2
                all_vertices.append(list(pt))
                all_normals.append(list(face_normal_dir * normal))
            
            if face_offset > 0:
                for i in range(num_segments):
                    all_indices.extend([face_base, face_base + 1 + i, face_base + 1 + (i + 1) % num_segments])
            else:
                for i in range(num_segments):
                    all_indices.extend([face_base, face_base + 1 + (i + 1) % num_segments, face_base + 1 + i])
    
    def _add_bezel(self, all_vertices, all_indices, all_normals,
                   center, normal, perp1, perp2, num_segments):
        """Add retaining bezel/ring."""
        p = self.params
        bezel_inner_r = p.glass_radius
        bezel_outer_r = p.glass_radius + 0.015
        bezel_z = p.flange_thickness + p.glass_thickness + 0.002
        bezel_height = 0.008
        
        base_idx = len(all_vertices)
        
        # Bezel cylinder
        for t in [0, bezel_height]:
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                pt = center + bezel_outer_r * np.cos(theta) * perp1 + bezel_outer_r * np.sin(theta) * perp2 + (bezel_z + t) * normal
                all_vertices.append(list(pt))
                n = np.cos(theta) * perp1 + np.sin(theta) * perp2
                all_normals.append(list(n))
        
        for i in range(num_segments):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % num_segments
            i2 = base_idx + num_segments + i
            i3 = base_idx + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
        
        # Bezel top (annular)
        top_base = len(all_vertices)
        bezel_top_z = bezel_z + bezel_height
        
        for radius in [bezel_inner_r, bezel_outer_r]:
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                pt = center + radius * np.cos(theta) * perp1 + radius * np.sin(theta) * perp2 + bezel_top_z * normal
                all_vertices.append(list(pt))
                all_normals.append(list(normal))
        
        for i in range(num_segments):
            i0 = top_base + i
            i1 = top_base + (i + 1) % num_segments
            i2 = top_base + num_segments + i
            i3 = top_base + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i1, i2])
            all_indices.extend([i1, i3, i2])
    
    def _add_light_port(self, all_vertices, all_indices, all_normals,
                        center, normal, perp1, perp2, num_segments):
        """Add illumination light port."""
        p = self.params
        light_offset = p.flange_diameter / 2 + 0.02
        light_r = 0.015
        light_length = 0.030
        
        light_center = center + light_offset * perp1
        
        base_idx = len(all_vertices)
        
        for t in [0, light_length]:
            for i in range(12):
                theta = 2 * np.pi * i / 12
                pt = light_center + light_r * np.cos(theta) * normal + light_r * np.sin(theta) * perp2 + t * perp1
                all_vertices.append(list(pt))
                n = np.cos(theta) * normal + np.sin(theta) * perp2
                all_normals.append(list(n))
        
        for i in range(12):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % 12
            i2 = base_idx + 12 + i
            i3 = base_idx + 12 + (i + 1) % 12
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
    
    def _add_wiper(self, all_vertices, all_indices, all_normals,
                   center, normal, perp1, perp2, num_segments):
        """Add wiper mechanism."""
        p = self.params
        wiper_offset = -p.body_depth + 0.005
        blade_length = p.glass_radius * 0.8
        blade_width = 0.015
        blade_thick = 0.003
        
        wiper_center = center + wiper_offset * normal
        
        base_idx = len(all_vertices)
        
        # Simplified wiper blade
        corners = [
            wiper_center + blade_width/2 * perp2 - blade_thick/2 * normal,
            wiper_center + blade_width/2 * perp2 + blade_thick/2 * normal,
            wiper_center + blade_width/2 * perp2 + blade_thick/2 * normal + blade_length * perp1,
            wiper_center + blade_width/2 * perp2 - blade_thick/2 * normal + blade_length * perp1,
            wiper_center - blade_width/2 * perp2 - blade_thick/2 * normal,
            wiper_center - blade_width/2 * perp2 + blade_thick/2 * normal,
            wiper_center - blade_width/2 * perp2 + blade_thick/2 * normal + blade_length * perp1,
            wiper_center - blade_width/2 * perp2 - blade_thick/2 * normal + blade_length * perp1,
        ]
        
        for corner in corners:
            all_vertices.append(list(corner))
            all_normals.append(list(perp1))
        
        # Box faces
        faces = [
            ([0, 1, 2, 3], perp2),
            ([4, 7, 6, 5], -perp2),
            ([0, 3, 7, 4], -normal),
            ([1, 5, 6, 2], normal),
        ]
        
        for face_indices, _ in faces:
            fi = [base_idx + i for i in face_indices]
            all_indices.extend([fi[0], fi[1], fi[2]])
            all_indices.extend([fi[0], fi[2], fi[3]])
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box."""
        verts = self.vertices
        return verts.min(axis=0), verts.max(axis=0)
    
    def to_warp_mesh(self) -> Any:
        """Create a Warp mesh object."""
        if wp is None:
            raise ImportError("NVIDIA Warp is required")
        return wp.Mesh(
            points=wp.array(self.vertices, dtype=wp.vec3),
            indices=wp.array(self.indices.flatten(), dtype=wp.int32)
        )


# Factory functions

def create_standard_sight_glass(location: Tuple[float, float, float],
                                diameter: float = 0.1,
                                **kwargs) -> SightGlass:
    """Create a standard sight glass."""
    params = SightGlassParams(
        glass_diameter=diameter,
        location=location,
        **kwargs
    )
    return SightGlass(params)


def create_illuminated_sight_glass(location: Tuple[float, float, float],
                                   diameter: float = 0.1,
                                   **kwargs) -> SightGlass:
    """Create a sight glass with illumination port."""
    params = SightGlassParams(
        glass_diameter=diameter,
        location=location,
        light_port=True,
        **kwargs
    )
    return SightGlass(params)
