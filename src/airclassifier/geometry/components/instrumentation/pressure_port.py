"""
Pressure port components for air classification systems.

This module provides pressure measurement port geometries
for process instrumentation.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class PressurePortParams:
    """
    Parameters for pressure measurement port.
    
    Attributes:
        port_type: Type ("flush_mount", "extended", "averaging")
        connection_size: NPT connection ("1/4 NPT", "1/2 NPT", etc.)
        port_diameter: Port inner diameter [m]
        location: Position (x, y, z) [m]
        surface_normal: Normal direction of mounting surface
        boss_diameter: Weld boss outer diameter [m]
        boss_height: Weld boss height [m]
        extension_length: Length for extended type [m]
    """
    port_type: str = "flush_mount"
    connection_size: str = "1/2 NPT"
    port_diameter: float = 0.010
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    surface_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    boss_diameter: float = 0.035
    boss_height: float = 0.020
    extension_length: float = 0.050
    
    def __post_init__(self):
        """Set port diameter based on connection size."""
        npt_sizes = {
            "1/8 NPT": 0.006,
            "1/4 NPT": 0.008,
            "3/8 NPT": 0.010,
            "1/2 NPT": 0.013,
            "3/4 NPT": 0.019,
            "1 NPT": 0.025,
        }
        if self.port_diameter == 0.010 and self.connection_size in npt_sizes:
            self.port_diameter = npt_sizes[self.connection_size]
    
    @property
    def port_radius(self) -> float:
        """Port inner radius [m]."""
        return self.port_diameter / 2
    
    @property
    def boss_radius(self) -> float:
        """Boss outer radius [m]."""
        return self.boss_diameter / 2
    
    @property
    def normal_normalized(self) -> Tuple[float, float, float]:
        """Normalized surface normal."""
        n = np.array(self.surface_normal)
        return tuple(n / np.linalg.norm(n))


class PressurePort:
    """
    Pressure measurement port geometry.
    
    Generates mesh for pressure transmitter connection points.
    """
    
    def __init__(self, params: PressurePortParams):
        """
        Initialize pressure port.
        
        Args:
            params: Port parameters
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
    
    def generate_mesh(self, num_segments: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triangular mesh for the pressure port.
        
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
        
        # Weld boss
        self._add_boss(all_vertices, all_indices, all_normals,
                      center, normal, perp1, perp2, num_segments)
        
        # Connection fitting on top of boss
        if p.port_type == "flush_mount":
            self._add_flush_fitting(all_vertices, all_indices, all_normals,
                                    center, normal, perp1, perp2, num_segments)
        elif p.port_type == "extended":
            self._add_extended_fitting(all_vertices, all_indices, all_normals,
                                       center, normal, perp1, perp2, num_segments)
        else:  # averaging
            self._add_averaging_fitting(all_vertices, all_indices, all_normals,
                                        center, normal, perp1, perp2, num_segments)
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def _add_boss(self, all_vertices, all_indices, all_normals,
                  center, normal, perp1, perp2, num_segments):
        """Add weld boss cylinder."""
        p = self.params
        base_idx = len(all_vertices)
        
        # Boss outer surface
        for t in [0, p.boss_height]:
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                local_x = p.boss_radius * np.cos(theta)
                local_y = p.boss_radius * np.sin(theta)
                pt = center + local_x * perp1 + local_y * perp2 + t * normal
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
        
        # Boss top face (annular ring around port hole)
        top_base = len(all_vertices)
        top_center = center + p.boss_height * normal
        
        for radius in [p.port_radius, p.boss_radius]:
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                pt = top_center + radius * np.cos(theta) * perp1 + radius * np.sin(theta) * perp2
                all_vertices.append(list(pt))
                all_normals.append(list(normal))
        
        for i in range(num_segments):
            i0 = top_base + i
            i1 = top_base + (i + 1) % num_segments
            i2 = top_base + num_segments + i
            i3 = top_base + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i1, i2])
            all_indices.extend([i1, i3, i2])
    
    def _add_flush_fitting(self, all_vertices, all_indices, all_normals,
                           center, normal, perp1, perp2, num_segments):
        """Add flush mount fitting (hex head)."""
        p = self.params
        fitting_height = 0.020
        hex_radius = p.boss_radius * 0.9
        fitting_start = center + p.boss_height * normal
        
        base_idx = len(all_vertices)
        
        # Hex profile (6 sides)
        num_hex = 6
        for t in [0, fitting_height]:
            for i in range(num_hex):
                theta = 2 * np.pi * i / num_hex + np.pi/6
                local_x = hex_radius * np.cos(theta)
                local_y = hex_radius * np.sin(theta)
                pt = fitting_start + local_x * perp1 + local_y * perp2 + t * normal
                all_vertices.append(list(pt))
                n = np.cos(theta) * perp1 + np.sin(theta) * perp2
                all_normals.append(list(n))
        
        for i in range(num_hex):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % num_hex
            i2 = base_idx + num_hex + i
            i3 = base_idx + num_hex + (i + 1) % num_hex
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
    
    def _add_extended_fitting(self, all_vertices, all_indices, all_normals,
                              center, normal, perp1, perp2, num_segments):
        """Add extended fitting (tube extending into process)."""
        p = self.params
        tube_radius = p.port_radius * 0.8
        fitting_start = center + p.boss_height * normal
        
        # First add hex head
        self._add_flush_fitting(all_vertices, all_indices, all_normals,
                                center, normal, perp1, perp2, num_segments)
        
        # Extension tube (going into vessel, opposite to normal)
        base_idx = len(all_vertices)
        
        for t in [0, -p.extension_length]:
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                local_x = tube_radius * np.cos(theta)
                local_y = tube_radius * np.sin(theta)
                pt = center + local_x * perp1 + local_y * perp2 + t * normal
                all_vertices.append(list(pt))
                n = np.cos(theta) * perp1 + np.sin(theta) * perp2
                all_normals.append(list(n))
        
        for i in range(num_segments):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % num_segments
            i2 = base_idx + num_segments + i
            i3 = base_idx + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i1, i2])
            all_indices.extend([i1, i3, i2])
    
    def _add_averaging_fitting(self, all_vertices, all_indices, all_normals,
                               center, normal, perp1, perp2, num_segments):
        """Add averaging pitot tube fitting."""
        p = self.params
        tube_radius = p.port_radius * 0.6
        
        # Main body
        self._add_flush_fitting(all_vertices, all_indices, all_normals,
                                center, normal, perp1, perp2, num_segments)
        
        # Averaging tube (diamond profile with multiple holes)
        base_idx = len(all_vertices)
        tube_length = p.extension_length * 1.5
        
        for t in [0, -tube_length]:
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                local_x = tube_radius * np.cos(theta)
                local_y = tube_radius * np.sin(theta)
                pt = center + local_x * perp1 + local_y * perp2 + t * normal
                all_vertices.append(list(pt))
                n = np.cos(theta) * perp1 + np.sin(theta) * perp2
                all_normals.append(list(n))
        
        for i in range(num_segments):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % num_segments
            i2 = base_idx + num_segments + i
            i3 = base_idx + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i1, i2])
            all_indices.extend([i1, i3, i2])
    
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

def create_flush_pressure_port(location: Tuple[float, float, float],
                               connection_size: str = "1/2 NPT",
                               **kwargs) -> PressurePort:
    """Create a flush mount pressure port."""
    params = PressurePortParams(
        port_type="flush_mount",
        connection_size=connection_size,
        location=location,
        **kwargs
    )
    return PressurePort(params)


def create_extended_pressure_port(location: Tuple[float, float, float],
                                  extension_length: float = 0.05,
                                  **kwargs) -> PressurePort:
    """Create an extended pressure port."""
    params = PressurePortParams(
        port_type="extended",
        location=location,
        extension_length=extension_length,
        **kwargs
    )
    return PressurePort(params)


def create_averaging_pressure_port(location: Tuple[float, float, float],
                                   **kwargs) -> PressurePort:
    """Create an averaging pitot pressure port."""
    params = PressurePortParams(
        port_type="averaging",
        location=location,
        **kwargs
    )
    return PressurePort(params)
