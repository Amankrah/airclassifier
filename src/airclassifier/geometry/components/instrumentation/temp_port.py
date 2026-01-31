"""
Temperature port (thermowell) components for air classification systems.

This module provides thermowell geometries for temperature
measurement in process equipment.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class TemperaturePortParams:
    """
    Parameters for temperature measurement port (thermowell).
    
    Attributes:
        thermowell_diameter: Thermowell outer diameter [m]
        immersion_length: Immersion depth into process [m]
        connection_type: Connection type ("threaded", "flanged", "weld")
        element_type: Sensor type ("RTD", "thermocouple")
        bore_diameter: Internal bore diameter [m]
        connection_size: Connection size for threaded ("1/2 NPT", etc.)
        flange_diameter: Flange OD for flanged type [m]
        location: Position (x, y, z) [m]
        surface_normal: Normal direction of mounting surface
    """
    thermowell_diameter: float = 0.016  # ~5/8 inch typical
    immersion_length: float = 0.100
    connection_type: str = "threaded"
    element_type: str = "RTD"
    bore_diameter: float = 0.008
    connection_size: str = "1/2 NPT"
    flange_diameter: float = 0.060
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    surface_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    
    @property
    def thermowell_radius(self) -> float:
        """Thermowell outer radius [m]."""
        return self.thermowell_diameter / 2
    
    @property
    def normal_normalized(self) -> Tuple[float, float, float]:
        """Normalized surface normal."""
        n = np.array(self.surface_normal)
        return tuple(n / np.linalg.norm(n))


class TemperaturePort:
    """
    Temperature port (thermowell) geometry.
    
    Generates mesh for thermowells used for temperature measurement.
    """
    
    def __init__(self, params: TemperaturePortParams):
        """
        Initialize temperature port.
        
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
        Generate triangular mesh for the thermowell.
        
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
        
        # Connection head
        if p.connection_type == "threaded":
            self._add_threaded_head(all_vertices, all_indices, all_normals,
                                    center, normal, perp1, perp2, num_segments)
        elif p.connection_type == "flanged":
            self._add_flanged_head(all_vertices, all_indices, all_normals,
                                   center, normal, perp1, perp2, num_segments)
        else:  # weld
            self._add_weld_head(all_vertices, all_indices, all_normals,
                               center, normal, perp1, perp2, num_segments)
        
        # Thermowell stem (into process)
        self._add_stem(all_vertices, all_indices, all_normals,
                      center, normal, perp1, perp2, num_segments)
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def _add_threaded_head(self, all_vertices, all_indices, all_normals,
                           center, normal, perp1, perp2, num_segments):
        """Add threaded connection head."""
        p = self.params
        head_radius = 0.015  # Hex head radius
        head_height = 0.025
        
        base_idx = len(all_vertices)
        
        # Hex profile
        num_hex = 6
        for t in [0, head_height]:
            for i in range(num_hex):
                theta = 2 * np.pi * i / num_hex
                local_x = head_radius * np.cos(theta)
                local_y = head_radius * np.sin(theta)
                pt = center + local_x * perp1 + local_y * perp2 + t * normal
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
        
        # Top cap
        cap_base = len(all_vertices)
        cap_center = center + head_height * normal
        all_vertices.append(list(cap_center))
        all_normals.append(list(normal))
        
        for i in range(num_hex):
            theta = 2 * np.pi * i / num_hex
            pt = cap_center + head_radius * np.cos(theta) * perp1 + head_radius * np.sin(theta) * perp2
            all_vertices.append(list(pt))
            all_normals.append(list(normal))
        
        for i in range(num_hex):
            all_indices.extend([cap_base, cap_base + 1 + i, cap_base + 1 + (i + 1) % num_hex])
    
    def _add_flanged_head(self, all_vertices, all_indices, all_normals,
                          center, normal, perp1, perp2, num_segments):
        """Add flanged connection head."""
        p = self.params
        flange_r = p.flange_diameter / 2
        flange_thickness = 0.012
        neck_r = p.thermowell_radius * 1.5
        neck_height = 0.025
        
        # Flange disc
        base_idx = len(all_vertices)
        for t in [0, flange_thickness]:
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                pt = center + flange_r * np.cos(theta) * perp1 + flange_r * np.sin(theta) * perp2 + t * normal
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
        
        # Neck
        neck_base = len(all_vertices)
        neck_start = center + flange_thickness * normal
        
        for t in [0, neck_height]:
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                pt = neck_start + neck_r * np.cos(theta) * perp1 + neck_r * np.sin(theta) * perp2 + t * normal
                all_vertices.append(list(pt))
                n = np.cos(theta) * perp1 + np.sin(theta) * perp2
                all_normals.append(list(n))
        
        for i in range(num_segments):
            i0 = neck_base + i
            i1 = neck_base + (i + 1) % num_segments
            i2 = neck_base + num_segments + i
            i3 = neck_base + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
    
    def _add_weld_head(self, all_vertices, all_indices, all_normals,
                       center, normal, perp1, perp2, num_segments):
        """Add weld-in connection head."""
        p = self.params
        boss_r = p.thermowell_radius * 2
        boss_height = 0.015
        
        base_idx = len(all_vertices)
        for t in [0, boss_height]:
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                pt = center + boss_r * np.cos(theta) * perp1 + boss_r * np.sin(theta) * perp2 + t * normal
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
    
    def _add_stem(self, all_vertices, all_indices, all_normals,
                  center, normal, perp1, perp2, num_segments):
        """Add thermowell stem."""
        p = self.params
        stem_r = p.thermowell_radius
        
        base_idx = len(all_vertices)
        
        # Stem cylinder (going into vessel)
        for t in [0, -p.immersion_length]:
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                pt = center + stem_r * np.cos(theta) * perp1 + stem_r * np.sin(theta) * perp2 + t * normal
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
        
        # Rounded tip
        tip_base = len(all_vertices)
        tip_center = center - p.immersion_length * normal
        all_vertices.append(list(tip_center - 0.005 * normal))
        all_normals.append(list(-normal))
        
        for i in range(num_segments):
            theta = 2 * np.pi * i / num_segments
            pt = tip_center + stem_r * np.cos(theta) * perp1 + stem_r * np.sin(theta) * perp2
            all_vertices.append(list(pt))
            all_normals.append(list(-normal))
        
        for i in range(num_segments):
            all_indices.extend([tip_base, tip_base + 1 + (i + 1) % num_segments, tip_base + 1 + i])
    
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

def create_threaded_thermowell(location: Tuple[float, float, float],
                               immersion_length: float = 0.1,
                               **kwargs) -> TemperaturePort:
    """Create a threaded thermowell."""
    params = TemperaturePortParams(
        connection_type="threaded",
        location=location,
        immersion_length=immersion_length,
        **kwargs
    )
    return TemperaturePort(params)


def create_flanged_thermowell(location: Tuple[float, float, float],
                              immersion_length: float = 0.15,
                              flange_diameter: float = 0.06,
                              **kwargs) -> TemperaturePort:
    """Create a flanged thermowell."""
    params = TemperaturePortParams(
        connection_type="flanged",
        location=location,
        immersion_length=immersion_length,
        flange_diameter=flange_diameter,
        **kwargs
    )
    return TemperaturePort(params)


def create_weld_thermowell(location: Tuple[float, float, float],
                           immersion_length: float = 0.1,
                           **kwargs) -> TemperaturePort:
    """Create a weld-in thermowell."""
    params = TemperaturePortParams(
        connection_type="weld",
        location=location,
        immersion_length=immersion_length,
        **kwargs
    )
    return TemperaturePort(params)
