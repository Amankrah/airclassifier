"""
Grounding and bonding point components for air classification systems.

This module provides static grounding/bonding connection geometries
for explosion protection in dust-handling equipment.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class GroundingPointParams:
    """
    Parameters for static grounding/bonding connection.
    
    Attributes:
        location: Position (x, y, z) [m]
        stud_diameter: Grounding stud diameter [m]
        stud_length: Stud projection length [m]
        stud_type: Type ("weld_stud", "threaded")
        resistance_max: Maximum resistance to ground [Ω]
        washer_diameter: Washer outer diameter [m]
        boss_diameter: Weld boss diameter [m]
        boss_height: Weld boss height [m]
        surface_normal: Normal direction of mounting surface
    """
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    stud_diameter: float = 0.010  # M10 typical
    stud_length: float = 0.025
    stud_type: str = "weld_stud"
    resistance_max: float = 1.0  # 1 ohm max
    washer_diameter: float = 0.025
    boss_diameter: float = 0.030
    boss_height: float = 0.005
    surface_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    
    @property
    def stud_radius(self) -> float:
        """Stud radius [m]."""
        return self.stud_diameter / 2
    
    @property
    def normal_normalized(self) -> Tuple[float, float, float]:
        """Normalized surface normal."""
        n = np.array(self.surface_normal)
        return tuple(n / np.linalg.norm(n))


class GroundingPoint:
    """
    Grounding/bonding point geometry.
    
    Generates mesh for static grounding studs used for
    explosion protection bonding.
    """
    
    def __init__(self, params: GroundingPointParams):
        """
        Initialize grounding point.
        
        Args:
            params: Grounding point parameters
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
    
    def generate_mesh(self, num_segments: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triangular mesh for the grounding point.
        
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
        
        # Weld boss (if weld stud)
        if p.stud_type == "weld_stud":
            boss_r = p.boss_diameter / 2
            base_idx = len(all_vertices)
            
            # Boss cylinder
            for t in [0, p.boss_height]:
                for i in range(num_segments):
                    theta = 2 * np.pi * i / num_segments
                    local_x = boss_r * np.cos(theta)
                    local_y = boss_r * np.sin(theta)
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
            
            # Boss top face (annular)
            top_base = len(all_vertices)
            boss_top = center + p.boss_height * normal
            
            for radius in [p.stud_radius, boss_r]:
                for i in range(num_segments):
                    theta = 2 * np.pi * i / num_segments
                    pt = boss_top + radius * np.cos(theta) * perp1 + radius * np.sin(theta) * perp2
                    all_vertices.append(list(pt))
                    all_normals.append(list(normal))
            
            for i in range(num_segments):
                i0 = top_base + i
                i1 = top_base + (i + 1) % num_segments
                i2 = top_base + num_segments + i
                i3 = top_base + num_segments + (i + 1) % num_segments
                all_indices.extend([i0, i1, i2])
                all_indices.extend([i1, i3, i2])
            
            stud_base_offset = p.boss_height
        else:
            stud_base_offset = 0
        
        # Stud cylinder
        stud_base = len(all_vertices)
        stud_start = center + stud_base_offset * normal
        
        for t in [0, p.stud_length]:
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                local_x = p.stud_radius * np.cos(theta)
                local_y = p.stud_radius * np.sin(theta)
                pt = stud_start + local_x * perp1 + local_y * perp2 + t * normal
                all_vertices.append(list(pt))
                n = np.cos(theta) * perp1 + np.sin(theta) * perp2
                all_normals.append(list(n))
        
        for i in range(num_segments):
            i0 = stud_base + i
            i1 = stud_base + (i + 1) % num_segments
            i2 = stud_base + num_segments + i
            i3 = stud_base + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
        
        # Stud top (flat cap)
        cap_base = len(all_vertices)
        cap_center = stud_start + p.stud_length * normal
        all_vertices.append(list(cap_center))
        all_normals.append(list(normal))
        
        for i in range(num_segments):
            theta = 2 * np.pi * i / num_segments
            pt = cap_center + p.stud_radius * np.cos(theta) * perp1 + p.stud_radius * np.sin(theta) * perp2
            all_vertices.append(list(pt))
            all_normals.append(list(normal))
        
        for i in range(num_segments):
            all_indices.extend([cap_base, cap_base + 1 + i, cap_base + 1 + (i + 1) % num_segments])
        
        # Add washer and nut representation
        washer_base = len(all_vertices)
        washer_z = stud_start + (p.stud_length - 0.008) * normal  # Near top
        washer_r = p.washer_diameter / 2
        washer_thick = 0.002
        
        for radius in [p.stud_radius + 0.001, washer_r]:
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                pt = washer_z + radius * np.cos(theta) * perp1 + radius * np.sin(theta) * perp2
                all_vertices.append(list(pt))
                all_normals.append(list(-normal))  # Bottom face
        
        for i in range(num_segments):
            i0 = washer_base + i
            i1 = washer_base + (i + 1) % num_segments
            i2 = washer_base + num_segments + i
            i3 = washer_base + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
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

def create_weld_stud_ground(location: Tuple[float, float, float],
                            surface_normal: Tuple[float, float, float] = (0, 0, 1),
                            stud_size: str = "M10",
                            **kwargs) -> GroundingPoint:
    """
    Create a weld stud grounding point.
    
    Args:
        location: Position (x, y, z) [m]
        surface_normal: Surface normal direction
        stud_size: Metric stud size ("M8", "M10", "M12")
        **kwargs: Additional parameters
        
    Returns:
        GroundingPoint instance
    """
    stud_diameters = {
        "M6": 0.006,
        "M8": 0.008,
        "M10": 0.010,
        "M12": 0.012,
    }
    
    params = GroundingPointParams(
        location=location,
        stud_diameter=stud_diameters.get(stud_size, 0.010),
        stud_type="weld_stud",
        surface_normal=surface_normal,
        **kwargs
    )
    return GroundingPoint(params)


def create_threaded_ground(location: Tuple[float, float, float],
                           surface_normal: Tuple[float, float, float] = (0, 0, 1),
                           **kwargs) -> GroundingPoint:
    """
    Create a threaded grounding point.
    
    Args:
        location: Position (x, y, z) [m]
        surface_normal: Surface normal direction
        **kwargs: Additional parameters
        
    Returns:
        GroundingPoint instance
    """
    params = GroundingPointParams(
        location=location,
        stud_type="threaded",
        surface_normal=surface_normal,
        boss_height=0,  # No boss for threaded
        **kwargs
    )
    return GroundingPoint(params)


def create_grounding_system(locations: List[Tuple[float, float, float]],
                            normals: List[Tuple[float, float, float]] = None,
                            **kwargs) -> List[GroundingPoint]:
    """
    Create multiple grounding points for a system.
    
    Args:
        locations: List of positions
        normals: List of surface normals (optional)
        **kwargs: Additional parameters for all points
        
    Returns:
        List of GroundingPoint instances
    """
    if normals is None:
        normals = [(0, 0, 1)] * len(locations)
    
    points = []
    for loc, norm in zip(locations, normals):
        point = create_weld_stud_ground(loc, norm, **kwargs)
        points.append(point)
    
    return points
