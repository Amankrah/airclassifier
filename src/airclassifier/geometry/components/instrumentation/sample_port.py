"""
Sample port components for air classification systems.

This module provides sample extraction port geometries
for process sampling and quality control.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class SamplePortParams:
    """
    Parameters for in-line sample extraction port.
    
    Attributes:
        port_diameter: Port diameter [m]
        valve_type: Valve type ("ball", "plug", "slide")
        sample_type: Sampling method ("isokinetic", "scoop", "thief")
        location: Position (x, y, z) [m]
        surface_normal: Normal direction of mounting surface
        valve_body_diameter: Valve body OD [m]
        valve_length: Valve body length [m]
        nozzle_length: Sample nozzle length [m]
    """
    port_diameter: float = 0.025
    valve_type: str = "ball"
    sample_type: str = "scoop"
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    surface_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    valve_body_diameter: float = 0.040
    valve_length: float = 0.060
    nozzle_length: float = 0.030
    
    @property
    def port_radius(self) -> float:
        """Port inner radius [m]."""
        return self.port_diameter / 2
    
    @property
    def valve_radius(self) -> float:
        """Valve body radius [m]."""
        return self.valve_body_diameter / 2
    
    @property
    def normal_normalized(self) -> Tuple[float, float, float]:
        """Normalized surface normal."""
        n = np.array(self.surface_normal)
        return tuple(n / np.linalg.norm(n))


class SamplePort:
    """
    Sample extraction port geometry.
    
    Generates mesh for sample ports with various valve types.
    """
    
    def __init__(self, params: SamplePortParams):
        """
        Initialize sample port.
        
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
        Generate triangular mesh for the sample port.
        
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
        
        # Valve body
        self._add_valve_body(all_vertices, all_indices, all_normals,
                            center, normal, perp1, perp2, num_segments)
        
        # Handle
        self._add_handle(all_vertices, all_indices, all_normals,
                        center, normal, perp1, perp2, num_segments)
        
        # Sample nozzle (into process)
        if p.sample_type == "isokinetic":
            self._add_isokinetic_nozzle(all_vertices, all_indices, all_normals,
                                        center, normal, perp1, perp2, num_segments)
        else:
            self._add_simple_nozzle(all_vertices, all_indices, all_normals,
                                    center, normal, perp1, perp2, num_segments)
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def _add_valve_body(self, all_vertices, all_indices, all_normals,
                        center, normal, perp1, perp2, num_segments):
        """Add valve body."""
        p = self.params
        base_idx = len(all_vertices)
        
        # Main valve body cylinder
        for t in [0, p.valve_length]:
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                pt = center + p.valve_radius * np.cos(theta) * perp1 + p.valve_radius * np.sin(theta) * perp2 + t * normal
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
        
        # Top cap with outlet hole
        cap_base = len(all_vertices)
        cap_center = center + p.valve_length * normal
        
        for radius in [p.port_radius, p.valve_radius]:
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                pt = cap_center + radius * np.cos(theta) * perp1 + radius * np.sin(theta) * perp2
                all_vertices.append(list(pt))
                all_normals.append(list(normal))
        
        for i in range(num_segments):
            i0 = cap_base + i
            i1 = cap_base + (i + 1) % num_segments
            i2 = cap_base + num_segments + i
            i3 = cap_base + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i1, i2])
            all_indices.extend([i1, i3, i2])
    
    def _add_handle(self, all_vertices, all_indices, all_normals,
                    center, normal, perp1, perp2, num_segments):
        """Add valve handle."""
        p = self.params
        handle_length = 0.060
        handle_radius = 0.008
        handle_center = center + (p.valve_length / 2) * normal
        
        base_idx = len(all_vertices)
        
        # Handle extends perpendicular to normal
        for t in [0, handle_length]:
            hc = handle_center + t * perp1
            for i in range(8):
                theta = 2 * np.pi * i / 8
                pt = hc + handle_radius * np.cos(theta) * normal + handle_radius * np.sin(theta) * perp2
                all_vertices.append(list(pt))
                n = np.cos(theta) * normal + np.sin(theta) * perp2
                all_normals.append(list(n))
        
        for i in range(8):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % 8
            i2 = base_idx + 8 + i
            i3 = base_idx + 8 + (i + 1) % 8
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
        
        # Handle grip ball
        grip_base = len(all_vertices)
        grip_center = handle_center + handle_length * perp1
        grip_radius = 0.015
        
        all_vertices.append(list(grip_center))
        all_normals.append(list(perp1))
        
        for i in range(num_segments):
            theta = 2 * np.pi * i / num_segments
            pt = grip_center + grip_radius * np.cos(theta) * normal + grip_radius * np.sin(theta) * perp2
            all_vertices.append(list(pt))
            n = np.cos(theta) * normal + np.sin(theta) * perp2
            all_normals.append(list(n))
        
        for i in range(num_segments):
            all_indices.extend([grip_base, grip_base + 1 + i, grip_base + 1 + (i + 1) % num_segments])
    
    def _add_simple_nozzle(self, all_vertices, all_indices, all_normals,
                           center, normal, perp1, perp2, num_segments):
        """Add simple sample nozzle."""
        p = self.params
        nozzle_r = p.port_radius * 0.8
        
        base_idx = len(all_vertices)
        
        for t in [0, -p.nozzle_length]:
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                pt = center + nozzle_r * np.cos(theta) * perp1 + nozzle_r * np.sin(theta) * perp2 + t * normal
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
    
    def _add_isokinetic_nozzle(self, all_vertices, all_indices, all_normals,
                               center, normal, perp1, perp2, num_segments):
        """Add isokinetic sample nozzle (with tapered inlet)."""
        p = self.params
        nozzle_r = p.port_radius * 0.8
        inlet_r = p.port_radius * 1.2  # Flared inlet
        
        base_idx = len(all_vertices)
        
        # Tapered section
        for t, radius in [(0, nozzle_r), (-p.nozzle_length * 0.3, nozzle_r),
                          (-p.nozzle_length * 0.4, inlet_r), (-p.nozzle_length, inlet_r)]:
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                pt = center + radius * np.cos(theta) * perp1 + radius * np.sin(theta) * perp2 + t * normal
                all_vertices.append(list(pt))
                n = np.cos(theta) * perp1 + np.sin(theta) * perp2
                all_normals.append(list(n))
        
        for ring in range(3):
            for i in range(num_segments):
                i0 = base_idx + ring * num_segments + i
                i1 = base_idx + ring * num_segments + (i + 1) % num_segments
                i2 = base_idx + (ring + 1) * num_segments + i
                i3 = base_idx + (ring + 1) * num_segments + (i + 1) % num_segments
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

def create_ball_valve_sample_port(location: Tuple[float, float, float],
                                  port_diameter: float = 0.025,
                                  **kwargs) -> SamplePort:
    """Create a ball valve sample port."""
    params = SamplePortParams(
        valve_type="ball",
        port_diameter=port_diameter,
        location=location,
        **kwargs
    )
    return SamplePort(params)


def create_isokinetic_sample_port(location: Tuple[float, float, float],
                                  port_diameter: float = 0.025,
                                  **kwargs) -> SamplePort:
    """Create an isokinetic sample port."""
    params = SamplePortParams(
        valve_type="ball",
        sample_type="isokinetic",
        port_diameter=port_diameter,
        location=location,
        **kwargs
    )
    return SamplePort(params)
