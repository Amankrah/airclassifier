"""
Silencer / muffler components for air classification systems.

This module provides acoustic silencer geometries for noise
reduction in air handling systems.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class SilencerParams:
    """
    Parameters for acoustic silencer/muffler.
    
    Attributes:
        silencer_type: Type ("absorptive", "reactive", "combination")
        diameter: Duct diameter [m]
        length: Silencer length [m]
        num_splitters: Number of splitter baffles
        splitter_thickness: Splitter thickness [m]
        absorption_material: Absorption material type
        insertion_loss: Design insertion loss [dB]
        shell_thickness: Outer shell thickness [m]
        center: Center position (x, y, z) [m]
        direction: Flow direction (dx, dy, dz)
    """
    silencer_type: str = "absorptive"
    diameter: float = 0.3
    length: float = 1.0
    num_splitters: int = 0
    splitter_thickness: float = 0.05
    splitter_gap: float = 0.08
    absorption_material: str = "mineral_wool"
    insertion_loss: float = 15.0  # dB
    shell_thickness: float = 0.003
    shell_diameter: float = None  # Auto-calculated
    flanged: bool = True
    flange_width: float = 0.05
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    
    def __post_init__(self):
        """Calculate shell diameter if not specified."""
        if self.shell_diameter is None:
            # Shell is larger to accommodate absorption
            self.shell_diameter = self.diameter * 1.5
    
    @property
    def inner_radius(self) -> float:
        """Inner duct radius [m]."""
        return self.diameter / 2
    
    @property
    def outer_radius(self) -> float:
        """Outer shell radius [m]."""
        return self.shell_diameter / 2
    
    @property
    def direction_normalized(self) -> Tuple[float, float, float]:
        """Normalized direction vector."""
        d = np.array(self.direction)
        return tuple(d / np.linalg.norm(d))
    
    def get_pressure_drop(self, velocity: float) -> float:
        """
        Estimate pressure drop through silencer [Pa].
        
        Args:
            velocity: Air velocity [m/s]
            
        Returns:
            Pressure drop [Pa]
        """
        # Simplified estimate - silencers typically add 50-200 Pa
        rho = 1.2  # kg/m³ air
        K = 1.5 + 0.5 * self.num_splitters  # Loss coefficient
        return 0.5 * rho * velocity**2 * K


class Silencer:
    """
    Acoustic silencer/muffler geometry.
    
    Generates mesh for inline duct silencers with various
    configurations for noise reduction.
    """
    
    def __init__(self, params: SilencerParams):
        """
        Initialize silencer.
        
        Args:
            params: Silencer parameters
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
        Generate triangular mesh for the silencer.
        
        Args:
            num_segments: Circumferential segments
            
        Returns:
            Tuple of (vertices, indices, normals)
        """
        all_vertices = []
        all_indices = []
        all_normals = []
        
        p = self.params
        center = np.array(p.center)
        direction = np.array(p.direction_normalized)
        
        # Create local coordinate system
        if abs(direction[2]) < 0.9:
            perp1 = np.cross(direction, [0, 0, 1])
        else:
            perp1 = np.cross(direction, [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)
        
        # Outer shell
        self._add_shell(all_vertices, all_indices, all_normals,
                       center, direction, perp1, perp2, num_segments)
        
        # Inner duct (perforated)
        self._add_inner_duct(all_vertices, all_indices, all_normals,
                            center, direction, perp1, perp2, num_segments)
        
        # End caps / transitions
        self._add_end_transitions(all_vertices, all_indices, all_normals,
                                  center, direction, perp1, perp2, num_segments)
        
        # Flanges if specified
        if p.flanged:
            self._add_flanges(all_vertices, all_indices, all_normals,
                            center, direction, perp1, perp2, num_segments)
        
        # Splitter baffles if specified
        if p.num_splitters > 0:
            self._add_splitters(all_vertices, all_indices, all_normals,
                               center, direction, perp1, perp2, num_segments)
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def _add_shell(self, all_vertices, all_indices, all_normals,
                   center, direction, perp1, perp2, num_segments):
        """Add outer shell cylinder."""
        p = self.params
        base_idx = len(all_vertices)
        
        start = center - (p.length / 2) * direction
        
        for t in [0, 1]:
            pos = start + t * p.length * direction
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                offset = p.outer_radius * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
                all_vertices.append(list(pos + offset))
                n = np.cos(theta) * perp1 + np.sin(theta) * perp2
                all_normals.append(list(n))
        
        for i in range(num_segments):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % num_segments
            i2 = base_idx + num_segments + i
            i3 = base_idx + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
    
    def _add_inner_duct(self, all_vertices, all_indices, all_normals,
                        center, direction, perp1, perp2, num_segments):
        """Add inner perforated duct."""
        p = self.params
        base_idx = len(all_vertices)
        
        start = center - (p.length / 2) * direction
        
        for t in [0, 1]:
            pos = start + t * p.length * direction
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                offset = p.inner_radius * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
                all_vertices.append(list(pos + offset))
                # Normal points inward for inner surface
                n = -(np.cos(theta) * perp1 + np.sin(theta) * perp2)
                all_normals.append(list(n))
        
        for i in range(num_segments):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % num_segments
            i2 = base_idx + num_segments + i
            i3 = base_idx + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i1, i2])
            all_indices.extend([i1, i3, i2])
    
    def _add_end_transitions(self, all_vertices, all_indices, all_normals,
                             center, direction, perp1, perp2, num_segments):
        """Add end cap / transition rings."""
        p = self.params
        
        for end_mult in [-1, 1]:
            end_pos = center + (p.length / 2) * end_mult * direction
            base_idx = len(all_vertices)
            
            # Annular ring connecting inner to outer
            for radius in [p.inner_radius, p.outer_radius]:
                for i in range(num_segments):
                    theta = 2 * np.pi * i / num_segments
                    offset = radius * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
                    all_vertices.append(list(end_pos + offset))
                    all_normals.append(list(end_mult * direction))
            
            for i in range(num_segments):
                i0 = base_idx + i
                i1 = base_idx + (i + 1) % num_segments
                i2 = base_idx + num_segments + i
                i3 = base_idx + num_segments + (i + 1) % num_segments
                if end_mult > 0:
                    all_indices.extend([i0, i1, i2])
                    all_indices.extend([i1, i3, i2])
                else:
                    all_indices.extend([i0, i2, i1])
                    all_indices.extend([i1, i2, i3])
    
    def _add_flanges(self, all_vertices, all_indices, all_normals,
                     center, direction, perp1, perp2, num_segments):
        """Add mounting flanges at ends."""
        p = self.params
        flange_r = p.inner_radius + p.flange_width
        flange_thick = 0.015
        
        for end_mult in [-1, 1]:
            end_pos = center + (p.length / 2) * end_mult * direction
            base_idx = len(all_vertices)
            
            # Flange ring
            for t in [0, flange_thick]:
                flange_pos = end_pos + t * end_mult * direction
                for i in range(num_segments):
                    theta = 2 * np.pi * i / num_segments
                    offset = flange_r * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
                    all_vertices.append(list(flange_pos + offset))
                    n = np.cos(theta) * perp1 + np.sin(theta) * perp2
                    all_normals.append(list(n))
            
            for i in range(num_segments):
                i0 = base_idx + i
                i1 = base_idx + (i + 1) % num_segments
                i2 = base_idx + num_segments + i
                i3 = base_idx + num_segments + (i + 1) % num_segments
                all_indices.extend([i0, i2, i1])
                all_indices.extend([i1, i2, i3])
    
    def _add_splitters(self, all_vertices, all_indices, all_normals,
                       center, direction, perp1, perp2, num_segments):
        """Add splitter baffles."""
        p = self.params
        
        start = center - (p.length / 2) * direction
        
        for splitter_idx in range(p.num_splitters):
            # Position splitters across the duct
            offset_frac = (splitter_idx + 1) / (p.num_splitters + 1) - 0.5
            splitter_offset = p.inner_radius * 2 * offset_frac * perp1
            
            base_idx = len(all_vertices)
            
            # Splitter as thin box
            half_thick = p.splitter_thickness / 2
            half_height = p.inner_radius * 0.8
            
            # Vertices for splitter box
            corners = []
            for t in [0, p.length]:
                for ht in [-half_height, half_height]:
                    for th in [-half_thick, half_thick]:
                        pos = start + t * direction + splitter_offset + ht * perp2 + th * perp1
                        corners.append(list(pos))
            
            for corner in corners:
                all_vertices.append(corner)
                all_normals.append(list(perp1))  # Simplified normal
            
            # Create box faces (simplified)
            # Front face
            all_indices.extend([base_idx, base_idx + 1, base_idx + 3])
            all_indices.extend([base_idx, base_idx + 3, base_idx + 2])
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box."""
        verts = self.vertices
        return verts.min(axis=0), verts.max(axis=0)
    
    def get_insertion_loss(self, frequency: float = 500) -> float:
        """
        Get estimated insertion loss at given frequency [dB].
        
        Args:
            frequency: Sound frequency [Hz]
            
        Returns:
            Insertion loss [dB]
        """
        p = self.params
        # Simplified frequency-dependent calculation
        # Maximum attenuation around 250-1000 Hz for absorptive silencers
        if p.silencer_type == "absorptive":
            peak_freq = 500
            # Bell curve around peak
            loss = p.insertion_loss * np.exp(-((np.log10(frequency) - np.log10(peak_freq))**2) / 0.5)
            return max(loss, p.insertion_loss * 0.3)
        else:
            return p.insertion_loss
    
    def to_warp_mesh(self) -> Any:
        """Create a Warp mesh object."""
        if wp is None:
            raise ImportError("NVIDIA Warp is required")
        return wp.Mesh(
            points=wp.array(self.vertices, dtype=wp.vec3),
            indices=wp.array(self.indices.flatten(), dtype=wp.int32)
        )


# Factory functions

def create_absorptive_silencer(diameter: float = 0.3,
                               length: float = 1.0,
                               insertion_loss: float = 15.0,
                               **kwargs) -> Silencer:
    """
    Create an absorptive (dissipative) silencer.
    
    Args:
        diameter: Duct diameter [m]
        length: Silencer length [m]
        insertion_loss: Target insertion loss [dB]
        **kwargs: Additional parameters
        
    Returns:
        Silencer instance
    """
    params = SilencerParams(
        silencer_type="absorptive",
        diameter=diameter,
        length=length,
        insertion_loss=insertion_loss,
        num_splitters=0,
        **kwargs
    )
    return Silencer(params)


def create_splitter_silencer(diameter: float = 0.5,
                             length: float = 1.5,
                             num_splitters: int = 2,
                             **kwargs) -> Silencer:
    """
    Create a splitter-type silencer for larger ducts.
    
    Args:
        diameter: Duct diameter [m]
        length: Silencer length [m]
        num_splitters: Number of splitter baffles
        **kwargs: Additional parameters
        
    Returns:
        Silencer instance
    """
    params = SilencerParams(
        silencer_type="absorptive",
        diameter=diameter,
        length=length,
        num_splitters=num_splitters,
        insertion_loss=20.0 + 5 * num_splitters,
        **kwargs
    )
    return Silencer(params)


def create_reactive_silencer(diameter: float = 0.3,
                             length: float = 0.6,
                             **kwargs) -> Silencer:
    """
    Create a reactive (expansion chamber) silencer.
    
    Args:
        diameter: Duct diameter [m]
        length: Silencer length [m]
        **kwargs: Additional parameters
        
    Returns:
        Silencer instance
    """
    params = SilencerParams(
        silencer_type="reactive",
        diameter=diameter,
        length=length,
        shell_diameter=diameter * 2.5,  # Larger expansion
        insertion_loss=10.0,
        **kwargs
    )
    return Silencer(params)
