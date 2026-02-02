"""
Support and exhaust system assembly for air classification systems.

This module provides assembly of support structures (frames, legs),
silencers, and exhaust stacks for complete system installations.

Coordinate System (Y-up):
    - X: horizontal (width)
    - Y: vertical (height) - UP
    - Z: horizontal (depth)
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class SupportExhaustParams:
    """
    Parameters for support and exhaust system.
    
    Attributes:
        frame_width: Support frame width [m] (X direction)
        frame_depth: Support frame depth [m] (Z direction)
        frame_height: Support frame height [m] (Y direction - vertical)
        has_legs: Whether to include equipment legs
        leg_height: Leg height if included [m]
        num_platform_levels: Number of platform levels
        has_silencer: Whether to include silencer
        silencer_diameter: Silencer diameter [m]
        silencer_length: Silencer length [m]
        has_exhaust_stack: Whether to include exhaust stack
        stack_diameter: Stack diameter [m]
        stack_height: Stack height [m]
        center: Center position (x, y, z) [m]
    """
    frame_width: float = 2.5
    frame_depth: float = 2.0
    frame_height: float = 3.0
    has_legs: bool = True
    leg_height: float = 0.5
    num_legs: int = 4
    num_platform_levels: int = 1
    has_silencer: bool = True
    silencer_diameter: float = 0.3
    silencer_length: float = 1.0
    has_exhaust_stack: bool = True
    stack_diameter: float = 0.3
    stack_height: float = 4.0
    stack_cap_type: str = "conical"
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    @property
    def total_height(self) -> float:
        """Total system height including stack [m]."""
        h = self.frame_height
        if self.has_legs:
            h += self.leg_height
        if self.has_exhaust_stack:
            h += self.stack_height
        return h


class SupportExhaustAssembly:
    """
    Complete support and exhaust system assembly.
    
    Assembles structural frames, legs, silencers, and exhaust stacks.
    Uses Y-up coordinate system (Y is vertical).
    """
    
    def __init__(self, params: SupportExhaustParams):
        """
        Initialize support and exhaust assembly.
        
        Args:
            params: System parameters
        """
        self.params = params
        self._components: Dict[str, Any] = {}
        self._vertices: Optional[np.ndarray] = None
        self._indices: Optional[np.ndarray] = None
        self._create_components()
    
    def _create_components(self):
        """Create all support and exhaust components."""
        # Lazy imports
        from ..components.supports import (
            create_standard_frame, create_tubular_legs
        )
        from ..components.silencer import create_absorptive_silencer
        from ..components.exhaust_stack import create_standard_exhaust_stack
        
        p = self.params
        cx, cy, cz = p.center
        
        # Equipment legs at base (Y is vertical)
        if p.has_legs:
            legs = create_tubular_legs(
                num_legs=p.num_legs,
                height=p.leg_height,
                mounting_diameter=min(p.frame_width, p.frame_depth) * 0.9,
                center=(cx, cy, cz)
            )
            self._components['legs'] = legs
            frame_base_y = cy + p.leg_height
        else:
            frame_base_y = cy
        
        # Structural frame (sits on top of legs)
        platform_levels = [
            p.frame_height * (i + 1) / (p.num_platform_levels + 1)
            for i in range(p.num_platform_levels)
        ]
        platform_levels.append(p.frame_height)
        
        frame = create_standard_frame(
            width=p.frame_width,
            depth=p.frame_depth,
            height=p.frame_height,
            platform_levels=platform_levels,
            center=(cx, frame_base_y, cz)
        )
        self._components['frame'] = frame
        
        # Calculate top of frame (Y coordinate)
        frame_top_y = frame_base_y + p.frame_height
        
        # Silencer (horizontal along X, near top of frame)
        if p.has_silencer:
            silencer_y = frame_top_y - 0.3
            silencer = create_absorptive_silencer(
                diameter=p.silencer_diameter,
                length=p.silencer_length,
                center=(cx + p.frame_width/2 + p.silencer_length/2 + 0.1, silencer_y, cz),
                direction=(1, 0, 0)  # Horizontal along X
            )
            self._components['silencer'] = silencer
        
        # Exhaust stack (vertical along Y, on top of frame)
        if p.has_exhaust_stack:
            stack = create_standard_exhaust_stack(
                diameter=p.stack_diameter,
                height=p.stack_height,
                cap_type=p.stack_cap_type,
                center=(cx, frame_top_y, cz)
            )
            self._components['exhaust_stack'] = stack
    
    def get_component(self, name: str) -> Any:
        """Get a specific component by name."""
        return self._components.get(name)
    
    def get_component_names(self) -> List[str]:
        """Get list of all component names."""
        return list(self._components.keys())
    
    def build_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build combined mesh from all components.
        
        Returns:
            Tuple of (vertices, indices)
        """
        all_vertices = []
        all_indices = []
        vertex_offset = 0
        
        for name, component in self._components.items():
            verts = component.vertices
            inds = component.indices
            
            all_vertices.append(verts)
            all_indices.append(inds + vertex_offset)
            vertex_offset += len(verts)
        
        if all_vertices:
            self._vertices = np.vstack(all_vertices).astype(np.float32)
            self._indices = np.concatenate(all_indices).astype(np.int32)
        else:
            self._vertices = np.array([], dtype=np.float32).reshape(0, 3)
            self._indices = np.array([], dtype=np.int32)
        
        return self._vertices, self._indices
    
    @property
    def vertices(self) -> np.ndarray:
        """Get combined mesh vertices."""
        if self._vertices is None:
            self.build_mesh()
        return self._vertices
    
    @property
    def indices(self) -> np.ndarray:
        """Get combined mesh indices."""
        if self._indices is None:
            self.build_mesh()
        return self._indices
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box of entire system."""
        verts = self.vertices
        if len(verts) == 0:
            return np.zeros(3), np.zeros(3)
        return verts.min(axis=0), verts.max(axis=0)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of support and exhaust system."""
        bounds_min, bounds_max = self.get_bounds()
        
        return {
            'frame_dimensions': (self.params.frame_width, self.params.frame_depth, self.params.frame_height),
            'total_height_m': self.params.total_height,
            'has_legs': self.params.has_legs,
            'leg_height_m': self.params.leg_height if self.params.has_legs else 0,
            'has_silencer': self.params.has_silencer,
            'silencer_length_m': self.params.silencer_length if self.params.has_silencer else 0,
            'has_exhaust_stack': self.params.has_exhaust_stack,
            'stack_height_m': self.params.stack_height if self.params.has_exhaust_stack else 0,
            'num_components': len(self._components),
            'total_vertices': len(self.vertices),
            'total_triangles': len(self.indices) // 3,
            'bounds_min': list(bounds_min),
            'bounds_max': list(bounds_max),
        }
    
    def to_warp_mesh(self) -> Any:
        """Create a Warp mesh object."""
        if wp is None:
            raise ImportError("NVIDIA Warp is required")
        return wp.Mesh(
            points=wp.array(self.vertices, dtype=wp.vec3),
            indices=wp.array(self.indices.flatten(), dtype=wp.int32)
        )
    
    def print_summary(self):
        """Print a summary of the support and exhaust system."""
        summary = self.get_system_summary()
        
        print("=" * 60)
        print("SUPPORT & EXHAUST SYSTEM SUMMARY")
        print("=" * 60)
        print(f"Frame: {summary['frame_dimensions'][0]:.1f} x {summary['frame_dimensions'][1]:.1f} x {summary['frame_dimensions'][2]:.1f} m")
        print(f"Total Height: {summary['total_height_m']:.1f} m")
        print("-" * 60)
        print("COMPONENTS:")
        if summary['has_legs']:
            print(f"  Legs: {self.params.num_legs} x {summary['leg_height_m']:.2f} m height")
        if summary['has_silencer']:
            print(f"  Silencer: {summary['silencer_length_m']:.2f} m length")
        if summary['has_exhaust_stack']:
            print(f"  Exhaust Stack: {summary['stack_height_m']:.1f} m height")
        print("-" * 60)
        print("GEOMETRY:")
        print(f"  Total Components: {summary['num_components']}")
        print(f"  Total Vertices: {summary['total_vertices']}")
        print(f"  Total Triangles: {summary['total_triangles']}")
        print("=" * 60)


# Factory functions

def create_standard_support_exhaust(frame_height: float = 3.0,
                                    stack_height: float = 4.0,
                                    **kwargs) -> SupportExhaustAssembly:
    """
    Create a standard support and exhaust system.
    
    Args:
        frame_height: Frame height [m] (Y direction - vertical)
        stack_height: Exhaust stack height [m]
        **kwargs: Additional parameters
        
    Returns:
        SupportExhaustAssembly instance
    """
    params = SupportExhaustParams(
        frame_height=frame_height,
        stack_height=stack_height,
        **kwargs
    )
    return SupportExhaustAssembly(params)


def create_compact_support(frame_width: float = 1.5,
                           frame_depth: float = 1.5,
                           frame_height: float = 2.0,
                           **kwargs) -> SupportExhaustAssembly:
    """
    Create a compact support system for smaller equipment.
    
    Args:
        frame_width: Frame width [m] (X direction)
        frame_depth: Frame depth [m] (Z direction)
        frame_height: Frame height [m] (Y direction - vertical)
        **kwargs: Additional parameters
        
    Returns:
        SupportExhaustAssembly instance
    """
    params = SupportExhaustParams(
        frame_width=frame_width,
        frame_depth=frame_depth,
        frame_height=frame_height,
        has_silencer=False,
        has_exhaust_stack=True,
        stack_height=2.0,
        **kwargs
    )
    return SupportExhaustAssembly(params)


def create_industrial_support(frame_width: float = 4.0,
                              frame_depth: float = 3.0,
                              frame_height: float = 5.0,
                              **kwargs) -> SupportExhaustAssembly:
    """
    Create an industrial-scale support and exhaust system.
    
    Args:
        frame_width: Frame width [m] (X direction)
        frame_depth: Frame depth [m] (Z direction)
        frame_height: Frame height [m] (Y direction - vertical)
        **kwargs: Additional parameters
        
    Returns:
        SupportExhaustAssembly instance
    """
    params = SupportExhaustParams(
        frame_width=frame_width,
        frame_depth=frame_depth,
        frame_height=frame_height,
        leg_height=0.8,
        num_platform_levels=2,
        has_silencer=True,
        silencer_diameter=0.5,
        silencer_length=1.5,
        has_exhaust_stack=True,
        stack_diameter=0.5,
        stack_height=8.0,
        **kwargs
    )
    return SupportExhaustAssembly(params)
